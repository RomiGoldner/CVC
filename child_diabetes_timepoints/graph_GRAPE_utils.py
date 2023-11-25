import pandas as pd
import networkx as nx
import faiss
from igraph import Graph
from tqdm import tqdm
import os
import numpy as np
import ensmallen as ens
from child_diabetes_timepoints.graph_visualizer import GraphVisualizer
from grape.embedders import Node2VecSkipGramEnsmallen, Node2VecGloVeEnsmallen, Node2VecCBOWEnsmallen, BoostNEKarateClub
from grape.embedders import DeepWalkCBOWEnsmallen, DeepWalkSkipGramEnsmallen, DeepWalkGloVeEnsmallen
from grape import Graph as GRAPEGraph
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from child_diabetes_timepoints import graph_utils as gu



# using cosine similarity
def create_knn_graph_faiss_large_dataset(embeddings, k, nlist=100, nprobe=10):
    dimension = embeddings.shape[1]
    embeddings = embeddings.astype('float32')
    # Normalize the embeddings to use cosine similarity
    faiss.normalize_L2(embeddings)
    # # Ensure the embeddings are of type float32
    # embeddings = embeddings.astype('float32')
    # Build the FAISS index
    quantizer = faiss.IndexFlatL2(dimension)  # the quantizer defines the metric to use
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    assert not index.is_trained
    index.train(embeddings)
    assert index.is_trained
    index.add(embeddings)
    index.nprobe = nprobe  # find 2 most similar items
    # Find the k nearest neighbors for each point
    _, indices = index.search(embeddings, k+1)  # +1 because the point is its own neighbor
    # Define the edges
    edges = [(i, indices[i, j]) for i in range(len(indices)) for j in range(1, k+1)]
    # Create an empty undirected Graph
    g = Graph(len(embeddings), edges)
    return g


# graph (created using faiss) to networkx (undirected)
def faiss_to_networkx(g, df, only_numerical_features=False):
    # Create a NetworkX graph
    G = nx.Graph()
    # Add nodes
    for i in range(g.vcount()):
        # Get attributes for node i from the dataframe
        if only_numerical_features:
            attributes = df.iloc[i][['Age_num']].to_dict()
        else:
            attributes = df.iloc[i][['v_family', 'v_gene', 'j_family', 'j_gene', 'Sequences', 'Age']].to_dict()
        # Add node with attributes
        G.add_node(i, **attributes)
    # Add edges
    for edge in g.get_edgelist():
        G.add_edge(*edge)
    return G


# export graph to gexf format
def export_graph_to_gexf(G, path_to_save, filename):
    nx.write_gexf(G, os.path.join(path_to_save, filename))


def convert_age_to_years(age_str):
    number, unit = age_str.split()
    number = int(number)
    if unit.lower().startswith('month'):
        number /= 12  # convert months to years
    return number


def create_node_types_csv(df, node_type_column='j_gene'):
    # Make sure your dataframe index matches your node indices
    df = df.reset_index(drop=True)
    # Save node types as CSV
    df[[node_type_column]].to_csv('node_types.csv', header=False)


def create_GRAPE_embeddings_2(case_patient_embeddings_dict: dict, cases_dataframes_dict: dict, df_type: str, embedder_function = Node2VecSkipGramEnsmallen):
    type = df_type.split('_')[0]
    graph_embeddings_dict = {}
    # Iterate over the patient dictionary
    for patient_id, patient_sample_list in tqdm(case_patient_embeddings_dict.items(), desc='Processing patients'):
        # Iterate over the list of dataframes for this patient
        for timepoint, patient_sample in tqdm(enumerate(patient_sample_list), total=len(patient_sample_list), desc=f'Processing timepoints for patient {patient_id}'):
            #  Create graph of embeddings
            embeddings = patient_sample.embeddings
            patient_graph = create_knn_graph_faiss_large_dataset(embeddings, 2)
            patient_graph = faiss_to_networkx(patient_graph, cases_dataframes_dict[patient_id][timepoint])
            # Write the graph to a file
            edge_list_file = "graph.edgelist"
            nx.write_weighted_edgelist(patient_graph, edge_list_file)

            # Create a CSV file with node types
            node_types = pd.DataFrame.from_dict(nx.get_node_attributes(patient_graph, 'j_gene'), orient='index', columns=['node_type'])
            node_types.to_csv("node_types.csv")
            # Create a CSV file with unique node types
            pd.DataFrame(node_types['node_type'].unique(), columns=['node_type']).to_csv("unique_node_types.csv", index=False)

            # Create an Ensmallen graph from the edge list file and node types file
            ensmallen_graph = ens.Graph.from_csv(
                edge_path=edge_list_file,
                node_path="node_types.csv",  # path to the CSV file with node types
                node_type_path="unique_node_types.csv",
                node_list_node_types_column="node_type",
                directed=False,
                default_edge_type='links'
            )
            # Delete the temporary file
            os.remove(edge_list_file)
            os.remove('node_types.csv')
            os.remove('unique_node_types.csv')
            # Use Grape library to create embeddings
            embedder = embedder_function()
            #embedder = Node2VecSkipGramEnsmallen()
            # Use the Node2Vec SkipGram model from the Grape library to create embeddings
            embedding_result = embedder.fit_transform(ensmallen_graph)
            # Store the graph and the embeddings in the dictionary
            graph_embeddings_dict[f'{patient_id}_{timepoint}'] = {
                'graph': ensmallen_graph,
                'embeddings': embedding_result
            }
    return graph_embeddings_dict


def create_GRAPE_embeddings_timepoints(timepoint_embeddings_dict: dict, timepoint_sequences_dict: dict, embedder_function = Node2VecSkipGramEnsmallen):
    print('Creating GRAPE embeddings')
    graph_embeddings_dict = {}
    # Iterate over the timepoint dictionary
    for timepoint, embeddings in tqdm(timepoint_embeddings_dict.items(), desc='Processing timepoints'):
        sequences_df = timepoint_sequences_dict[timepoint]
        # Create graph of embeddings
        patient_graph = create_knn_graph_faiss_large_dataset(embeddings, 2)
        patient_graph = faiss_to_networkx(patient_graph, sequences_df)
        # Write the graph to a file
        edge_list_file = "graph.edgelist"
        nx.write_weighted_edgelist(patient_graph, edge_list_file)
        # Create a CSV file with node types
        node_types = pd.DataFrame.from_dict(nx.get_node_attributes(patient_graph, 'j_gene'), orient='index', columns=['node_type'])
        node_types.to_csv("node_types.csv")
        # Create a CSV file with unique node types
        pd.DataFrame(node_types['node_type'].unique(), columns=['node_type']).to_csv("unique_node_types.csv", index=False)
        # Create an Ensmallen graph from the edge list file and node types file
        ensmallen_graph = ens.Graph.from_csv(
            edge_path=edge_list_file,
            node_path="node_types.csv",  # path to the CSV file with node types
            node_type_path="unique_node_types.csv",
            node_list_node_types_column="node_type",
            directed=False,
            default_edge_type='links'
        )
        # Delete the temporary file
        os.remove(edge_list_file)
        os.remove('node_types.csv')
        os.remove('unique_node_types.csv')

        # Use Grape library to create embeddings
        embedder = embedder_function()
        # Use the Node2Vec SkipGram model from the Grape library to create embeddings
        embedding_result = embedder.fit_transform(ensmallen_graph)
        # Store the graph and the embeddings in the dictionary
        graph_embeddings_dict[f'{timepoint}'] = {
            'graph': ensmallen_graph,
            'embeddings': embedding_result
        }
    return graph_embeddings_dict


# plot node types using UMAP
def plot_node_types_umap(graph, embeddings, k=13, plot_type='ALL', decomposition_method='UMAP'):
    # create a GraphVisualizer object
    visualizer = GraphVisualizer(graph, decomposition_method=decomposition_method)
    # fit the nodes with the embeddings
    visualizer.fit_nodes(embeddings)
    # plot the node types
    if plot_type == 'NODE_TYPES':
        visualizer.plot_node_types(k=k)
    elif plot_type == 'ALL':
        visualizer.fit_and_plot_all(embeddings)


# concatenate embeddings and sequences of patient by timepoint
def create_timepoint_embeddings(case_patient_embeddings_dict):
    # Initialize new dictionaries
    timepoint_embeddings_dict = {}
    timepoint_sequences_dict = {}
    # Iterate over case_patient_embeddings_dict
    for patient_id, patient_sample_list in case_patient_embeddings_dict.items():
        # Iterate over the list of patient_samples
        for timepoint, patient_sample in enumerate(patient_sample_list):
            # Append the embeddings and sequences to the appropriate lists in their respective dicts
            if timepoint in timepoint_embeddings_dict:
                timepoint_embeddings_dict[timepoint] = np.concatenate(
                    (timepoint_embeddings_dict[timepoint], patient_sample.embeddings),
                    axis=0)
                timepoint_sequences_dict[timepoint] = pd.concat(
                    [timepoint_sequences_dict[timepoint], patient_sample.sequences_df])
            else:
                timepoint_embeddings_dict[timepoint] = patient_sample.embeddings
                timepoint_sequences_dict[timepoint] = patient_sample.sequences_df
    return timepoint_embeddings_dict, timepoint_sequences_dict


# combine case and control dictionaries
def combine_case_control(case_dict, control_dict):
    combined_dict = {}
    # Add cases
    for key, value in case_dict.items():
        combined_dict['case_' + key] = {"label": 1, "embeddings": value}
    # Add controls
    for key, value in control_dict.items():
        combined_dict['control_' + key] = {"label": 0, "embeddings": value}
    return combined_dict


# create patient graph
def create_GRAPE_patient_graph(embeddings, cases_dataframes_dict, only_numerical_features=False):
    patient_graph = create_knn_graph_faiss_large_dataset(embeddings, 2)
    # Add sequences and other attributes to the graph
    if only_numerical_features:
        patient_graph = faiss_to_networkx(patient_graph, cases_dataframes_dict[['Age_num']], only_numerical_features)
    else:
        # patient_graph = faiss_to_networkx(patient_graph, cases_dataframes_dict[patient_id][timepoint], only_numerical_features)
        patient_graph = faiss_to_networkx(patient_graph, cases_dataframes_dict, only_numerical_features)
    return patient_graph

def create_cvc_graph_dict(case_patient_embeddings_dict: dict, cases_dataframes_dict: dict, df_type: str):
    type = df_type.split('_')[0]
    graph_embeddings_dict = {}
    # Iterate over the patient dictionary
    for patient_id, patient_sample_list in tqdm(case_patient_embeddings_dict.items(), desc='Processing patients'):
        # Iterate over the list of dataframes for this patient
        for timepoint, patient_sample in tqdm(enumerate(patient_sample_list), total=len(patient_sample_list), desc=f'Processing timepoints for patient {patient_id}'):
            #  Create graph of embeddings
            embeddings = patient_sample.embeddings
            patient_graph = create_knn_graph_faiss_large_dataset(embeddings, 2)
            patient_graph = faiss_to_networkx(patient_graph, cases_dataframes_dict[patient_id][timepoint])
            # calculate age_num for the patient
            age_num = cases_dataframes_dict[int(patient_id)][int(timepoint)]['Age'].apply(convert_age_to_years)
            # Add to dictionary
            graph_embeddings_dict[f'{patient_id}_{timepoint}'] = {
                'graph': patient_graph,
                'embeddings': embeddings,
                'age_num': age_num
            }
    return graph_embeddings_dict


# # create patient graph embeddings using the GraphSAGE library using the node2vec embeddings created by GRAPE
# def create_GRAPE_patient_graph_embeddings(patient_node_embeddings_dict, patient_dataframe_dict, label, str_label, only_numerical_features=True):
#     graph_embeddings_dict = {}
#     prefix = str_label
#     # Iterate over the patient dictionary
#     for patient_timepoint, info in tqdm(patient_node_embeddings_dict.items(), desc='Processing patients'):
#         # Separate prefix, patient_id, and timepoint
#         patient_id, timepoint = patient_timepoint.split("_")
#         # embedding_result = info["embeddings"]['embeddings']
#         embedding_result = info["embeddings"]
#         # Extract node embeddings
#         node_embeddings_array = np.array([emb for emb in embedding_result.get_all_node_embedding()])
#         # Take the mean of the node embeddings
#         avg_node_embedding = np.mean(node_embeddings_array, axis=0)
#         # Get the corresponding data
#         patient_data = patient_dataframe_dict[int(patient_id)][int(timepoint)]
#         # Create patient graph
#         patient_graph = create_GRAPE_patient_graph(avg_node_embedding, patient_data)
#         node_features = pd.DataFrame.from_records([data for node, data in patient_graph.nodes(data=True)], index=[node for node, data in patient_graph.nodes(data=True)])
#         layer_sizes = [50, 50]
#         batch_size = 10
#         num_samples = [10, 5]
#         # Create graph embeddings
#         graph_embeddings_dict[f'{prefix}_{patient_id}_{timepoint}'] = {
#             'embedding': gu.create_graph_embedding(graph=patient_graph, node_features=node_features, layer_sizes=layer_sizes, batch_size=batch_size, num_samples=num_samples), # Customize as needed
#             'label': label
#         }
#     return graph_embeddings_dict

def create_GRAPE_patient_graph_embeddings(patient_node_embeddings_dict, patient_dataframe_dict, label, str_label, only_numerical_features=True, create_embedding=False):
    graph_dict = {}
    prefix = str_label
    # Iterate over the patient dictionary
    for patient_timepoint, info in tqdm(patient_node_embeddings_dict.items(), desc='Processing patients'):
        # Separate patient_id, and timepoint
        patient_id, timepoint = patient_timepoint.split("_")
        # Extract node embeddings
        embedding_result = info["embeddings"]
        node_embeddings_array = np.array([emb for emb in embedding_result.get_all_node_embedding()])
        # Take the mean of the node embeddings
        avg_node_embedding = np.mean(node_embeddings_array, axis=0)
        # Get the corresponding data
        patient_data = patient_dataframe_dict[int(patient_id)][int(timepoint)]
        # Apply the function to the 'Age' column
        patient_dataframe_dict[int(patient_id)][int(timepoint)]['Age_num'] = patient_dataframe_dict[int(patient_id)][int(timepoint)]['Age'].apply(convert_age_to_years)
        # Create patient graph
        patient_graph = create_GRAPE_patient_graph(avg_node_embedding, patient_data, only_numerical_features)
        if create_embedding:
            node_features = pd.DataFrame.from_records([data for node, data in patient_graph.nodes(data=True)], index=[node for node, data in patient_graph.nodes(data=True)])
            layer_sizes = [50, 50]
            batch_size = 10
            num_samples = [10, 5]
            # Create graph embeddings
            graph_dict[f'{prefix}_{patient_id}_{timepoint}'] = {
                'embedding': gu.create_graph_embedding(graph=patient_graph, node_features=node_features, layer_sizes=layer_sizes, batch_size=batch_size, num_samples=num_samples), # Customize as needed
                'label': label
            }
        else:
            graph_dict[f'{prefix}_{patient_id}_{timepoint}'] = patient_graph
    return graph_dict


#convert a dictionary of NetworkX graphs (used GRAPE) to a list of PyTorch Geometric Data objects
def convert_to_pyg_data(graph_dict):
    data_list = []
    for key, nx_graph in graph_dict.items():
        # Extract the timepoint from the key
        timepoint = int(key.split('_')[-1])

        # Convert the NetworkX graph to a PyTorch Geometric Data object
        data = from_networkx(nx_graph)
        # Create tensors for Age_num and timepoint features
        age_num_tensor = data.Age_num.unsqueeze(1)
        timepoint_tensor = torch.full((data.num_nodes, 1), timepoint, dtype=torch.float)
        # Concatenate these tensors with the existing node features (if any)
        features_to_concat = [age_num_tensor, timepoint_tensor]
        if data.x is not None:
            features_to_concat.insert(0, data.x)
        data.x = torch.cat(features_to_concat, dim=1)
        # create 3 labels - cases of timepoint 3, cases of timepoins 0-2, controls
        if timepoint == 3 and key.split('_')[0] == 'case':
            data.y = torch.tensor([1], dtype=torch.long)  # label
        elif timepoint < 3 and key.split('_')[0] == 'case':
            data.y = torch.tensor([2], dtype=torch.long)  # label
        else:
            data.y = torch.tensor([0], dtype=torch.long)  # label

        #data.y = torch.tensor([int(key.split('_')[0] == 'case')], dtype=torch.long)  # label
        del data.Age_num
        data_list.append(data)
    return data_list