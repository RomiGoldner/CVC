import pandas as pd
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix
import torch
from torch_geometric.data import Data
import numpy as np
import faiss
from sklearn.neighbors import NearestNeighbors
from igraph import Graph
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from tensorflow.keras import Model
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap


# Compute similarity matrix
def cosine_similarity_pytorch(x):
    x_norm = x / x.norm(dim=-1, keepdim=True)
    similarity = torch.mm(x_norm, x_norm.t())
    return similarity

# Calculate graph metrics, returns node metrics and graph metrics
def calculate_graph_metrics(G, calculate_node_metrics):
    df_node_metrics = None

    # Calculate metrics
    if calculate_node_metrics:
        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        # eigenvector_centrality = nx.eigenvector_centrality(G)

        # Convert centrality measures to dataframes
        df_degree = pd.DataFrame.from_dict(degree_centrality, orient='index', columns=['degree_centrality'])
        df_closeness = pd.DataFrame.from_dict(closeness_centrality, orient='index', columns=['closeness_centrality'])
        df_betweenness = pd.DataFrame.from_dict(betweenness_centrality, orient='index', columns=['betweenness_centrality'])
        # df_eigenvector = pd.DataFrame.from_dict(eigenvector_centrality, orient='index', columns=['eigenvector_centrality'])

        # Merge centrality dataframes
        # df_node_metrics = pd.concat([df_degree, df_closeness, df_betweenness, df_eigenvector], axis=1)
        df_node_metrics = pd.concat([df_degree, df_closeness, df_betweenness], axis=1)

    density = nx.density(G)
    clustering_coefficient = nx.average_clustering(G)
    transitivity = nx.transitivity(G)
    connected_components = nx.number_connected_components(G)
    # Create DataFrame for graph metrics
    data = {'density': [density], 'clustering_coefficient': [clustering_coefficient],
            'transitivity': [transitivity], 'connected_components': [connected_components]}
    df_graph_metrics = pd.DataFrame(data)

    if calculate_node_metrics:
        return df_node_metrics, df_graph_metrics
    return df_graph_metrics

# create graph from embeddings, with GPU support
def create_graph(embeddings, similarity_function, threshold, device):
    # Convert numpy array to tensor and move to device
    embeddings = torch.from_numpy(embeddings).float().to(device)
    # Compute similarity matrix
    similarity_matrix = similarity_function(embeddings)
    # Create edge_index tensor
    edge_index = torch.tensor([[i, j] for i in range(similarity_matrix.shape[0]) for j in range(similarity_matrix.shape[1])
                               if similarity_matrix[i, j] > threshold], dtype=torch.long).t().contiguous()
    # Move edge_index to device
    edge_index = edge_index.to(device)
    # Create PyG graph
    graph = Data(x=embeddings, edge_index=edge_index)
    return graph

# create graph from embeddings using scipy
def create_graph_scipy(embeddings, threshold):
    # Calculate pairwise cosine distances
    distances = pdist(embeddings, metric='cosine')
    # Convert distances to similarities
    similarities = 1 - distances
    # Convert to square form
    similarity_matrix = squareform(similarities)
    # Apply threshold to create adjacency matrix
    adjacency_matrix = (similarity_matrix >= threshold).astype(int)
    # Convert to sparse format
    adjacency_matrix = coo_matrix(adjacency_matrix)
    # Convert to PyG format
    edge_index = torch.tensor(np.vstack(adjacency_matrix.nonzero()), dtype=torch.long)
    # Convert embeddings to PyTorch tensor
    embeddings = torch.tensor(embeddings, dtype=torch.float)
    # Create PyG graph
    graph = Data(x=embeddings, edge_index=edge_index)
    return graph

# create graph from embeddings using igraph
def create_knn_graph_igraph(embeddings, k):
    # Fit nearest neighbors model
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(embeddings)
    # Get the k nearest neighbors for each point
    distances, indices = nbrs.kneighbors(embeddings)
    # Define the edges and weights
    edges = [(i, indices[i, j]) for i in range(len(indices)) for j in range(1, k)]
    weights = [distances[i, j] for i in range(len(distances)) for j in range(1, k)]
    # Create an empty undirected Graph
    g = Graph(n=len(embeddings), edges=edges, directed=False)
    # Set edge weights
    g.es["weight"] = weights
    return g


def create_knn_graph_faiss_batch(embeddings, k, batch_size):
    # Convert the embeddings to float32 format
    embeddings = embeddings.astype('float32')
    # Normalize the embeddings so they all have the same length
    embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    # Build a FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    # Create an empty graph
    G = nx.Graph()
    # Iterate over the embeddings in batches
    for i in tqdm(range(0, embeddings.shape[0], batch_size)):
        # Get the current batch
        batch = embeddings[i:i+batch_size]
        # Add the batch to the index
        index.add(batch)
        # Find the most similar embeddings to the current embedding
        _, indices = index.search(batch, k+1)
        # Iterate over the most similar embeddings
        for j in range(indices.shape[0]):
            # Add the edges from this embedding to the others in its neighborhood
            for l in range(1, indices.shape[1]):
                G.add_edge(i+j, indices[j, l])
    return G

# nlist is the number of Voronoi cells (clusters) for the index.
# A larger nlist will give you more accurate results but will be slower.
# nprobe is the number of cells that FAISS inspects during a search.
# A larger nprobe makes the search more exhaustive and thus more accurate, but also slower.
def create_knn_graph_faiss_large_dataset_L2_dist(embeddings, k, nlist=100, nprobe=10):
    dimension = embeddings.shape[1]
    # Normalize the embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]
    # Ensure the embeddings are of type float32
    embeddings = embeddings.astype('float32')
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



# graph (created using faiss) to networkx
def faiss_to_networkx_L2(data, sequences):
    edge_list = data.edges()
    num_nodes = data.number_of_nodes()
    # Create a NetworkX graph
    G = nx.DiGraph()
    # Add nodes
    for i in range(num_nodes):
        # Use the sequence as a node attribute
        G.add_node(i, sequence=sequences[i])
    # Add edges
    for edge in edge_list:
        G.add_edge(edge[0], edge[1])
    return G


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


# process patient data (create graphs, calculate metrics, save graphs and metrics)
from typing import Tuple, Optional


def process_patient_embeddings(case_patient_embeddings_dict: dict, cases_dataframes_dict: dict, df_type: str, GRAPH_DIR: str, calculate_node_metrics: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    type = df_type.split('_')[0]
    # Initialize two empty dataframes to store the results
    node_results_df = pd.DataFrame()
    graph_results_df = pd.DataFrame()

    # Iterate over the patient dictionary
    for patient_id, patient_sample_list in tqdm(case_patient_embeddings_dict.items(), desc='Processing patients'):
        # Iterate over the list of dataframes for this patient
        for timepoint, patient_sample in tqdm(enumerate(patient_sample_list), total=len(patient_sample_list), desc=f'Processing timepoints for patient {patient_id}'):
            #  Create graph of embeddings
            embeddings = patient_sample.embeddings
            patient_graph = create_knn_graph_faiss_large_dataset(embeddings, 2)
            # Calculate graph metrics
            sequences = patient_sample.sequences_df['Sequences'].tolist()
            # Add v_family, v_gene, j_family, j_gene to the graph, from the cases_dataframes_dict dictionary
            #patient_graph= gu.faiss_to_networkx(patient_graph, sequences)
            patient_graph = faiss_to_networkx(patient_graph, cases_dataframes_dict[patient_id][timepoint])
            # export graph to dot
            export_graph_to_gexf(patient_graph, path_to_save=GRAPH_DIR, filename=f'{type}_patient_{patient_id}_timepoint_{timepoint}.gexf')
            if not calculate_node_metrics:
                graph_metrics_df = calculate_graph_metrics(patient_graph, calculate_node_metrics=calculate_node_metrics)
                graph_metrics_df['patient_id'] = patient_id
                graph_metrics_df['timepoint'] = timepoint
            else:
                node_metrics_df, graph_metrics_df = calculate_graph_metrics(patient_graph, calculate_node_metrics=calculate_node_metrics)
                # Add the patient id and timepoint to the metrics dataframes
                node_metrics_df['patient_id'] = patient_id
                node_metrics_df['timepoint'] = timepoint
                graph_metrics_df['patient_id'] = patient_id
                graph_metrics_df['timepoint'] = timepoint
                # Append node results
                node_results_df = node_results_df.append(node_metrics_df, ignore_index=True)

            # Append the graph results to the overall results dataframes
            graph_results_df = graph_results_df.append(graph_metrics_df, ignore_index=True)
    return graph_results_df, (node_results_df if calculate_node_metrics else None)



# plot metrics
def plot_metrics(graph_results_df, graph_results_df_name: str):
    # add to the title of its case or control by name of the input file. the first word before the underscore is the case or control
    type = graph_results_df_name.split('_')[0]

    # Line plot showing the evolution of graph density over time for each patient
    for patient_id in graph_results_df['patient_id'].unique():
        patient_data = graph_results_df[graph_results_df['patient_id'] == patient_id]
        plt.plot(patient_data['timepoint'], patient_data['density'], label=f'Patient {patient_id}')
    plt.xlabel('Timepoint')
    plt.ylabel('Graph Density')
    plt.title(f'Graph Density Over Time for {type} Patients')
    plt.legend()
    plt.show()

    # Bar plot showing the average clustering coefficient at each timepoint for each patient
    graph_results_df.groupby(['patient_id', 'timepoint'])['clustering_coefficient'].mean().unstack().plot(kind='bar')
    plt.xlabel('Patient ID')
    plt.ylabel('Average Clustering Coefficient')
    plt.title(f'Average Clustering Coefficient Over Time for {type} Patients')
    plt.show()

    # Bar plot showing the average transitivity at each timepoint for each patient
    graph_results_df.groupby(['patient_id', 'timepoint'])['transitivity'].mean().unstack().plot(kind='bar')
    plt.xlabel('Patient ID')
    plt.ylabel('Average Transitivity')
    plt.title(f'Average Transitivity Over Time for {type} Patients')
    plt.show()

    # Bar plot showing the average number of connected components at each timepoint for each patient
    graph_results_df.groupby(['patient_id', 'timepoint'])['connected_components'].mean().unstack().plot(kind='bar')
    plt.xlabel('Patient ID')
    plt.ylabel('Average Number of Connected Components')
    plt.title(f'Average Number of Connected Components Over Time for {type} Patients')
    plt.show()


# create graph embeddings
# layer sizes is a list of the number of hidden nodes in each layer of the GraphSAGE model
# batch size is the number of nodes per batch
# num samples is the number of neighbours sampled per node for the GraphSAGE model
def create_graph_embedding(graph, node_features, layer_sizes, batch_size, num_samples):
    # Load your graph data into a StellarGraph object
    G = StellarGraph(graph, node_features=node_features)
    # Create a generator
    generator = GraphSAGENodeGenerator(G, batch_size, num_samples)
    # Build the GraphSAGE model
    graphsage_model = GraphSAGE(
        layer_sizes=layer_sizes,
        generator=generator,
        bias=True,
        dropout=0.5,
    )
    # Create a new model that outputs embeddings instead of class predictions
    x_inp, x_out = graphsage_model.in_out_tensors()
    embedding_model = Model(inputs=x_inp, outputs=x_out)
    # Generate embeddings for each node
    node_ids = list(G.nodes())
    node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(node_ids, node_ids)
    node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)
    # Aggregate the node embeddings to create a graph-level embedding
    graph_embedding = np.mean(node_embeddings, axis=0)
    return graph_embedding


def convert_age_to_years(age_str):
    number, unit = age_str.split()
    number = int(number)
    if unit.lower().startswith('month'):
        number /= 12  # convert months to years
    return number


# create patient graph embeddings dictionary
def create_patient_graph_embeddings(case_patient_embeddings_dict: dict, cases_dataframes_dict: dict, df_type: str):
    type = df_type.split('_')[0]
    graph_embeddings_dict = {}
    # Iterate over the patient dictionary
    for patient_id, patient_sample_list in tqdm(case_patient_embeddings_dict.items(), desc='Processing patients'):
        # Iterate over the list of dataframes for this patient
        for timepoint, patient_sample in tqdm(enumerate(patient_sample_list), total=len(patient_sample_list), desc=f'Processing timepoints for patient {patient_id}'):
            # Apply the function to the 'Age' column
            cases_dataframes_dict[patient_id][timepoint]['Age_num'] = cases_dataframes_dict[patient_id][timepoint]['Age'].apply(convert_age_to_years)
            # Create patient graph
            patient_graph = create_patient_graph(patient_sample, cases_dataframes_dict, patient_id, timepoint, only_numerical_features=True)
            # export graph to dot
            # gu.export_graph_to_gexf(patient_graph, path_to_save=GRAPH_DIR, filename=f'{type}_patient_{patient_id}_timepoint_{timepoint}.gexf')
            # Create and store graph embeddings
            # extract node features from the graph
            node_features = pd.DataFrame.from_records([data for node, data in patient_graph.nodes(data=True)], index=[node for node, data in patient_graph.nodes(data=True)])
            layer_sizes = [50, 50]
            batch_size = 10
            num_samples = [10, 5]
            # create graph embeddings
            graph_embeddings_dict[f'{patient_id}_{timepoint}'] = create_graph_embedding(graph=patient_graph, node_features=node_features, layer_sizes=layer_sizes, batch_size=batch_size, num_samples=num_samples)
    return graph_embeddings_dict

def create_node_types_csv(df, node_type_column='j_gene'):
    # Make sure your dataframe index matches your node indices
    df = df.reset_index(drop=True)
    # Save node types as CSV
    df[[node_type_column]].to_csv('node_types.csv', header=False)


# create patient graph
def create_patient_graph(patient_sample, cases_dataframes_dict, patient_id, timepoint, only_numerical_features=False):
    #  Create graph of embeddings
    embeddings = patient_sample.embeddings
    patient_graph = create_knn_graph_faiss_large_dataset(embeddings, 2)
    # Add sequences and other attributes to the graph
    if only_numerical_features:
        patient_graph = faiss_to_networkx(patient_graph, cases_dataframes_dict[patient_id][timepoint][['Age_num']], only_numerical_features)
    else:
        patient_graph = faiss_to_networkx(patient_graph, cases_dataframes_dict[patient_id][timepoint], only_numerical_features)
    return patient_graph


# process patient graph metrics
def process_patient_graph_metrics(patient_graph, patient_id, timepoint, calculate_node_metrics=False):
    # Calculate graph metrics
    if not calculate_node_metrics:
        graph_metrics_df = calculate_graph_metrics(patient_graph, calculate_node_metrics=calculate_node_metrics)
        graph_metrics_df['patient_id'] = patient_id
        graph_metrics_df['timepoint'] = timepoint
    else:
        node_metrics_df, graph_metrics_df = calculate_graph_metrics(patient_graph, calculate_node_metrics=calculate_node_metrics)
        # Add the patient id and timepoint to the metrics dataframes
        node_metrics_df['patient_id'] = patient_id
        node_metrics_df['timepoint'] = timepoint
        graph_metrics_df['patient_id'] = patient_id
        graph_metrics_df['timepoint'] = timepoint
    return node_metrics_df if calculate_node_metrics else None, graph_metrics_df


def plot_cases_control(case_graph_embeddings_dict, control_graph_embeddings_dict, dim_reduction_type='UMAP', random_state=42):
    # extract embeddings and labels from your dictionaries
    case_embeddings = list(case_graph_embeddings_dict.values())
    case_labels = list(case_graph_embeddings_dict.keys())
    control_embeddings = list(control_graph_embeddings_dict.values())
    control_labels = list(control_graph_embeddings_dict.keys())
    embeddings = case_embeddings + control_embeddings
    labels = case_labels + control_labels
    # generate colors: 0 for case, 1 for control
    colors = [0]*len(case_embeddings) + [1]*len(control_embeddings)
    # transform data to 2D
    if dim_reduction_type=='UMAP':
        reduced_embeddings = umap.UMAP(random_state=random_state).fit_transform(embeddings)
    elif dim_reduction_type=='PCA':
        reduced_embeddings = PCA(n_components=2).fit_transform(embeddings)
    # create a scatter plot
    plt.figure(figsize=(10, 10))
    # create a custom color map
    cmap = mcolors.ListedColormap(['blue', 'red'])
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, cmap=cmap, s=15)
    # annotate points by label
    for label, (x, y) in zip(labels, reduced_embeddings):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    plt.gca().set_aspect('equal', 'datalim')
    # add a colorbar with custom tick labels
    colorbar = plt.colorbar(scatter, ticks=[0, 1])
    colorbar.set_ticklabels(['Case', 'Control'])
    plt.title(dim_reduction_type + ' : Cases and Controls', fontsize=16)
    plt.show()


def plot_timepoint_graph_embeddings(case_graph_embeddings_dict, ages_dict, type, dim_reduction_type, random_state=42):
    # extract embeddings and labels from your dictionary
    embeddings = list(case_graph_embeddings_dict.values())
    labels = list(case_graph_embeddings_dict.keys())
    # parse patient IDs from labels for coloring
    patient_ids = [label.split('_')[0] for label in labels]
    # use LabelEncoder to convert patient IDs to integers for coloring
    le = LabelEncoder()
    patient_ids_encoded = le.fit_transform(patient_ids)
    # fit UMAP and transform data to 2D
    if dim_reduction_type=='UMAP':
        umap_embeddings = umap.UMAP(random_state=random_state).fit_transform(embeddings)
    elif dim_reduction_type=='PCA':
        umap_embeddings = PCA(n_components=2).fit_transform(embeddings)
    elif dim_reduction_type=='TSNE':
        # extract embeddings and labels from your dictionary and convert embeddings to numpy array
        embeddings = np.array(list(case_graph_embeddings_dict.values()))
        labels = list(case_graph_embeddings_dict.keys())
        umap_embeddings = TSNE(n_components=2, perplexity=min(30, embeddings.shape[0]-1)).fit_transform(embeddings)
    # create a scatter plot
    plt.figure(figsize=(10, 10))
    # use the 'tab10' colormap
    cmap = plt.cm.get_cmap('Set3')
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=patient_ids_encoded, cmap=cmap, s=15)
    # label points by age
    for label, (x, y) in zip(labels, umap_embeddings):
        age = ages_dict[label].round(2)
        graph_label = f'{label} : {age}'
        plt.annotate(graph_label, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    plt.gca().set_aspect('equal', 'datalim')
    # add a colorbar
    colorbar = plt.colorbar(scatter, boundaries=np.arange(len(set(patient_ids))+1)-0.5)
    colorbar.set_ticks(np.arange(len(set(patient_ids))))
    colorbar.set_ticklabels(le.classes_)
    plt.title(dim_reduction_type + ' of ' + type + ' Patients 0-' + str(len(set(patient_ids))) +
              ', colored by age (graph - mean of nodes)', fontsize=16)
    plt.show()


def create_age_dict(cases_dataframes_dict):
    ages_dict = {}
    for patient_id, df_list in cases_dataframes_dict.items():
        for timepoint, df in enumerate(df_list):
            ages_dict[f'{patient_id}_{timepoint}'] = df['Age_num'].values[0]
    return ages_dict


# covert a dictionary of NetworkX graphs to a list of PyTorch Geometric Data objects
def convert_cvc_graph_to_pyg_data(graph_dict, type):
    data_list = []
    for key, graph_info in graph_dict.items():
        nx_graph = graph_info['graph']
        age_num = graph_info['age_num'].iloc[0]
        # Extract the timepoint from the key
        timepoint = int(key.split('_')[-1])
        # Convert the NetworkX graph to a PyTorch Geometric Data object
        data = from_networkx(nx_graph)
        # Create tensors for Age_num and timepoint features
        age_num_tensor = torch.full((data.num_nodes, 1), age_num, dtype=torch.float)
        timepoint_tensor = torch.full((data.num_nodes, 1), timepoint, dtype=torch.float)
        # Concatenate these tensors with the existing node features (if any)
        features_to_concat = [age_num_tensor, timepoint_tensor]
        if data.x is not None:
            features_to_concat.insert(0, data.x)
        data.x = torch.cat(features_to_concat, dim=1)
        # Create 3 labels - cases of timepoint 3, cases of timepoints 0-2, controls
        if timepoint == 3 and type == 'case':
            data.y = torch.tensor([1], dtype=torch.long)  # label
        elif timepoint < 3 and type == 'case':
            data.y = torch.tensor([2], dtype=torch.long)  # label
        else:
            data.y = torch.tensor([0], dtype=torch.long)  # label
        data_list.append(data)
    return data_list