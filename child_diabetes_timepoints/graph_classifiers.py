import torch
import random
from tqdm.auto import tqdm
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATConv, TopKPooling
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


def custom_collate_fn(data_list):
    print("in custom collate")
    batch = Batch()

    # These lists will store data from all graphs for each attribute
    batch.edge_index = torch.cat([data.edge_index for data in data_list], dim=1)
    batch.x = torch.cat([data.x for data in data_list], dim=0)
    batch.y = torch.stack([data.y for data in data_list])

    batch.v_family = torch.stack([data.v_family for data in data_list])
    batch.v_gene = torch.stack([data.v_gene for data in data_list])
    batch.j_family = torch.stack([data.j_family for data in data_list])
    batch.j_gene = torch.stack([data.j_gene for data in data_list])
    batch.Sequences = torch.stack([data.Sequences for data in data_list])
    batch.Age = torch.stack([data.Age for data in data_list])

    # Compute batch vector for batch-wise graph separation
    batch.batch = []
    for i, data in enumerate(data_list):
        batch.batch.append(torch.full((data.x.size(0), ), i, dtype=torch.long))
    batch.batch = torch.cat(batch.batch, dim=0)

    # Compute the cumulative sum for `ptr` (pointer to graph indices)
    cumsum_nodes = 0
    batch.ptr = [cumsum_nodes]
    for data in data_list:
        cumsum_nodes += data.num_nodes
        batch.ptr.append(cumsum_nodes)
    batch.ptr = torch.tensor(batch.ptr)
    return batch


def my_train_test_split(
        combined_data_list, combined_patient_id_list, combined_label_list, test_size=0.2, random_state=22):
    patient_num_data_points = 4

    num_patients = len(combined_patient_id_list) // patient_num_data_points
    num_test_patients = int(num_patients * test_size)
    print(f'Number of test patients: {num_test_patients}')

    case_patient_ids = [x for x, y in zip(combined_patient_id_list, combined_label_list) if y == 1]
    control_patient_ids = [x for x, y in zip(combined_patient_id_list, combined_label_list) if y == 0]
    unique_case_ids = list(set(case_patient_ids))
    unique_control_ids = list(set(control_patient_ids))

    # Calculate the number of patients to include in the test set
    num_test_cases = int(len(unique_case_ids) * test_size)
    num_test_controls = int(len(unique_control_ids) * test_size)

    # Shuffle the patient IDs to randomize the selection
    random.seed(random_state)
    random.shuffle(unique_case_ids)
    random.shuffle(unique_control_ids)

    # Split the patient IDs into test and train sets
    test_case_ids = unique_case_ids[:num_test_cases]
    train_case_ids = unique_case_ids[num_test_cases:]
    test_control_ids = unique_control_ids[:num_test_controls]
    train_control_ids = unique_control_ids[num_test_controls:]

    # Combine the test and train patient IDs to get the final sets
    test_patient_ids = test_case_ids + test_control_ids
    train_patient_ids = train_case_ids + train_control_ids

    # now take the data and labels for the test and train patient ids
    train_data_list, train_labels = [], []
    test_data_list, test_labels = [], []
    for data, patient_id, label in zip(combined_data_list, combined_patient_id_list, combined_label_list):
        if patient_id in train_patient_ids:
            train_data_list.append(data)
            train_labels.append(label)
        elif patient_id in test_patient_ids:
            test_data_list.append(data)
            test_labels.append(label)
        else:
            assert False, f'Error with code'
    return train_data_list, test_data_list, train_labels, test_labels




def train(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_batches = len(train_loader)
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y) # CHANGE THIS TO BINARY CROSS ENTROPY LOSS
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        total_loss += loss.item()

    avg_loss = total_loss / total_batches
    return avg_loss


def test(model, test_loader, device):
    model.eval()
    correct = 0
    labels_list = []
    pred_list = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            for i in range(len(pred)):
                labels_list.append(data.y[i].item())
                pred_list.append(pred[i].item())

    return correct / len(test_loader.dataset), labels_list, pred_list



class GraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, dropout=0.5):
        super(GraphClassifier, self).__init__()

        num_layers, start_numhidden = 3, 128
        exp_decay = 0.5

        hidden_channels = [num_node_features]
        curr_numhidden = start_numhidden
        for i in range(num_layers):
            hidden_channels.append(curr_numhidden)
            curr_numhidden = int(curr_numhidden * exp_decay)
        print("hidden_channels", hidden_channels)
        conv_layers = []
        for i in range(len(hidden_channels) - 1):
            conv_layers.append(pyg_nn.GCNConv(hidden_channels[i], hidden_channels[i+1]))

        self.graph_convs = torch.nn.ModuleList(conv_layers)
        self.classifier = torch.nn.Linear(conv_layers[-1].out_channels, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # raise Exception()
        for conv in self.graph_convs:
            x = conv(x, edge_index)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class GATGraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, dropout=0.3):
        super(GATGraphClassifier, self).__init__()

        embedding_size = 128
        heads = 3
        self.conv1 = GATConv(num_node_features, embedding_size, heads=heads, dropout=dropout)
        self.head_transform1 = torch.nn.Linear(embedding_size * heads, embedding_size)
        self.pool1 = TopKPooling(embedding_size, ratio=0.8)
        self.conv2 = GATConv(embedding_size, embedding_size, heads=heads, dropout=dropout)
        self.head_transform2 = torch.nn.Linear(embedding_size * heads, embedding_size)
        self.pool2 = TopKPooling(embedding_size, ratio=0.8)
        self.conv3 = GATConv(embedding_size, embedding_size, heads=heads, dropout=dropout)
        self.head_transform3 = torch.nn.Linear(embedding_size * heads, embedding_size)
        self.pool3 = TopKPooling(embedding_size, ratio=0.8)

        self.linear1 = torch.nn.Linear(embedding_size * 2, embedding_size)
        self.linear2 = torch.nn.Linear(embedding_size, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.head_transform1(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index)
        x = self.head_transform2(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv3(x, edge_index)
        x = self.head_transform3(x)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3
        x = self.linear1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.linear2(x)

        return F.log_softmax(x, dim=1)
