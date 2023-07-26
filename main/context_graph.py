import os
import torch
from model import Model
from dataset import MyDataset, MySampler
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = args.dataset


if __name__ == '__main__':
    if dataset_name == 'TwiBot20':
        num_hops = 2
    else:
        num_hops = 4
    dataset = MyDataset(dataset_name, 15, num_hops, -0.5)

    if dataset_name == 'TwiBot20':
        indices = torch.arange(229580)
    else:
        indices = torch.cat([dataset.train_indices, dataset.val_indices, dataset.test_indices], dim=0)

    sampler = MySampler(indices, training=False)
    loader = DataLoader(dataset, batch_size=1024, sampler=sampler, collate_fn=dataset.get_collate_fn())

    checkpoints = os.listdir(dataset_name)
    checkpoints = sorted(checkpoints, reverse=True)
    checkpoint = checkpoints[0]

    checkpoint_path = '{}/{}'.format(dataset_name, checkpoint)
    print(checkpoint_path)

    model = Model(numerical_dim=dataset.numerical.shape[-1],
                  categorical_dim=dataset.categorical.shape[-1],
                  description_dim=dataset.description.shape[-1],
                  tweet_dim=dataset.tweet.shape[-1],
                  dropout=0.5,
                  act_fn=nn.LeakyReLU(),
                  node_dim=256,
                  num_similar_head=4).to(device)
    model.load_state_dict(torch.load(checkpoint_path))

    edge_index, edge_weight = [], []
    for batch in tqdm(loader, ncols=50):
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        sub_edge_index, sub_edge_weight = model(batch, return_graph=True)
        edge_index.append(sub_edge_index)
        edge_weight.append(sub_edge_weight)
    edge_index = torch.cat(edge_index, dim=-1).to('cpu')
    edge_weight = torch.cat(edge_weight, dim=-1).to('cpu')
    print(edge_index.shape, edge_weight.shape)
    torch.save([edge_index, edge_weight], '{}_context_graph.pt'.format(dataset_name))
