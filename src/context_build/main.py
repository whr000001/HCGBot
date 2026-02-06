# this file is used to obtain the candidate users as the neighbors in the context graph
# Namely, it severs as the Twitter graph constraint and the feature constraint (default) in the paper.

import json
import torch
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
import os
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--num_hops', type=int, default=1)
args = parser.parse_args()

dataset = args.dataset
num_hops = args.num_hops

if not os.path.exists('contexts'):
    os.mkdir('contexts')
path = 'contexts/{}_{}_hops_social_context.json'.format(dataset, num_hops)
if os.path.exists(path):
    exit(0)


if __name__ == '__main__':
    description_vec = torch.load('../../dataset/{}/simcse_vec.pt'.format(dataset))
    edge_index = torch.load('../../dataset/{}/whole_edge_index.pt'.format(dataset)).to('cuda')
    description_vec = description_vec / description_vec.norm(dim=1, keepdim=True)
    node_cnt = description_vec.shape[0]
    self_loop = torch.stack([torch.arange(node_cnt), torch.arange(node_cnt)]).to('cuda')
    edge_index = torch.cat([edge_index, self_loop], dim=-1)

    social_context = []

    for uid in tqdm(range(node_cnt), ncols=0):
        sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(uid, edge_index=edge_index, num_hops=num_hops)
        sub_nodes = sub_nodes[sub_nodes < node_cnt]
        user_vec = description_vec[uid]
        social_context_vecs = description_vec[sub_nodes]

        cos_sims = torch.matmul(user_vec, social_context_vecs.transpose(0, 1)).to('cpu').numpy().tolist()

        tup = zip(sub_nodes.to('cpu').numpy().tolist(), cos_sims)
        tup = sorted(tup, key=lambda x: x[1], reverse=True)
        tup = [item for item in tup if item[0] != uid]
        tup = tup[:100]
        social_context.append(tup)
    json.dump(social_context, open(path, 'w'))
