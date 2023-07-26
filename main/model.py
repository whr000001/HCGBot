import torch
import torch.nn as nn
from torch_geometric.nn.conv import RGCNConv
import numpy as np


class IndividualEncoder(nn.Module):
    def __init__(self, numerical_dim, categorical_dim, description_dim, tweet_dim, node_dim, act_fn, dropout):
        super().__init__()
        self.numerical_fc = nn.Linear(numerical_dim, node_dim // 4)
        self.categorical_fc = nn.Linear(categorical_dim, node_dim // 4)
        self.description_fc = nn.Linear(description_dim, node_dim // 4)
        self.tweet_fc = nn.Linear(tweet_dim, node_dim // 4)
        self.act_fn = act_fn
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        for _ in range(2):
            self.convs.append(RGCNConv(node_dim, node_dim, 2))

    def forward(self, data):
        numerical = self.numerical_fc(data['numerical'])
        categorical = self.categorical_fc(data['categorical'])
        description = self.description_fc(data['description'])
        tweet = self.tweet_fc(data['tweet'])
        initial_reps = self.dropout(self.act_fn(torch.cat([numerical, categorical, description, tweet], dim=-1)))
        individual_reps = initial_reps
        for cov in self.convs:
            individual_reps = cov(individual_reps, data['edge_index'], data['edge_type'])
            individual_reps = self.act_fn(individual_reps)
        return individual_reps


class WeightEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_mu = np.arange(-1, 1.1, 0.1)
        kernel_mu = list(kernel_mu)
        kernel_sigma = [20 for _ in kernel_mu]
        kernel_mu.append(0.99)
        kernel_sigma.append(100)
        self.kernel_mu = torch.tensor(kernel_mu, dtype=torch.float)
        self.kernel_sigma = torch.tensor(kernel_sigma, dtype=torch.float)

    def forward(self, weights):
        k, n = len(self.kernel_mu), len(weights)
        if n == 0:
            return torch.zeros(k).to(weights.device)
        mu = self.kernel_mu.repeat(n, 1).to(weights.device)
        sigma = self.kernel_sigma.repeat(n, 1).to(weights.device)

        sim_values = weights
        sim_values = sim_values.repeat(k, 1).T

        kernel_features = torch.exp(-0.5 * ((sim_values - mu) * sigma) ** 2)
        kernel_features = torch.sum(kernel_features, dim=0)

        return kernel_features


class Model(nn.Module):
    def __init__(self, numerical_dim, categorical_dim, description_dim, tweet_dim, node_dim, act_fn, dropout,
                 num_similar_head):
        super().__init__()
        self.node_dim = node_dim
        self.individual_encoder = IndividualEncoder(numerical_dim, categorical_dim, description_dim, tweet_dim,
                                                    node_dim, act_fn, dropout)
        self.weight_encoder = WeightEncoder()
        self.topological_encoder = nn.Linear(node_dim, node_dim)
        self.similarity_head_w = nn.Parameter(torch.randn(node_dim, num_similar_head))

        self.individual_trans = nn.Linear(node_dim, node_dim)
        self.topological_trans = nn.Linear(node_dim, node_dim)
        self.weight_trans = nn.Linear(22, node_dim)

        self.similarity_weight = 0.25
        self.act_fn = act_fn

        self.cls = nn.Sequential(
            nn.Linear(node_dim * 3, node_dim),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(node_dim, 2)
        )

        self.cls_loss = nn.CrossEntropyLoss()
        self.weight_loss = nn.MSELoss()

    def forward(self, data, return_graph=False, return_out=False):
        individual_reps = self.individual_encoder(data)

        individual_reps_head = individual_reps.unsqueeze(-1) * self.similarity_head_w

        is_train = data['is_train']

        weight_reps = []
        topological_reps = []
        weight_positive = []
        weight_negative = []
        row, col = [], []
        edge_weight = []
        for index, item in enumerate(data['context']):
            initial_weight = torch.tensor(item['weight'], dtype=torch.float).to(individual_reps.device)
            user = item['user']

            offset_weight = torch.cosine_similarity(individual_reps_head[index], individual_reps_head[user],
                                                    eps=1e-6)
            offset_weight = offset_weight.mean(-1)
            weight = self.similarity_weight * initial_weight + (1 - self.similarity_weight) * offset_weight

            topological = individual_reps[user]
            topological = topological[weight > 0]
            weight_reps.append(self.weight_encoder(weight[weight > 0]))

            for item_col, item_weight in zip(item['initial_user'], weight):
                if item_weight > 0:
                    row.append(data['uids'][index])
                    col.append(item_col)
                    edge_weight.append(item_weight)
            if len(topological) == 0:
                topological = torch.zeros(self.node_dim, dtype=torch.float).to(individual_reps.device)
            else:
                topological = self.topological_encoder(topological).mean(0)
                index_label = data['label'][index]
                labels = data['label'][user]
                user_is_train = is_train[user]
                labeled_idx = torch.not_equal(labels, 2)
                labeled_idx = torch.logical_and(user_is_train, labeled_idx)
                positive_idx = torch.eq(labels, index_label)
                negative_idx = torch.not_equal(labels, index_label)

                labeled_positive_idx = torch.logical_and(labeled_idx, positive_idx)
                labeled_negative_idx = torch.logical_and(labeled_idx, negative_idx)

                weight_positive.append(weight[labeled_positive_idx])
                weight_negative.append(weight[labeled_negative_idx])

            topological_reps.append(topological)

        if return_graph:
            edge_index = torch.tensor([row, col], dtype=torch.long).to(individual_reps.device)
            edge_weight = torch.tensor(edge_weight, dtype=torch.float).to(individual_reps.device)
            return edge_index, edge_weight

        topological_reps = self.act_fn(torch.stack(topological_reps))
        topological_reps = self.topological_trans(topological_reps)
        weight_reps = torch.stack(weight_reps)
        individual_reps = self.individual_trans(individual_reps)
        weight_reps = self.weight_trans(weight_reps)
        reps = torch.cat([individual_reps[:data['batch_size']],
                          topological_reps[:data['batch_size']],
                          weight_reps], dim=-1)
        if return_out:
            # return reps
            out = list(self.cls)[0](reps)
            out = list(self.cls)[1](out)
            # out = list(self.cls)[2](out)
            # out = list(self.cls)[3](out)
            # for i, layer in enumerate(list(self.cls)):
            #     if i == 0:
            #         out = layer(reps)
            return out
        cls = self.cls(reps)
        cls_loss = self.cls_loss(cls, data['label'][:data['batch_size']])

        if not weight_positive or not weight_negative:
            weight_loss = 0
        else:
            weight_positive = torch.cat(weight_positive, dim=-1)
            weight_negative = torch.cat(weight_negative, dim=-1)

            weight_positive_loss = self.weight_loss(weight_positive,
                                                    torch.ones(len(weight_positive)).to(weight_positive.device))
            weight_negative_loss = self.weight_loss(weight_negative,
                                                    torch.ones(len(weight_negative)).to(weight_positive.device) * -1)

            weight_loss = weight_positive_loss + weight_negative_loss
        return cls, cls_loss + weight_loss

