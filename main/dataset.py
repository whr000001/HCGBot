import os
import torch
import json
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import NeighborSampler
from torch.utils.data.sampler import Sampler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MySampler(Sampler):
    def __init__(self, indices, training, label=None, augmentation=0, data_source=None):
        super().__init__(data_source)
        if augmentation:
            label = label[indices]
            true_indices = indices[label == 1]
            false_indices = indices[label == 0]
            # true_indices = [true_indices for _ in range(augmentation + 1)]
            # true_indices = torch.cat(true_indices, dim=0)
            true_indices = [true_indices for _ in range(4)]
            true_indices = torch.cat(true_indices, dim=0)
            false_indices = [false_indices for _ in range(2)]
            false_indices = torch.cat(false_indices, dim=0)
            indices = torch.cat([false_indices, true_indices], dim=0)
        self.indices = indices
        self.training = training

    def __iter__(self):
        if self.training:
            for i in self.indices[torch.randperm(len(self.indices))]:
                yield i
        else:
            for i in self.indices:
                yield i

    def __len__(self):
        return len(self.indices)


dataset_length = {
    'TwiBot20': 229580,
    'TwiBot22': 1000000,
}


class MyDataset(Dataset):
    def __init__(self, dataset_name, top_k_count, num_hops, threshold):
        self.length = dataset_length[dataset_name]
        self.uids = torch.arange(self.length)

        self.label = torch.load('../../dataset/{}/label.pt'.format(dataset_name))

        self.numerical = torch.load('../../dataset/{}/numerical.pt'.format(dataset_name))
        self.categorical = torch.load('../../dataset/{}/categorical.pt'.format(dataset_name))

        self.description = torch.load('../../dataset/{}/description.pt'.format(dataset_name))
        self.tweet = torch.load('../../dataset/{}/tweet.pt'.format(dataset_name))

        self.train_indices = torch.load('../../dataset/{}/train.pt'.format(dataset_name))
        self.val_indices = torch.load('../../dataset/{}/val.pt'.format(dataset_name))
        self.test_indices = torch.load('../../dataset/{}/test.pt'.format(dataset_name))

        self.is_train = torch.zeros(self.length, dtype=torch.bool)
        self.is_train[self.train_indices] = True

        self.edge_index = torch.load('../../dataset/{}/edge_index.pt'.format(dataset_name))
        self.edge_type = torch.load('../../dataset/{}/edge_type.pt'.format(dataset_name))
        self.neighbor_sampler = NeighborSampler(
            edge_index=self.edge_index,
            node_idx=self.uids,
            sizes=[-1] * 2
        )

        social_context = json.load(open('../context_graph/contexts/{}_{}_hops_social_context.json'.
                                        format(dataset_name, num_hops)))
        user_without_description = set(json.load(
                open('../../dataset/{}/{}_user_without_description.json'.format(dataset_name, dataset_name)))
            )
        self.social_context = []

        src, tgt = [], []
        for index, item in tqdm(enumerate(social_context), ncols=50):
            if index in user_without_description:
                self.social_context.append({
                    'user': [],
                    'weight': []
                })
                continue
            context = [_ for _ in item if _[1] > threshold and _[0] not in user_without_description]
            context = context[:top_k_count]
            for _ in context:
                src.append(index)
                tgt.append(_[0])
            context = {
                'user': [_[0] for _ in context],
                'weight': [_[1] for _ in context]
            }
            self.social_context.append(context)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        out = {
            'uid': self.uids[index],
            'label': self.label[index],
            'context': self.social_context[index]
        }
        return out

    def get_collate_fn(self):
        def my_collate(batch):
            batch_size = len(batch)
            uid2batch_id = {}
            sample_idx = []
            uids = []
            for index, item in enumerate(batch):
                uid2batch_id[item['uid'].item()] = index
                sample_idx.append(item['uid'].item())
                uids.append(item['uid'].item())
            cnt = len(uid2batch_id)
            for item in batch:
                context = item['context']
                for user in context['user']:
                    if user not in uid2batch_id:
                        sample_idx.append(user)
                        uid2batch_id[user] = cnt
                        cnt += 1

            out = self.neighbor_sampler.sample(sample_idx)
            sample_idx = out[1]
            edge_index = out[2][0].edge_index
            edge_type = self.edge_type[out[2][0].e_id]

            numerical = self.numerical[sample_idx]
            categorical = self.categorical[sample_idx]
            tweet = self.tweet[sample_idx]
            description = self.description[sample_idx]
            is_train = self.is_train[sample_idx]

            label = self.label[sample_idx]

            context = []
            for item in batch:
                context_user = [uid2batch_id[_] for _ in item['context']['user']]
                context.append({
                    'user': context_user,
                    'weight': item['context']['weight'],
                    'initial_user': item['context']['user']
                })
            return {
                'numerical': numerical,
                'categorical': categorical,
                'tweet': tweet,
                'description': description,
                'context': context,
                'label': label,
                'batch_size': batch_size,
                'edge_index': edge_index,
                'edge_type': edge_type,
                'uids': uids,
                'is_train': is_train
            }
        return my_collate


if __name__ == '__main__':
    dataset = MyDataset('TwiBot20', 15, 2, 0.0)

