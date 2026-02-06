# this file is used to obtain the reported performance

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


def forward_one_batch(model, batch):
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    return model(batch)


@torch.no_grad()
def validation(model, loader):
    model.eval()
    all_truth = []
    all_preds = []
    for batch in loader:
        out, _ = forward_one_batch(model, batch)
        preds = out.argmax(-1).to('cpu')
        truth = batch['label'][:batch['batch_size']]
        all_truth.append(truth)
        all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    return all_truth, all_preds


def inference(checkpoint_path):
    model = Model(numerical_dim=dataset.numerical.shape[-1],
                  categorical_dim=dataset.categorical.shape[-1],
                  description_dim=dataset.description.shape[-1],
                  tweet_dim=dataset.tweet.shape[-1],
                  dropout=0.5,
                  act_fn=nn.LeakyReLU(),
                  node_dim=256,
                  num_similar_head=4).to(device)
    model.load_state_dict(torch.load(checkpoint_path))

    all_truth, all_preds = validation(model, test_loader)
    acc = accuracy_score(all_truth, all_preds) * 100
    f1 = f1_score(all_truth, all_preds) * 100
    pre = precision_score(all_truth, all_preds) * 100
    rec = recall_score(all_truth, all_preds) * 100
    print('f1: {:.2f} acc: {:.2f} pre: {:.2f} rec: {:.2f}'.format(f1, acc, pre, rec))


if __name__ == '__main__':
    if dataset_name == 'TwiBot20':
        num_hops = 2
    else:
        num_hops = 4
    dataset = MyDataset(dataset_name, 15, num_hops, -0.5)

    test_sampler = MySampler(dataset.test_indices, training=False)
    test_loader = DataLoader(dataset, batch_size=1024, sampler=test_sampler, collate_fn=dataset.get_collate_fn())

    checkpoints = os.listdir(dataset_name)
    checkpoints = sorted(checkpoints, reverse=True)

    for checkpoint in checkpoints[:5]:
        inference('{}/{}'.format(dataset_name, checkpoint))
