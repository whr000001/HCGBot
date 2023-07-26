import os
import numpy as np
import torch
from model import Model
from dataset import MyDataset, MySampler
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = args.dataset


def forward_one_batch(model, batch):
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    return model(batch, return_out=True)


@torch.no_grad()
def validation(model, loader):
    model.eval()
    output = []
    for batch in tqdm(loader, ncols=50):
        out = forward_one_batch(model, batch)
        out = torch.softmax(out, dim=-1)
        output.append(out)
    #     all_probs.append(out.to('cpu'))
    #     preds = out.argmax(-1).to('cpu')
    #     truth = batch['label'][:batch['batch_size']]
    #     all_truth.append(truth)
    #     all_preds.append(preds)
    # all_preds = torch.cat(all_preds, dim=0).numpy()
    # all_truth = torch.cat(all_truth, dim=0).numpy()
    # all_probs = torch.cat(all_probs, dim=0).numpy()
    return output


def get_out(checkpoint_path):
    model = Model(numerical_dim=dataset.numerical.shape[-1],
                  categorical_dim=dataset.categorical.shape[-1],
                  description_dim=dataset.description.shape[-1],
                  tweet_dim=dataset.tweet.shape[-1],
                  dropout=0.5,
                  act_fn=nn.LeakyReLU(),
                  node_dim=256,
                  num_similar_head=4).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    output = validation(model, all_loader)
    output = torch.cat(output, dim=0)
    print(output.shape)
    torch.save(output, '{}_reps.pt'.format(dataset_name))


if __name__ == '__main__':
    if dataset_name == 'TwiBot20':
        num_hops = 2
    else:
        num_hops = 4
    dataset = MyDataset(dataset_name, 15, num_hops, -0.5)

    indices = dataset.test_indices

    sampler = MySampler(indices, training=False)
    all_loader = DataLoader(dataset, batch_size=1024, sampler=sampler, collate_fn=dataset.get_collate_fn())

    checkpoints = os.listdir(dataset_name)
    checkpoints = sorted(checkpoints, reverse=True)

    checkpoint = checkpoints[0]
    print(checkpoint)
    get_out('{}/{}'.format(dataset_name, checkpoint))
