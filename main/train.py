import os
import torch
from model import Model
from dataset import MyDataset, MySampler
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import random
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = args.dataset

if dataset_name == 'TwiBot20':
    lr = 1e-3
    batch_size = 64
    checkpoints_path = 'TwiBot20'
    augmentation = 0
    dropout = 0.3
    metric = accuracy_score
    num_hops = 2
else:
    lr = 1e-3
    batch_size = 1024
    checkpoints_path = 'TwiBot22'
    augmentation = 10
    dropout = 0.5
    metric = f1_score
    num_hops = 4


def forward_one_batch(model, batch):
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    return model(batch)


def train_one_epoch(model, optimizer):
    model.train()
    ave_loss = 0
    cnt = 0
    for batch in tqdm(train_loader, ncols=50):
        optimizer.zero_grad()
        _, loss = forward_one_batch(model, batch)
        loss.backward()
        optimizer.step()
        ave_loss += loss.item() * batch['batch_size']
        cnt += batch['batch_size']
    ave_loss /= cnt
    print('train loss: {:.4f}'.format(ave_loss))


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
    return metric(all_truth, all_preds) * 100


def train(save_path='checkpoints'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model = Model(numerical_dim=dataset.numerical.shape[-1],
                  categorical_dim=dataset.categorical.shape[-1],
                  description_dim=dataset.description.shape[-1],
                  tweet_dim=dataset.tweet.shape[-1],
                  dropout=dropout,
                  act_fn=nn.LeakyReLU(),
                  node_dim=256,
                  num_similar_head=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_acc = 0
    best_state = model.state_dict()
    for key, value in best_state.items():
        best_state[key] = value.clone()
    no_up_limit = 12
    no_up = 0

    while True:
        train_one_epoch(model, optimizer)
        acc = validation(model, val_loader)
        if acc > best_acc:
            best_acc = acc
            for key, value in model.state_dict().items():
                best_state[key] = value.clone()
            no_up = 0
        else:
            no_up += 1
        if no_up >= no_up_limit:
            break
        print('no_up: {} now best val acc: {:.2f}'.format(no_up, best_acc))

    model.load_state_dict(best_state)
    acc = validation(model, test_loader)
    print('test acc: {:.2f}'.format(acc))
    torch.save(best_state, '{}/{:.2f}.pt'.format(save_path, acc))


if __name__ == '__main__':
    dataset = MyDataset(dataset_name, 15, num_hops, -0.5)
    while True:
        train_indices = dataset.train_indices
        val_indices = dataset.val_indices
        test_indices = dataset.test_indices

        train_sampler = MySampler(train_indices, training=True, label=dataset.label, augmentation=augmentation)  # 22 10
        val_sampler = MySampler(val_indices, training=False)
        test_sample = MySampler(test_indices, training=False)

        print(len(train_sampler), len(val_sampler), len(test_sample))

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                                  collate_fn=dataset.get_collate_fn())
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler,
                                collate_fn=dataset.get_collate_fn())
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sample,
                                 collate_fn=dataset.get_collate_fn())

        train(checkpoints_path)
