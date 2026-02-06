import json
import torch
from tqdm import tqdm
import ijson
from queue import Queue
import math
import numpy as np


dataset_name = 'TwiBot20'


# the weighted edge homophily, as eq. 3
def measure_weighted_edge_homophily(edge_index, edge_value, label):
    src, tgt = edge_index
    cnt, total = 0, 0
    for index in tqdm(range(len(src)), ncols=0, leave=False):
        if label[src[index]].item() == 2 or label[tgt[index]].item() == 2:
            continue
        if label[src[index]] == label[tgt[index]]:
            total += edge_value[index].item()
        cnt += edge_value[index].item()
    return total / cnt


def sigmoid(x, tau=0.25):
    x = x / tau
    return 1 / (1 + np.exp(-x))


# the class-independent homophily (our proposed one)
def measure_homophily(edge_index, edge_value, label):
    cnt_0 = label[label == 0].shape[0]
    cnt_1 = label[label == 1].shape[0]
    cnt = cnt_0 + cnt_1
    expectation = (cnt_0 * (cnt_0 - 1) + cnt_1 * (cnt_1 - 1)) / cnt / (cnt - 1)  # the expectation, refer to eq. 4
    weighted_edge_homophily = measure_weighted_edge_homophily(edge_index, edge_value, label)
    return sigmoid(weighted_edge_homophily - expectation)


# edge homophliy
def measure_edge_homophily(edge_index, label):
    src, tgt = edge_index
    cnt, total = 0, 0
    for index in tqdm(range(len(src)), ncols=0, leave=False):
        if label[src[index]].item() == 2 or label[tgt[index]].item() == 2:
            continue
        if label[src[index]] == label[tgt[index]]:
            total += 1
        cnt += 1
    return total / cnt


# the node homophily
def measure_node_homophily(edge_index, label):
    src, tgt = edge_index
    node = {}
    for index in range(len(label)):
        node[index] = []
    for x, y in zip(src, tgt):
        x, y = x.item(), y.item()
        node[x].append(y)
        node[y].append(x)
    total = 0
    total_cnt = 0
    for index, item in enumerate(tqdm(label, ncols=0, leave=False)):
        if item.item() == 2:
            continue
        item_label = label[index]
        neighbor_label = label[node[index]]
        cnt = 0
        labeled_cnt = 0
        for x in neighbor_label:
            if x.item() == 2:
                continue
            labeled_cnt += 1
            if x.item() == item_label.item():
                cnt += 1
        if labeled_cnt == 0:
            cnt = cnt
        else:
            cnt = cnt / labeled_cnt
        total_cnt += cnt
        total += 1
    return total_cnt / total


# the improved homophily
def measure_improved_homophily(edge_index, label):
    src, tgt = edge_index
    node = {}
    for index in range(len(label)):
        node[index] = []
    for x, y in zip(src, tgt):
        x, y = x.item(), y.item()
        node[x].append(y)
        node[y].append(x)
    res = 0
    for indices in [0, 1]:
        cnt = 0
        total = 0
        h_total = 0
        h_total_cnt = 0
        for index, item in enumerate(tqdm(label, ncols=0, leave=False)):
            if item.item() == 2:
                continue
            total += 1
            if item.item() == indices:
                cnt += 1
                item_label = label[index]
                neighbor_label = label[node[index]]
                h_cnt = 0
                labeled_cnt = 0
                for x in neighbor_label:
                    if x.item() == 2:
                        continue
                    labeled_cnt += 1
                    if x.item() == item_label.item():
                        h_cnt += 1
                if labeled_cnt == 0:
                    h_cnt = h_cnt
                else:
                    h_cnt = h_cnt / labeled_cnt
                h_total_cnt += h_cnt
                h_total += 1
        h_k = h_total_cnt / h_total
        Ckn = cnt / total
        res += max(0.0, h_k - Ckn)
    return res / 2


def main():
    label = torch.load('../../dataset/{}/label.pt'.format(dataset_name))
    mutual_follow_edge_index = torch.load('../../dataset/{}/edge_index.pt'.format(dataset_name))
    mutual_follow_edge_value = torch.ones(mutual_follow_edge_index.shape[1])
    context_edge_index, context_edge_value = torch.load('../main/{}_context_graph.pt'.format(dataset_name))

    print('node homophily')
    print(measure_node_homophily(mutual_follow_edge_index, label))
    print(measure_node_homophily(context_edge_index, label))

    print('edge homophily')
    print(measure_edge_homophily(mutual_follow_edge_index, label))
    print(measure_edge_homophily(context_edge_index, label))

    # print(measure_weighted_edge_homophily(mutual_follow_edge_index, mutual_follow_edge_value, label))
    # print(measure_weighted_edge_homophily(context_edge_index, context_edge_value, label))

    print('improved homophily')
    print(measure_improved_homophily(mutual_follow_edge_index, label))
    print(measure_improved_homophily(context_edge_index, label))

    print('independent homophily')
    print(measure_homophily(mutual_follow_edge_index, mutual_follow_edge_value, label))
    print(measure_homophily(context_edge_index, context_edge_value, label))

    exit(0)


if __name__ == '__main__':
    main()
