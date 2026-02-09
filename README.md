# HCGBot

This is the official repository for the paper at TKDE: [HCGBot: Learning Homophilous Context Graphs for Twitter Bot Detection](https://doi.org/10.1109/TKDE.2026.3656720)

## What you can get from HCGBot
- A context graph learning strategy for detecting social bots, where the learned graph is more homophilous, achieving better performance.
- A better context graph for graph-based social bot detection models (TwiBot-20 and TwiBot-22 datasets).

## How to train/evaluate HCGBot

### Datasets
Due to privacy restrictions, we can only provide processed/de-anonymized data. You could access the data from Baidu Disk: [https://pan.baidu.com/s/1fMYDGoZ2Snq8Oi9MTn0B0Q?pwd=adn7](https://pan.baidu.com/s/1fMYDGoZ2Snq8Oi9MTn0B0Q?pwd=adn7)
Or, if you require the complete raw data, please see the [dataset homepage](https://twibot22.github.io/).

### Basic Usage

#### Preprocess
First, make sure you have downloaded the preprocessed data.
Then, you should construct the candidate neighbors in the context graph for each user, running:
```
cd src
cd context_build
python main.py --[dataset] --[num_hops]
```
where you could choose TwiBot20 or TwiBot22 as [dataset], and 1-6 as [num_hops]. Otherwise, you can also download this data from Baidu Disk.

#### Train, evaluate HCGBot
Run:
```
cd src
cd main
python train.py --[dataset] 
```
where you could change the default config in train.py.
To obtain the training performance, pls run:
```
python inference.py --[dataset]
```

#### Obtain the context graph
After training HCGBot, you could obtain a better context graph for social bot detection, running:
```
cd src
cd main
python obtain_context_graph.py --[dataset] 
```
We have also provided our learned context graph, which you could download from Baidu Disk. 

(**Note: This graph was obtained through supervised learning based on the original segmentation of the dataset. If you intend to use this graph directly, please be aware of the risk of label leakage.**)

#### The class-independent homophily
You could obtain the homophily of the follow graph and the context graph by running:
```
cd src
cd class-independent_homophily
python main.py
```
We have also provided other homophily measures in this file. You could directly employ them. (Make sure cite the property papers)


## Citation
If you find our work interesting/helpful, please consider citing this paper
```
@article{wan2026hcgbot,
  title={HCGBot: Learning Homophilous Context Graphs for Twitter Bot Detection},
  author={Wan, Herun and Luo, Minnan and Wang, Jihong and Chang, Xiaojun and Zheng, Qinghua},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2026},
  publisher={IEEE}
}
```

## Question?
Feel free to open issues in this repository! Instead of emails, GitHub issues are much better at facilitating a conversation between you and our team to address your needs. You can also contact Herun Wan through `wanherun at stu.xjtu.edu.cn`.

## Updating

### 20260209
- We have provided a simple tutorial on using HCGBot.
- We have uploaded the preprocessed data and related resources to Baidu Disk.
### 20260206
- We have uploaded the newest codes.
### 20260131
- Our paper has been accepted to the TKDE!🙌🙌🙌
- We plan to refine this repository by March.

### Before
- We have uploaded related codes. However, it's missing a lot of details.
