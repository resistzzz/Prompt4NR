import os
import argparse
import pickle
import time
import sys

from tqdm import tqdm
from datetime import datetime
import torch.cuda
from torch.utils.data import DataLoader

from transformers import BertTokenizer
from transformers import AdamW

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model import BERTPrompt4NR
from prepro_data import *
from utils import evaluate


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23342'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def load_model(model_name, args):
    tokenizer = BertTokenizer.from_pretrained(model_name)

    new_tokens = ['[NSEP]']
    tokenizer.add_tokens(new_tokens)
    new_vocab_size = len(tokenizer)
    args.vocab_size = new_vocab_size

    answer = ['unrelated', 'related']
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)

    net = BERTPrompt4NR(model_name, answer_ids, args)
    return net, tokenizer


def eval(model, rank, world_size, data_loader):
    model.eval()
    data_loader = tqdm(data_loader)
    val_scores = []
    acc_cnt = torch.zeros(2).to(rank)
    acc_cnt_pos = torch.zeros(2).to(rank)
    imp_ids = []
    labels = []
    for step, data in enumerate(data_loader):
        batch_enc, batch_attn, batch_labs, batch_imp = data
        imp_ids = imp_ids + batch_imp
        labels = labels + batch_labs.cpu().numpy().tolist()

        batch_enc = batch_enc.to(rank)
        batch_attn = batch_attn.to(rank)
        batch_labs = batch_labs.to(rank)

        loss, scores = model(batch_enc, batch_attn, batch_labs)

        ranking_scores = scores[:, 1].detach()
        val_scores.append(ranking_scores)

        predict = torch.argmax(scores.detach(), dim=1)
        num_correct = (predict == batch_labs).sum()
        acc_cnt[0] += num_correct
        acc_cnt[1] += predict.size(0)

        positive_idx = torch.where(batch_labs == 1)[0]
        num_correct_pos = (predict[positive_idx] == batch_labs[positive_idx]).sum()
        acc_cnt_pos[0] += num_correct_pos
        acc_cnt_pos[1] += positive_idx.size(0)

    dist.all_reduce(acc_cnt, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_cnt_pos, op=dist.ReduceOp.SUM)

    acc = acc_cnt[0] / acc_cnt[1]
    acc_pos = acc_cnt_pos[0] / acc_cnt_pos[1]
    pos_ratio = acc_cnt_pos[1] / acc_cnt[1]

    val_scores = torch.cat(val_scores, dim=0)
    val_impids = torch.IntTensor(imp_ids).to(rank)
    val_labels = torch.IntTensor(labels).to(rank)

    val_scores_list = [torch.zeros_like(val_scores).to(rank) for _ in range(world_size)]
    val_impids_list = [torch.zeros_like(val_impids).to(rank) for _ in range(world_size)]
    val_labels_list = [torch.zeros_like(val_labels).to(rank) for _ in range(world_size)]

    dist.all_gather(val_scores_list, val_scores)
    dist.all_gather(val_impids_list, val_impids)
    dist.all_gather(val_labels_list, val_labels)

    return val_scores_list, acc.item(), acc_pos.item(), pos_ratio.item(), val_impids_list, val_labels_list


def ddp_main(rank, world_size, args):
    args.rank = rank
    args.world_size = world_size
    init_seed(rank + 1)
    if rank == 0:
        if args.log:
            sys.stdout = Logger(args.log_file, sys.stdout)
    setup(rank, world_size)

    print('| distributed init rank {}'.format(rank))
    dist.barrier()

    # load model
    net, tokenizer = load_model(args.model_name, args)

    # load data
    news_dict = pickle.load(open(os.path.join(args.data_path, 'news.txt'), 'rb'))
    test_dataset = MyDataset(args, tokenizer, news_dict, status='test')

    if rank == 0:
        print(args)
        print('Vocabulary size of tokenizer after adding new tokens : %d' % args.vocab_size)
        print(test_dataset[0]['sentence'])
        print('num test: %d' % len(test_dataset))

    test_sampler = DistributedSampler(test_dataset,
                                      rank=rank,
                                      num_replicas=world_size)
    nw = 2
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': test_sampler,
                   'shuffle': False, 'pin_memory': False,
                   'num_workers': nw, 'collate_fn': test_dataset.collate_fn}

    test_loader = DataLoader(test_dataset, **test_kwargs)

    net = net.to(rank)
    net = DDP(net, device_ids=[rank])

    dist.barrier()

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    net.module.load_state_dict(torch.load(args.model_file, map_location=map_location))

    with torch.no_grad():
        st_test = time.time()
        test_scores, acc_test, acc_pos_test, pos_ratio_test, test_impids, test_labels = \
            eval(net, rank, world_size, test_loader)
        impressions = {}  # {1: {'score': [], 'lab': []}}
        for i in range(world_size):
            scores, imp_id, labs = test_scores[i], test_impids[i], test_labels[i]
            assert scores.size() == imp_id.size() == labs.size()
            scores = scores.cpu().numpy().tolist()
            imp_id = imp_id.cpu().numpy().tolist()
            labs = labs.cpu().numpy().tolist()
            for j in range(len(scores)):
                sco, imp, lab = scores[j], imp_id[j], labs[j]
                if imp not in impressions:
                    impressions[imp] = {'score': [], 'lab': []}
                    impressions[imp]['score'].append(sco)
                    impressions[imp]['lab'].append(lab)
                else:
                    impressions[imp]['score'].append(sco)
                    impressions[imp]['lab'].append(lab)
        predicts, truths = [], []
        for imp in impressions:
            sims, labs = impressions[imp]['score'], impressions[imp]['lab']
            sl_zip = sorted(zip(sims, labs), key=lambda x: x[0], reverse=True)
            sort_sims, sort_labs = zip(*sl_zip)
            predicts.append(list(range(1, len(sort_labs) + 1, 1)))
            truths.append(sort_labs)

        auc_test, mrr_test, ndcg5_test, ndcg10_test = evaluate(predicts, truths)
        end_test = time.time()
        test_spend = (end_test - st_test) / 60

        if rank == 0:
            print("Test: AUC: %0.4f\tMRR: %0.4f\tnDCG@5: %0.4f\tnDCG@10: %0.4f\t[Test-Time: %0.2f mim]" %
                  (auc_test, mrr_test, ndcg5_test, ndcg10_test, test_spend))
    cleanup()


if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../DATA/MIND-Small', type=str, help='Path')
    parser.add_argument('--model_name', default='bert-base-uncased', type=str)

    parser.add_argument('--test_batch_size', default=15, type=int, help='test batch_size')
    parser.add_argument('--max_his', default=50, type=int, help='max number of history')
    parser.add_argument('--max_tokens', default=500, type=int, help='max number of tokens')

    parser.add_argument('--max_his_len', default=450, type=int, help='max number of history')

    parser.add_argument('--device', default='cuda', help='device id')
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')

    parser.add_argument('--model_file', default='', type=str, help='model file')
    parser.add_argument('--log', default=False, type=bool, help='whether write log file')
    # parser.add_argument('--log', default=True, type=bool, help='whether write log file')

    args = parser.parse_args()

    if args.data_path == '../DATA/MIND-Demo':
        if args.log:
            if not os.path.exists('./log-Test'):
                os.makedirs('./log-Test')
            log_file = './log-Test/' + 'Tbs' + str(args.test_batch_size) + '-' + str(datetime.now())[-5:]+'.txt'
            args.log_file = log_file
    else:   # Mind-Small
        if args.log:
            if not os.path.exists('./log-Test-Small'):
                os.makedirs('./log-Test-Small')
            log_file = './log-Test-Small/' + 'Tbs' + str(args.test_batch_size) + '-' + str(datetime.now())[-5:]+'.txt'
            args.log_file = log_file

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(ddp_main,
             args=(WORLD_SIZE, args),
             nprocs=WORLD_SIZE,
             join=True)
    t1 = time.time()
    run_time = (t1 - t0) / 3600
    print('Running time: %0.4f' % run_time)