
import os
import argparse
import pickle
import time
import sys
import shutil
import warnings
warnings.filterwarnings('ignore')

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


def load_tokenizer(model_name, args):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    conti_tokens1 = []
    for i in range(args.num_conti1):
        conti_tokens1.append('[P' + str(i + 1) + ']')
    conti_tokens2 = []
    for i in range(args.num_conti2):
        conti_tokens2.append('[Q' + str(i + 1) + ']')

    new_tokens = ['[NSEP]']
    tokenizer.add_tokens(new_tokens)

    conti_tokens = conti_tokens1 + conti_tokens2
    tokenizer.add_tokens(conti_tokens)

    new_vocab_size = len(tokenizer)
    args.vocab_size = new_vocab_size

    return tokenizer, conti_tokens1, conti_tokens2


def load_model(model_name, tokenizer, args):
    answer = ['bad', 'good']
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)

    net = BERTPrompt4NR(model_name, answer_ids, args)
    return net


def train(model, optimizer, data_loader, rank, world_size, epoch, sampler):
    model.train()
    mean_loss = torch.zeros(2).to(rank)
    acc_cnt = torch.zeros(2).to(rank)
    acc_cnt_pos = torch.zeros(2).to(rank)
    data_loader = tqdm(data_loader)
    if sampler:
        sampler.set_epoch(epoch)
    for step, data in enumerate(data_loader):
        batch_enc, batch_attn, batch_labs, batch_imp = data

        batch_enc = batch_enc.to(rank)
        batch_attn = batch_attn.to(rank)
        batch_labs = batch_labs.to(rank)

        loss, scores = model(batch_enc, batch_attn, batch_labs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss[0] += loss.item()
        mean_loss[1] += 1

        predict = torch.argmax(scores.detach(), dim=1)
        num_correct = (predict == batch_labs).sum()
        acc_cnt[0] += num_correct
        acc_cnt[1] += predict.size(0)

        positive_idx = torch.where(batch_labs == 1)[0]
        num_correct_pos = (predict[positive_idx] == batch_labs[positive_idx]).sum()
        acc_cnt_pos[0] += num_correct_pos
        acc_cnt_pos[1] += positive_idx.size(0)

    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_cnt, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_cnt_pos, op=dist.ReduceOp.SUM)

    loss_epoch = mean_loss[0] / mean_loss[1]
    acc = acc_cnt[0] / acc_cnt[1]
    acc_pos = acc_cnt_pos[0] / acc_cnt_pos[1]
    pos_ratio = acc_cnt_pos[1] / acc_cnt[1]

    return loss_epoch.item(), acc.item(), acc_pos.item(), pos_ratio.item()


@torch.no_grad()
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


def fsdp_main(rank, world_size, args):
    args.rank = rank
    args.world_size = world_size
    args.gpu = rank
    init_seed(rank + 1)
    if rank == 0:
        if args.log:
            sys.stdout = Logger(args.log_file, sys.stdout)
    setup(rank, world_size)

    print('| distributed init rank {}'.format(rank))
    dist.barrier()

    if rank == 0:
        print(args)

    # load tokenizer
    tokenizer, conti_tokens1, conti_tokens2 = load_tokenizer(args.model_name, args)
    conti_tokens = [conti_tokens1, conti_tokens2]

    # load model
    net = load_model(args.model_name, tokenizer, args)

    # load data
    news_dict = pickle.load(open(os.path.join(args.data_path, 'news.txt'), 'rb'))
    train_dataset = MyDataset(args, tokenizer, news_dict, conti_tokens, status='train')
    val_dataset = MyDataset(args, tokenizer, news_dict, conti_tokens, status='val')

    if rank == 0:
        print('Vocabulary size of tokenizer after adding new tokens : %d' % args.vocab_size)
        print('num train: %d\tnum val: %d' % (len(train_dataset), len(val_dataset)))
        print(train_dataset[0]['sentence'])
        print(val_dataset[3]['sentence'])

    train_sampler = DistributedSampler(train_dataset,
                                       rank=rank,
                                       num_replicas=world_size,
                                       shuffle=True)
    val_sampler = DistributedSampler(val_dataset,
                                     rank=rank,
                                     num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': train_sampler,
                    'shuffle': False, 'pin_memory': True, 'collate_fn': train_dataset.collate_fn}
    val_kwargs = {'batch_size': args.test_batch_size, 'sampler': val_sampler,
                  'shuffle': False, 'pin_memory': True, 'collate_fn': val_dataset.collate_fn}

    nw = 4
    cuda_kwargs = {'num_workers': nw, 'pin_memory': True}
    train_kwargs.update(cuda_kwargs)
    val_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)

    net = net.to(rank)
    net = DDP(net, device_ids=[rank])

    # AdamW
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.wd},
        {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3], gamma=0.1)

    metrics = ['auc', 'mrr', 'ndcg5', 'ndcg10']
    best_val_result = {}
    best_val_epoch = {}
    for m in metrics:
        best_val_result[m] = 0.0
        best_val_epoch[m] = 0

    for epoch in range(args.epochs):
        # #################################  train  ###################################
        st_tra = time.time()
        if rank == 0:
            print('--------------------------------------------------------------------')
            print('start training: ', datetime.now())
            print('Epoch: ', epoch)
            print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])

        loss, acc_tra, acc_pos_tra, pos_ratio_tra = \
            train(net, optimizer, train_loader, rank, world_size, epoch, train_sampler)
        # scheduler.step()

        end_tra = time.time()
        train_spend = (end_tra - st_tra) / 3600
        if rank == 0:
            print("Train Loss: %0.4f" % loss)
            print("Train ACC: %0.4f\tACC-Positive: %0.4f\tPositiveRatio: %0.4f\t[%0.2f]" %
                  (acc_tra, acc_pos_tra, pos_ratio_tra, train_spend))
            if args.model_save:
                file = args.save_dir + '/Epoch-' + str(epoch) + '.pt'
                print('save file', file)
                torch.save(net.module.state_dict(), file)

        # #################################  val  ###################################
        st_val = time.time()
        val_scores, acc_val, acc_pos_val, pos_ratio_val, val_impids, val_labels = \
            eval(net, rank, world_size, val_loader)
        impressions = {}    # {1: {'score': [], 'lab': []}}
        for i in range(world_size):
            scores, imp_id, labs = val_scores[i], val_impids[i], val_labels[i]
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

        auc_val, mrr_val, ndcg5_val, ndcg10_val = evaluate(predicts, truths)
        if auc_val > best_val_result['auc']:
            best_val_result['auc'] = auc_val
            best_val_epoch['auc'] = epoch
        if mrr_val > best_val_result['mrr']:
            best_val_result['mrr'] = mrr_val
            best_val_epoch['mrr'] = epoch
        if ndcg5_val > best_val_result['ndcg5']:
            best_val_result['ndcg5'] = ndcg5_val
            best_val_epoch['ndcg5'] = epoch
        if ndcg10_val > best_val_result['ndcg10']:
            best_val_result['ndcg10'] = ndcg10_val
            best_val_epoch['ndcg10'] = epoch
        end_val = time.time()
        val_spend = (end_val - st_val) / 3600

        if rank == 0:
            print("Validate: AUC: %0.4f\tMRR: %0.4f\tnDCG@5: %0.4f\tnDCG@10: %0.4f\t[Val-Time: %0.2f]" %
                  (auc_val, mrr_val, ndcg5_val, ndcg10_val, val_spend))
            print('Best Result: AUC: %0.4f \tMRR: %0.4f \tNDCG@5: %0.4f \t NDCG@10: %0.4f' %
                  (best_val_result['auc'], best_val_result['mrr'], best_val_result['ndcg5'], best_val_result['ndcg10']))
            print('Best Epoch: AUC: %d \tMRR: %d \tNDCG@5: %d \t NDCG@10: %d' %
                  (best_val_epoch['auc'], best_val_epoch['mrr'], best_val_epoch['ndcg5'], best_val_epoch['ndcg10']))
        dist.barrier()
    if rank == 0:
        best_epochs = [best_val_epoch['auc'], best_val_epoch['mrr'], best_val_epoch['ndcg5'], best_val_epoch['ndcg10']]
        best_epoch = max(set(best_epochs), key=best_epochs.count)
        if args.model_save:
            old_file = args.save_dir + '/Epoch-' + str(best_epoch) + '.pt'
            if not os.path.exists('./temp'):
                os.makedirs('./temp')
            copy_file = './temp' + '/BestModel.pt'
            shutil.copy(old_file, copy_file)
            print('Copy ' + old_file + ' >>> ' + copy_file)
    cleanup()


if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../DATA/MIND-Small', type=str, help='Path')
    parser.add_argument('--model_name', default='bert-base-uncased', type=str)

    parser.add_argument('--epochs', default=5, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=24, type=int, help='batch_size')
    parser.add_argument('--test_batch_size', default=200, type=int, help='test batch_size')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')

    parser.add_argument('--max_his', default=50, type=int, help='max number of history')
    parser.add_argument('--max_tokens', default=500, type=int, help='max number of tokens')
    parser.add_argument('--num_negs', default=4, type=int, help='number of negtives')

    parser.add_argument('--max_his_len', default=450, type=int, help='max number of history')

    parser.add_argument('--num_conti1', default=3, type=int, help='number of continuous tokens')
    parser.add_argument('--num_conti2', default=3, type=int, help='number of continuous tokens')

    parser.add_argument('--device', default='cuda', help='device id')
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')

    parser.add_argument('--model_save', default=True, type=bool, help='save model file')
    parser.add_argument('--log', default=True, type=bool, help='whether write log file')

    # parser.add_argument('--model_save', default=False, type=bool, help='save model file')
    # parser.add_argument('--log', default=False, type=bool, help='whether write log file')

    args = parser.parse_args()

    if args.model_save:
        save_dir = './model_save/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        args.save_dir = save_dir

    # Create log file
    if args.data_path == '../DATA/MIND-Demo':
        if args.log:
            if not os.path.exists('./log'):
                os.makedirs('./log')
            log_file = './log/' + 'bs' + str(args.batch_size) + \
                '-Tbs' + str(args.test_batch_size) + \
                '-lr' + str(args.lr) + \
                '-n' + str(args.num_conti1) + str(args.num_conti2) + \
                '-' + str(datetime.now())[-5:]+'.txt'
            args.log_file = log_file
    else:
        if args.log:
            if not os.path.exists('./log-Small'):
                os.makedirs('./log-Small')
            log_file = './log-Small/' + 'bs' + str(args.batch_size) + \
                '-Tbs' + str(args.test_batch_size) + \
                '-lr' + str(args.lr) + \
                '-n' + str(args.num_conti1) + str(args.num_conti2) + \
                '-' + str(datetime.now())[-5:]+'.txt'
            args.log_file = log_file

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
             args=(WORLD_SIZE, args),
             nprocs=WORLD_SIZE,
             join=True)
    t1 = time.time()
    run_time = (t1 - t0) / 3600
    print('Running time: %0.4f' % run_time)





