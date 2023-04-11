
import re
import random
import numpy as np
from torch.utils.data import Dataset
import pickle
import os
import torch


class MyDataset(Dataset):
    def __init__(self, args, tokenizer, news_dict, conti_tokens, status='train'):
        self.tokenizer = tokenizer
        self.news_dict = news_dict
        self.args = args
        self.status = status
        self.conti_tokens = conti_tokens

        self.data = []
        self.imp_lens = []
        if self.status == 'train':
            self.data_path = os.path.join(args.data_path, 'train.txt')
        elif self.status == 'val':
            self.data_path = os.path.join(args.data_path, 'val.txt')
        else:
            self.data_path = os.path.join(args.data_path, 'test.txt')
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def obtain_data(self, data):
        # if self.status == 'train':
        #     return data[0][:20], data[1][:20], data[2][:20], data[3][:20]
        # else:
        #     return data[0], data[1], data[2], data[3]
        return data[0], data[1], data[2], data[3]

    def prepro_train(self, imp_ids, behaviors, news_dict, K_samples,
                     max_his=50, max_title_len=10, max_candi_len=20, max_his_len=450):
        template1 = ''.join(self.conti_tokens[0]) + "<user_sentence>"
        template2 = ''.join(self.conti_tokens[1]) + "<candidate_news>"
        template3 = ''.join(self.conti_tokens[2]) + "[MASK]"
        template = template1 + "[SEP]" + template2 + "[SEP]" + template3

        for impid, behav in zip(imp_ids, behaviors):
            his_clicks = behav[0][-max_his:]
            his_clicks.reverse()
            his_titles = []
            for news in his_clicks:
                title = news_dict[news]['title']
                title = re.sub(r'[^A-Za-z0-9 ]+', '', title)

                title = ' '.join(title.split(' ')[:max_title_len])

                his_titles.append(title)
            his_sen = '[NSEP] ' + ' [NSEP] '.join(his_titles)
            his_sen_ids = self.tokenizer.encode(his_sen, add_special_tokens=False)[:max_his_len]
            his_sen = self.tokenizer.decode(his_sen_ids)
            base_sentence = template.replace("<user_sentence>", his_sen)

            positives = behav[1]
            negatives = behav[2]

            for news in positives:
                title = news_dict[news]['title']
                title = re.sub(r'[^A-Za-z0-9 ]+', '', title)

                title = ' '.join(title.split(' ')[:max_candi_len])

                sentence = base_sentence.replace("<candidate_news>", title)
                self.data.append({'sentence': sentence, 'target': 1, 'imp': impid})

                if len(negatives) >= K_samples:
                    sample_negs = random.sample(negatives, k=K_samples)
                else:
                    sample_negs = np.random.choice(negatives, K_samples, replace=True).tolist()

                for neg in sample_negs:
                    neg_title = news_dict[neg]['title']
                    neg_title = re.sub(r'[^A-Za-z0-9 ]+', '', neg_title)

                    neg_title = ' '.join(neg_title.split(' ')[:max_candi_len])

                    sentence = base_sentence.replace("<candidate_news>", neg_title)
                    self.data.append({'sentence': sentence, 'target': 0, 'imp': impid})

    def prepro_dev(self, imp_ids, behaviors, news_dict,
                   max_his=50, max_title_len=10, max_candi_len=20, max_his_len=450):
        template1 = ''.join(self.conti_tokens[0]) + "<user_sentence>"
        template2 = ''.join(self.conti_tokens[1]) + "<candidate_news>"
        template3 = ''.join(self.conti_tokens[2]) + "[MASK]"
        template = template1 + "[SEP]" + template2 + "[SEP]" + template3

        for impid, behav in zip(imp_ids, behaviors):
            if len(behav[0]) == 0:
                continue
            his_clicks = behav[0][-max_his:]
            his_clicks.reverse()
            his_titles = []
            for news in his_clicks:
                title = news_dict[news]['title']
                title = re.sub(r'[^A-Za-z0-9 ]+', '', title)

                title = ' '.join(title.split(' ')[:max_title_len])

                his_titles.append(title)
            his_sen = '[NSEP] ' + ' [NSEP] '.join(his_titles)
            his_sen_ids = self.tokenizer.encode(his_sen, add_special_tokens=False)[:max_his_len]
            his_sen = self.tokenizer.decode(his_sen_ids)
            base_sentence = template.replace("<user_sentence>", his_sen)

            positives = behav[1]
            negatives = behav[2]
            for news in positives:
                title = news_dict[news]['title']
                title = re.sub(r'[^A-Za-z0-9 ]+', '', title)

                title = ' '.join(title.split(' ')[:max_candi_len])

                sentence = base_sentence.replace("<candidate_news>", title)
                self.data.append({'sentence': sentence, 'target': 1, 'imp': impid})

            for neg in negatives:
                neg_title = news_dict[neg]['title']
                neg_title = re.sub(r'[^A-Za-z0-9 ]+', '', neg_title)

                neg_title = ' '.join(neg_title.split(' ')[:max_candi_len])

                sentence = base_sentence.replace("<candidate_news>", neg_title)
                self.data.append({'sentence': sentence, 'target': 0, 'imp': impid})

    def load_data(self):
        data = pickle.load(open(self.data_path, 'rb'))
        imps, users, times, behaviors = self.obtain_data(data)
        if self.status == 'train':
            self.prepro_train(imps, behaviors, self.news_dict, self.args.num_negs, self.args.max_his,
                              max_his_len=self.args.max_his_len)
        else:
            self.prepro_dev(imps, behaviors, self.news_dict, self.args.max_his,
                            max_his_len=self.args.max_his_len)

    def collate_fn(self, batch):
        sentences = [x['sentence'] for x in batch]
        target = [x['target'] for x in batch]
        imp_id = [x['imp'] for x in batch]

        encode_dict = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.args.max_tokens,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        batch_enc = encode_dict['input_ids']
        batch_attn = encode_dict['attention_mask']
        target = torch.LongTensor(target)

        return batch_enc, batch_attn, target, imp_id





