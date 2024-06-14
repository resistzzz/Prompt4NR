import re
import random
import numpy as np
from torch.utils.data import Dataset
import pickle
import os
import torch
from collections import Counter


class MyDataset(Dataset):
    def __init__(self, args, tokenizer, news_dict, status='train'):
        self.tokenizer = tokenizer
        self.news_dict = news_dict
        self.args = args
        self.status = status

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
        return data[0], data[1], data[2], data[3]

    # the god prompt
    # TODO limit the number of topics not the history length.
    # TODO when to use [SEP] [NSEP]
    # TODO when do we have to split words up, also in list? we can tokenize a list of topics
    # TODO when to encode + decode
    # TODO something's going wrong with the split() that truncates sentences incorrectly.
    def prepro_train(self, imp_ids, behaviors, news_dict, K_samples,
                     max_his=50, max_topics=200, max_title_len=10, max_candi_len=20, max_his_len=450,
                     prompt_type='combined'):
        if prompt_type == 'combined':
            template = "Past news topics of User from in descending order of relevance : <user_topics> [SEP] Most common news sentiment of user: <user_sentiment> [SEP] News: <candidate_news> [SEP]  Does the user click the news? [MASK]"
            for impid, behav in zip(imp_ids, behaviors):
                his_clicks = behav[0]
                his_clicks.reverse()
                history_topics = []
                history_sentiment = []
                # build list of all topics in history of user
                for news in his_clicks:
                    # add topics to user topics history
                    article_topics = list(news_dict[news]['topics'])
                    history_topics += article_topics
                    # add sentiment to user sentiment history
                    sentiment = news_dict[news]['sentiment']
                    history_sentiment.append(sentiment)

                topics_counted = Counter(history_topics)
                # sort unique topics based on frequency in users history
                sorted_topics = [item[0] for item in
                                 sorted(topics_counted.items(), key=lambda item: item[1], reverse=True)][:max_topics]
                # max_his = max number of topics
                sorted_topics_ids = self.tokenizer.encode(history_topics, add_special_tokens=False)
                history_topics_tokens = self.tokenizer.decode(sorted_topics_ids)
                # very unlikely edge case in which user has no topics in its entire history
                if len(sorted_topics) != 0:
                    # string of list of strings inside prompt?
                    base_sentence = template.replace("<user_topics>", history_topics_tokens)
                # edge case: what if all of user history there are no topics for articles
                else:
                    base_sentence = template.replace("<user_topics>", ' ')
                # count most frequent sentiment and add it to prompt.
                counted_sentiments = Counter(history_sentiment)
                most_common_sentiment = counted_sentiments.most_common(1)[0][0]
                base_sentence = base_sentence.replace("<user_sentiment>", most_common_sentiment)

                positives = behav[1]
                negatives = behav[2]

                for news in positives:
                    # process title
                    title = news_dict[news]['title']
                    title = re.sub(r'[^A-Za-z0-9 ]+', '', title)
                    title = ' '.join(title.split(' '))
                    # process abstract
                    abstract = news_dict[news]['abstract']
                    abstract = re.sub(r'[^A-Za-z0-9 ]+', '', abstract)
                    abstract = ' '.join(abstract.split(' '))
                    # concat title and abstract and replace in the prompt
                    title_and_abstract = title + abstract
                    sentence = base_sentence.replace("<candidate_news>", title_and_abstract)
                    self.data.append({'sentence': sentence, 'target': 1, 'imp': impid})

                    if len(negatives) >= K_samples:
                        sample_negs = random.sample(negatives, k=K_samples)
                    else:
                        sample_negs = np.random.choice(negatives, K_samples, replace=True).tolist()
                    # same functionality as last for loop but for negative samples
                    for neg in sample_negs:
                        neg_title = news_dict[neg]['title']
                        neg_title = re.sub(r'[^A-Za-z0-9 ]+', '', neg_title)
                        neg_title = ' '.join(neg_title.split(' ')[:max_candi_len])

                        abstract = news_dict[news]['abstract']
                        abstract = re.sub(r'[^A-Za-z0-9 ]+', '', abstract)
                        abstract = ' '.join(abstract.split(' '))

                        title_and_abstract = neg_title + '[NSEP]' + abstract
                        sentence = base_sentence.replace("<candidate_news>", title_and_abstract)
                        self.data.append({'sentence': sentence, 'target': 0, 'imp': impid})
        if prompt_type == 'sentiment':
            template = "User: <user_sentence> [SEP] Most common news sentiment of user: <user_sentiment> [SEP] News: <candidate_news> [SEP] Does the user click the news? [MASK]"
            for impid, behav in zip(imp_ids, behaviors):
                his_clicks = behav[0]
                his_clicks.reverse()
                his_titles = []
                history_sentiment = []
                for news in his_clicks:
                    title = news_dict[news]['title']
                    title = re.sub(r'[^A-Za-z0-9 ]+', '', title)
                    title = ' '.join(title.split(' ')[:max_title_len])
                    his_titles.append(title)

                    sentiment = news_dict[news]['sentiment']
                    history_sentiment.append(sentiment)
                his_sen = '[NSEP] ' + ' [NSEP] '.join(his_titles)
                his_sen_ids = self.tokenizer.encode(his_sen, add_special_tokens=False)[:max_his_len]
                his_sen = self.tokenizer.decode(his_sen_ids)
                base_sentence = template.replace("<user_sentence>", his_sen)

                counted_sentiments = Counter(history_sentiment)
                most_common_sentiment = counted_sentiments.most_common(1)[0][0]
                base_sentence = base_sentence.replace("<user_sentiment>", most_common_sentiment)

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

        if prompt_type == 'topics':
            template = "Users topics: <topics> [SEP] News: <candidate_news> [SEP] covering: <candidate_topics> [SEP] Does the user click the news? [MASK]"
            for impid, behav in zip(imp_ids, behaviors):
                his_clicks = behav[0][-max_his:]
                his_clicks.reverse()
                history_topics = []
                for news in his_clicks:
                    # use keywords from title and subtitle
                    article_topics = list(news_dict[news]['topics'])
                    history_topics += article_topics
                topics_counted = Counter(history_topics)
                sorted_topics = [item[0] for item in
                                 sorted(topics_counted.items(), key=lambda item: item[1], reverse=True)][:max_topics]
                # maybe this is essential
                sorted_topics = '[NSEP] ' + ' [NSEP] '.join(sorted_topics)
                sorted_topics_ids = self.tokenizer.encode(history_topics, add_special_tokens=False)[:max_his_len]
                history_topics_tokens = self.tokenizer.decode(sorted_topics_ids)
                if len(sorted_topics) != 0:
                    # string of list of strings inside prompt?
                    base_sentence = template.replace("<topics>", history_topics_tokens)
                # edge case: what if all of user history there are no topics for articles
                else:
                    base_sentence = template.replace("<topics>", ' ')

                positives = behav[1]
                negatives = behav[2]

                for news in positives:
                    # title
                    title = news_dict[news]['title']
                    title = re.sub(r'[^A-Za-z0-9 ]+', '', title)
                    title = ' '.join(title.split(' ')[:max_candi_len])
                    sentence = base_sentence.replace("<candidate_news>", title)
                    # topics
                    article_topics = list(news_dict[news]['topics'])
                    article_topics_ids = self.tokenizer.encode(article_topics, add_special_tokens=False)[:max_his_len]
                    article_topics_tokens = self.tokenizer.decode(article_topics_ids)
                    sentence = sentence.replace('<candidate_topics>', article_topics_tokens)

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

                        # topics
                        article_topics = list(news_dict[news]['topics'])
                        article_topics_ids = self.tokenizer.encode(article_topics, add_special_tokens=False)[
                                             :max_his_len]
                        article_topics_tokens = self.tokenizer.decode(article_topics_ids)
                        sentence = sentence.replace('<candidate_topics>', article_topics_tokens)

                        self.data.append({'sentence': sentence, 'target': 0, 'imp': impid})

    def prepro_dev(self, imp_ids, behaviors, news_dict,
                   max_his=50, max_title_len=10, max_candi_len=20, max_his_len=450):
        template = "User: <user_sentence> [SEP] News: <candidate_news> [SEP] Does the user click the news? [MASK]"
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
