import copy
import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

class BaseHidden:
    def __init__(self, num_class, means, covariance):
        self.num_class = num_class
        self.means = means
        self.covariance = covariance
        self.train_data = None
    
    def generate_data_base_on_means_and_cov(self, labels, mean, covariance, num):
        res = []
        data = np.random.multivariate_normal(mean, covariance, num)
        for sample in data:
            ins = {
                'input_ids':sample.tolist(),
                'labels': labels,
                'float_check': True,
            }
            res.append(ins)
        return res

    def generate_hidden_data(self):
        res = []
        for idx in range(self.num_class):
            labels = idx
            mean = self.means[idx].cpu().numpy()
            cov = self.covariance.cpu().numpy()
            samples = self.generate_data_base_on_means_and_cov(labels, mean, cov, 960)
            res.extend(samples)
        return res
            

class BaseData:
    def __init__(self, args):
        self.args = args
        self.label_list = self._read_labels()
        self.id2label, self.label2id = [], {}
        self.label2task_id = {}
        self.train_data, self.val_data, self.test_data = None, None, None

    def _read_labels(self):
        """
        :return: only return the label name, in order to set label index from 0 more conveniently.
        """
        id2label = json.load(open(os.path.join(self.args.data_path, self.args.dataset_name, 'id2label.json')))
        return id2label

    def read_and_preprocess(self, **kwargs):
        raise NotImplementedError

    def add_labels(self, cur_labels, task_id):
        for c in cur_labels:
            if c not in self.id2label:
                self.id2label.append(c)
                self.label2id[c] = len(self.label2id)
                self.label2task_id[self.label2id[c]] = task_id

    def filter(self, labels, split='train'):
        if not isinstance(labels, list):
            labels = [labels]
        split = split.lower()
        res = []
        for label in labels:
            if split == 'train':
                if self.args.debug:
                    res += copy.deepcopy(self.train_data[label])[:10]
                else:
                    res += copy.deepcopy(self.train_data[label])
            elif split in ['dev', 'val']:
                if self.args.debug:
                    res += copy.deepcopy(self.val_data[label])[:10]
                else:
                    res += copy.deepcopy(self.val_data[label])
            elif split == 'test':
                if self.args.debug:
                    res += copy.deepcopy(self.test_data[label])[:10]
                else:
                    res += copy.deepcopy(self.test_data[label])
        for idx in range(len(res)):
            res[idx]["labels"] = self.label2id[res[idx]["labels"]]
        return res
    
    def get_random_positive_samples_by_label(self, label, k):
        """
        Get k random samples from a specified label.
        :param label: The label to filter samples by.
        :param k: The number of random samples to retrieve.
        :return: A list of k samples that have the specified label.
        """
        # Filter data based on the specified label

        # Randomly sample k elements from the filtered data
        random_samples = random.sample(self.train_data[label], k)
        return random_samples
    
    def filter_and_contrastive_learning_and_add_desciption(self, labels, descriptions):
        if not isinstance(labels, list):
            labels = [labels]
        # labels_label2id = [self.label2id[label_] for label_ in labels]
        print(labels)
        res = []
        for label in labels:
            pools = descriptions[label]
            sub_res = []
            for idxxxx, anchor in enumerate(self.train_data[label]):
                sub_sub_res = []
                # print(label)
                # if idxxx:
                #     print("-------")
                # for key, value in anchor.items():
                #     print(f"  {key}: {value}")
                    
                cur_label = anchor["labels"]
                if cur_label in ['P26', 'P3373', 'per:siblings', 'org:alternate_names', 'per:spouse',
                                        'per:alternate_names', 'per:other_family']:
                    continue
                negative_samples = []
                positive_samples = []
                for other_label in labels:
                    if other_label == label:
                        continue
                    new_negative = self.get_random_positive_samples_by_label(other_label, 1)
                    new_positive = self.get_random_positive_samples_by_label(label, 1)
                    for ins in new_negative:
                        negative_samples.append(ins)
                    for ins in new_positive:
                        positive_samples.append(ins)
                
                random.shuffle(negative_samples)
                random.shuffle(positive_samples)
                
                for idx in range(min(len(negative_samples), len(positive_samples), len(pools))):
                    negative_sample = negative_samples[idx]
                    positive_sample = positive_samples[idx]
                    descriptions_ids = pools[idx]
                    ins = {
                        'input_ids': anchor['input_ids'],  # default: add marker to the head entity and tail entity
                        'subject_marker_st': anchor['subject_marker_st'],
                        'object_marker_st': anchor['object_marker_st'],
                        'labels': anchor['labels'],
                        'input_ids_without_marker': anchor['input_ids_without_marker'],
                        'subject_st': anchor['subject_st'],
                        'subject_ed': anchor['subject_ed'],
                        'object_st': anchor['object_st'],
                        'object_ed': anchor['object_ed'],
                        'descriptions_ids': descriptions_ids,
                        
                        'negative_input_ids': negative_sample['input_ids'],  # default: add marker to the head entity and tail entity
                        'negative_subject_marker_st': negative_sample['subject_marker_st'],
                        'negative_object_marker_st': negative_sample['object_marker_st'],
                        'negative_subject_st': negative_sample['subject_st'],
                        'negative_subject_ed': negative_sample['subject_ed'],
                        'negative_object_st': negative_sample['object_st'],
                        'negative_object_ed': negative_sample['object_ed'],
                        
                        'positive_input_ids': positive_sample['input_ids'],  # default: add marker to the head entity and tail entity
                        'positive_subject_marker_st': positive_sample['subject_marker_st'],
                        'positive_object_marker_st': positive_sample['object_marker_st'],
                        'positive_subject_st': positive_sample['subject_st'],
                        'positive_subject_ed': positive_sample['subject_ed'],
                        'positive_object_st': positive_sample['object_st'],
                        'positive_object_ed': positive_sample['object_ed'],
                    }
                    sub_sub_res.append(ins)
                sub_res += sub_sub_res
            res += sub_res
        for idx in range(len(res)):
            res[idx]["labels"] = self.label2id[res[idx]["labels"]]
        return res
    
    def filter_and_add_desciption(self, labels, descriptions):
        if not isinstance(labels, list):
            labels = [labels]
        # labels_label2id = [self.label2id[label_] for label_ in labels]
        print(labels)
        res = []
        for label in labels:
            pools = {}
            if label in descriptions.keys():
                pools = descriptions[label]
            sub_res = []
            for anchor in self.train_data[label]:
                # print(label)
                # if idxxx:
                #     print("-------")
                # for key, value in anchor.items():
                #     print(f"  {key}: {value}")
                    
                cur_label = anchor["labels"]
                if cur_label in ['P26', 'P3373', 'per:siblings', 'org:alternate_names', 'per:spouse',
                                        'per:alternate_names', 'per:other_family']:
                    continue
                
                ins = {
                    'input_ids': anchor['input_ids'],  # default: add marker to the head entity and tail entity
                    'subject_marker_st': anchor['subject_marker_st'],
                    'object_marker_st': anchor['object_marker_st'],
                    'labels': anchor['labels'],
                    'input_ids_without_marker': anchor['input_ids_without_marker'],
                    'subject_st': anchor['subject_st'],
                    'subject_ed': anchor['subject_ed'],
                    'object_st': anchor['object_st'],
                    'object_ed': anchor['object_ed'],
                }
                
                for idx, pool in enumerate(pools):
                    ins.update({
                        f'description_ids_{idx}': pool
                    })
                sub_res.append(ins)
            res += sub_res
        for idx in range(len(res)):
            res[idx]["labels"] = self.label2id[res[idx]["labels"]]
        return res

    def filler_add_old_description(self, labels, descriptions, num):
        if not isinstance(labels, list):
            labels = [labels]
            
        # labels_label2id = [self.label2id[label_] for label_ in labels]
        print(labels)
        res = []
        for label in labels:
            pools = {}
            if label in descriptions.keys():
                pools = descriptions[label]
            sub_res = []
            cur_label = self.label2id[label]
            if cur_label in ['P26', 'P3373', 'per:siblings', 'org:alternate_names', 'per:spouse',
                                        'per:alternate_names', 'per:other_family']:
                    continue
            for i in range(num):                                  
                for pool in pools:
                    ins = {
                        'input_ids': pool,
                        'labels': cur_label,
                        'mlp1_term2': True,
                    }
                    sub_res.append(ins)
            res += sub_res
        return res
    
    def filter_and_add_desciption_and_old_description(self, labels, descriptions, seen_labels, old_descriptions):
        if not isinstance(labels, list):
            labels = [labels]
        print(labels)
        res = []
        lenght_seen_labels = len(seen_labels)
        count_negative_label = 0
        for label in labels:
            pools = {}
            if label in descriptions.keys():
                pools = descriptions[label]
            sub_res = []
            for anchor in self.train_data[label]:
                cur_label = anchor["labels"]
                if cur_label in ['P26', 'P3373', 'per:siblings', 'org:alternate_names', 'per:spouse',
                                        'per:alternate_names', 'per:other_family']:
                    continue
                
                ins = {
                    'input_ids': anchor['input_ids'],  # default: add marker to the head entity and tail entity
                    'subject_marker_st': anchor['subject_marker_st'],
                    'object_marker_st': anchor['object_marker_st'],
                    'labels': anchor['labels'],
                    'input_ids_without_marker': anchor['input_ids_without_marker'],
                    'subject_st': anchor['subject_st'],
                    'subject_ed': anchor['subject_ed'],
                    'object_st': anchor['object_st'],
                    'object_ed': anchor['object_ed'],
                }
                
                for idx, pool in enumerate(pools):
                    ins.update({
                        f'description_ids_{idx}': pool
                    })
                if lenght_seen_labels != 0:
                    old_labels = seen_labels[count_negative_label%lenght_seen_labels]
                    ins.update({
                            'old_labels': self.label2id[old_labels]
                        })
                    old_pools = {}
                    if old_labels in old_descriptions.keys():
                        old_pools = old_descriptions[old_labels]
                        
                    for idx, pool in enumerate(old_pools):
                        ins.update({
                            f'old_description_ids_{idx}': pool
                        })
                
                count_negative_label += 1
                sub_res.append(ins)
            res += sub_res
        for idx in range(len(res)):
            res[idx]["labels"] = self.label2id[res[idx]["labels"]]
        return res

class BaseDataset(Dataset):
    def __init__(self, data):
        if isinstance(data, dict):
            res = []
            for key in data.keys():
                res += data[key]
            data = res
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # cur_data = self.data[idx]
        # cur_data["idx"] = idx
        # mask_head = True if random.random() > 0.5 else False
        # input_ids, attention_mask, subject_start_pos, object_start_pos = mask_entity(cur_data["input_ids"], mask_head)
        # augment_data = {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "subject_start_pos": subject_start_pos,
        #     "object_start_pos": object_start_pos,
        #     "labels": cur_data["labels"],
        #     "idx": idx
        # }
        # return [cur_data, augment_data]
        return self.data[idx]


class BaseTripletDataset(Dataset):
    def __init__(self, data, len_seen_labels, len_cur_labels):
        
        if isinstance(data, dict):
            res = []
            for key in data.keys():
                res += data[key]
            data = res
        self.data = data
        self.len_seen_labels = len_seen_labels
        self.len_cur_labels = len_cur_labels
        self.cur_labels = self.preprocess_label()
        self.data = self.convert_into_triplets()
    
    def preprocess_label(self):
        list_label = [(self.len_seen_labels-1-i) for i in range(self.len_cur_labels)]
        return list_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # cur_data = self.data[idx]
        # cur_data["idx"] = idx
        # mask_head = True if random.random() > 0.5 else False
        # input_ids, attention_mask, subject_start_pos, object_start_pos = mask_entity(cur_data["input_ids"], mask_head)
        # augment_data = {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "subject_start_pos": subject_start_pos,
        #     "object_start_pos": object_start_pos,
        #     "labels": cur_data["labels"],
        #     "idx": idx
        # }
        # return [cur_data, augment_data]
        return self.data[idx]
    
    def convert_into_triplets(self):
        # print("Lenght_old: ", len(self.data))
        new_data = []
        for iddxx, anchor in enumerate(self.data):
            anchor_labels = anchor['labels']
            negative_samples = []
            positive_samples = []
            for label in self.cur_labels:
                if label == anchor_labels:
                    continue
                new_negative = self.get_random_positive_samples_by_label(label, 1)
                new_positive = self.get_random_positive_samples_by_label(anchor_labels, 1)
                for ins in new_negative:
                    negative_samples.append(ins)
                for ins in new_positive:
                    positive_samples.append(ins)
            for idx in range(min(len(negative_samples), len(positive_samples))):
                
                negative_sample = negative_samples[idx]
                positive_sample = positive_samples[idx]
                ins = {
                    'input_ids': anchor['input_ids'],  # default: add marker to the head entity and tail entity
                    'subject_marker_st': anchor['subject_marker_st'],
                    'object_marker_st': anchor['object_marker_st'],
                    'labels': anchor['labels'],
                    'input_ids_without_marker': anchor['input_ids_without_marker'],
                    'subject_st': anchor['subject_st'],
                    'subject_ed': anchor['subject_ed'],
                    'object_st': anchor['object_st'],
                    'object_ed': anchor['object_ed'],
                    
                    'negative_input_ids': negative_sample['input_ids'],  # default: add marker to the head entity and tail entity
                    'negative_subject_marker_st': negative_sample['subject_marker_st'],
                    'negative_object_marker_st': negative_sample['object_marker_st'],
                    'negative_subject_st': negative_sample['subject_st'],
                    'negative_subject_ed': negative_sample['subject_ed'],
                    'negative_object_st': negative_sample['object_st'],
                    'negative_object_ed': negative_sample['object_ed'],
                    
                    'positive_input_ids': positive_sample['input_ids'],  # default: add marker to the head entity and tail entity
                    'positive_subject_marker_st': positive_sample['subject_marker_st'],
                    'positive_object_marker_st': positive_sample['object_marker_st'],
                    'positive_subject_st': positive_sample['subject_st'],
                    'positive_subject_ed': positive_sample['subject_ed'],
                    'positive_object_st': positive_sample['object_st'],
                    'positive_object_ed': positive_sample['object_ed'],
                    
                }
                new_data.append(ins)
        # print("Lenght_new: ", len(new_data))
        return new_data    
    
    def get_random_positive_samples_by_label(self, label, k):
        """
        Get k random samples from a specified label.
        :param label: The label to filter samples by.
        :param k: The number of random samples to retrieve.
        :return: A list of k samples that have the specified label.
        """
        # Filter data based on the specified label
        filtered_data = [item for item in self.data if item['labels'] == label]

        # If k is greater than the number of available samples, return all the samples
        if k > len(filtered_data):
            return filtered_data

        # Randomly sample k elements from the filtered data
        random_samples = random.sample(filtered_data, k)
        return random_samples
    
    