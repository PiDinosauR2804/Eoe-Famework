import copy
import logging
import os
import pickle

import hydra
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
from sklearn import metrics
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed
import wandb_logger as loggerdb


from data import BaseDataset, BaseTripletDataset, BaseHidden
from trainers import BaseTrainer
from utils import CustomCollatorWithPadding, CustomFloatCollatorWithPadding, relation_data_augmentation, relation_data_augmentation_and_add_old_descriptions

logger = logging.getLogger(__name__)


class EoETrainer(BaseTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.task_idx = 0
        self.cur_seed = 0
        

    def run(self, data, model, tokenizer, label_order, seed=None):
        if seed is not None:
            set_seed(seed)
            self.cur_seed = seed
        default_data_collator = CustomCollatorWithPadding(tokenizer)
        float_data_collator = CustomFloatCollatorWithPadding(tokenizer)

        seen_labels = []
        all_cur_acc = []
        all_total_acc = []
        all_total_hit = []
        marker_ids = tuple([tokenizer.convert_tokens_to_ids(c) for c in self.args.additional_special_tokens])
        logger.info(f"marker ids: {marker_ids}")
        for task_idx in range(self.args.num_tasks):
            self.task_idx = task_idx
            cur_labels = [data.label_list[c] for c in label_order[task_idx]]
            data.add_labels(cur_labels, task_idx)

            logger.info(f"***** Task-{task_idx + 1} *****")
            logger.info(f"Current classes: {' '.join(cur_labels)}")
            
            train_data_old = data.filter(cur_labels, "train") 
            
            for cur_label in cur_labels:
                model.take_generate_description_genai_from_file(cur_label, self.args.dataset_name, tokenizer)
                # model.take_generate_description_MrLinh_from_file(cur_label, data.label2id[cur_label], self.args.dataset_name, tokenizer)
                
            pool = model.get_description_ids(cur_labels)
            old_pool = model.get_description_ids(seen_labels)
            train_data = data.filter_and_add_desciption_and_old_description(cur_labels, pool, seen_labels, old_pool) 
            
            # sample = train_data[0]
            # print("Anchor Sample:")
            # for key, value in sample.items():
            #     print(f"  {key}: {value}") 
            
            
            aug_train_data, num_train_labels = relation_data_augmentation_and_add_old_descriptions(
                    copy.deepcopy(train_data), len(seen_labels), copy.deepcopy(data.id2label), marker_ids, self.args.augment_type
                )   
            
            # sample = train_data[0]
            # print("Anchor Sample:")
            # for key, value in sample.items():
            #     print(f"  {key}: {value}") 
            
            num_train_labels = len(cur_labels)

            # train_dataset = BaseDataset(train_data)
            train_dataset = BaseDataset(aug_train_data)
            train_dataset_old = BaseDataset(train_data_old)     
            
            seen_labels += cur_labels
            
            # pool = model.get_description_ids(seen_labels)       
            # model.description_matrix = self.calculation_description_matrix(model, seen_labels, pool, model.number_description, default_data_collator)
            # print(model.description_matrix)
            
            model.new_task(num_train_labels)
            # print("2")
            # print(tokenizer.vocab_size)
            if self.task_idx == 0:
                expert_model = f"./ckpt/{self.args.dataset_name}_{seed}_{self.args.augment_type}.pth"
                # expert_model = f"/content/drive/MyDrive/FewRel_2021_all.pth"
                model.load_expert_model(expert_model)
                logger.info(f"load first task model from {expert_model}")
            else:
                self.train(
                    model=model,
                    train_dataset=train_dataset,
                    data_collator=default_data_collator
                )
                
            self.statistic(model, train_dataset_old, default_data_collator)
            
            # print(model.num_labels)
            
            
            # print(model.un_expert_distribution['class_mean'])
            # print(model.un_expert_distribution['accumulate_cov_shared'])
            baseUnHidden = BaseHidden(model.num_labels, model.un_expert_distribution['class_mean'], model.un_expert_distribution['accumulate_cov_shared'])
            un_hidden_data = baseUnHidden.generate_hidden_data()
            un_hidden_dataset = BaseDataset(un_hidden_data)  
                
            self.train_mlp(
                model=model,
                train_dataset=un_hidden_dataset,
                data_collator=float_data_collator,
                training_mlp2=True
            )      

            # print(model.classifier[-1].weight)
            # print("----------------------instructed representation----------------------")
            # print(model.in_expert_distribution['class_mean'])
            # print(model.in_expert_distribution['accumulate_cov_shared'])
            
            # print("----------------------uninstructed representation----------------------")
            # print(model.un_expert_distribution['class_mean'])
            # print(model.un_expert_distribution['accumulate_cov_shared'])
            baseInHidden = BaseHidden(model.num_labels, model.in_expert_distribution['class_mean'], model.in_expert_distribution['accumulate_cov_shared'])
            in_hidden_data = baseInHidden.generate_hidden_data()
            in_hidden_dataset = BaseDataset(in_hidden_data)  
                
            self.train_mlp(
                model=model,
                train_dataset=in_hidden_dataset,
                data_collator=float_data_collator,
                training_mlp2=False
            ) 
            
            
            os.makedirs(f"./ckpt/{self.args.dataset_name}-{seed}-{self.args.augment_type}", exist_ok=True)
            model.save_classifier(
                idx=self.task_idx,
                save_dir=f"./ckpt/{self.args.dataset_name}-{seed}-{self.args.augment_type}",
            )

            model.feature_extractor.save_and_load_all_adapters(
                self.task_idx,
                save_dir=f"./ckpt/{self.args.dataset_name}-{seed}-{self.args.augment_type}",
                save=True,
            )
            cur_test_data = data.filter(cur_labels, 'test')
            history_test_data = data.filter(seen_labels, 'test')

            cur_test_dataset = BaseDataset(cur_test_data)
            history_test_dataset = BaseDataset(history_test_data)

            cur_acc, cur_hit = self.eval(
                model=model,
                eval_dataset=cur_test_dataset,
                data_collator=default_data_collator,
                seen_labels=seen_labels,
                label2task_id=copy.deepcopy(data.label2task_id), 
                oracle=True,
            )

            total_acc, total_hit = self.eval(
                model=model,
                eval_dataset=history_test_dataset,
                data_collator=default_data_collator,
                seen_labels=seen_labels,
                label2task_id=copy.deepcopy(data.label2task_id),
            )

            all_cur_acc.append(cur_acc)
            all_total_acc.append(total_acc)
            all_total_hit.append(total_hit)
            loggerdb.log_metrics({"train/all_cur_acc": cur_acc})
            loggerdb.log_metrics({"train/all_total_acc": total_acc})
            loggerdb.log_metrics({"train/all_total_hit": total_hit})

        # save distribution
        save_data = {
            "distribution": model.un_expert_distribution,
            "seen_labels": seen_labels,
            "label2id": data.label2id,
        }
        save_file = f"{self.cur_seed}_distribution.pickle"
        save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        with open(save_dir + "/" + save_file, 'wb') as file:
            pickle.dump(save_data, file)

        return {
            "cur_acc": all_cur_acc,
            "total_acc": all_total_acc,
            "total_hit": all_total_hit,
        }
        

    def train(self, model, train_dataset, data_collator):
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=data_collator
        )
        len_dataloader = len(train_dataloader)
        num_examples = len(train_dataset)
        max_steps = len_dataloader * self.args.num_train_epochs

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Train batch size = {self.args.train_batch_size}")
        logger.info(f"  Total optimization steps = {max_steps}")

        no_decay = ["bias", "LayerNorm.weight"]
        parameters = [
            {'params': [p for n, p in model.named_parameters() if 'feature_extractor' in n and not any(nd in n for nd in no_decay)],
             'lr': self.args.learning_rate, 'weight_decay': 1e-2},
            {'params': [p for n, p in model.named_parameters() if 'feature_extractor' in n and any(nd in n for nd in no_decay)],
             'lr': self.args.learning_rate, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if 'feature_extractor' not in n and not any(nd in n for nd in no_decay)],
             'lr': self.args.classifier_learning_rate, 'weight_decay': 1e-2},
            {'params': [p for n, p in model.named_parameters() if 'feature_extractor' not in n and any(nd in n for nd in no_decay)],
             'lr': self.args.classifier_learning_rate, 'weight_decay': 0.0},
        ]
        self.optimizer = AdamW(parameters)

        progress_bar = tqdm(range(max_steps))
        for name, param in model.named_parameters():
            if param.requires_grad and "lora_" in name:
                print(name)
                break

        for epoch in range(self.args.num_train_epochs):
            model.train()
            for step, inputs in enumerate(train_dataloader):
                self.optimizer.zero_grad()

                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                self.optimizer.step()
                
                # self.optimizer.zero_grad()

                # inputs['use_origin'] = True
                # outputs = model(**inputs)
                # loss = outputs.loss
                # loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                # self.optimizer.step()

                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": loss.item()})

        progress_bar.close()
        
    def train_mlp(self, model, train_dataset, data_collator, training_mlp2):
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=data_collator
        )
        len_dataloader = len(train_dataloader)
        num_examples = len(train_dataset)
        max_steps = len_dataloader * self.args.num_train_epochs

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Train batch size = {self.args.train_batch_size}")
        logger.info(f"  Total optimization steps = {max_steps}")

        no_decay = ["bias", "LayerNorm.weight"]
        parameters = [
            {'params': [p for n, p in model.named_parameters() if 'feature_extractor' in n and not any(nd in n for nd in no_decay)],
             'lr': self.args.learning_rate, 'weight_decay': 1e-2},
            {'params': [p for n, p in model.named_parameters() if 'feature_extractor' in n and any(nd in n for nd in no_decay)],
             'lr': self.args.learning_rate, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if 'feature_extractor' not in n and not any(nd in n for nd in no_decay)],
             'lr': self.args.classifier_learning_rate, 'weight_decay': 1e-2},
            {'params': [p for n, p in model.named_parameters() if 'feature_extractor' not in n and any(nd in n for nd in no_decay)],
             'lr': self.args.classifier_learning_rate, 'weight_decay': 0.0},
        ]
        self.optimizer = AdamW(parameters)

        progress_bar = tqdm(range(max_steps))
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
                break

        for epoch in range(self.args.num_train_epochs):
            model.train()
            for step, inputs in enumerate(train_dataloader):
                self.optimizer.zero_grad()

                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                inputs.update({"training_mlp": True})
                if training_mlp2:
                    inputs.update({"mlp2": True})
                else:
                    inputs.update({"mlp1": True})

                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                self.optimizer.step()


                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": loss.item()})
        
        progress_bar.close()

    @torch.no_grad()
    def calculation_description_matrix(self, model, labels, description_ids, number_description_per_label, data_collator):
        description_ids_data = []
        for label in labels:
            if label in description_ids.keys():
                for v in description_ids[label]:
                    ins = {
                      "input_ids": v,
                    }
                    description_ids_data.append(ins)
        
        description_data = BaseDataset(description_ids_data)
        loader = DataLoader(
            description_data,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )
        model.eval()
        prelogits = []
        labels = []

        for step, inputs in enumerate(loader):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            inputs.update({"return_hidden_states_by_cls": True})

            prelogit = model(**inputs)

            prelogits.extend(prelogit.tolist())
        
        prelogits = np.array(prelogits)
        prelogits = prelogits.reshape(-1, number_description_per_label, model.query_size)
        prelogits = prelogits.mean(axis=1)
        prelogits = prelogits.reshape(-1, model.query_size)
        
        prelogits = torch.tensor(prelogits)
        cosine_distance_matrix = cdist(prelogits, prelogits, metric='cosine') * 20
        
        return torch.tensor(cosine_distance_matrix)
      

    @torch.no_grad()
    def eval(self, model, eval_dataset, data_collator, seen_labels, label2task_id, oracle=False):
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

        len_dataloader = len(eval_dataloader)
        num_examples = len(eval_dataset)

        logger.info("***** Running evaluating *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Eval batch size = {self.args.eval_batch_size}")

        progress_bar = tqdm(range(len_dataloader))

        golds = []
        preds = []
        pred_indices = []
        gold_indices = []
        # expert_task_preds = []
        # expert_class_preds = []
        # hits = 0
        model.eval()
        for step, inputs in enumerate(eval_dataloader):

            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            if oracle:
                inputs.update({"oracle": True, "task_idx": self.task_idx})
            outputs = model(**inputs)

            hit_pred = outputs.indices
            hit_gold = [label2task_id[c] for c in inputs["labels"].tolist()]
            pred_indices.extend(hit_pred)
            gold_indices.extend(hit_gold)

            predicts = outputs.preds.tolist()
            labels = inputs["labels"].tolist()
            golds.extend(labels)
            preds.extend(predicts)

            # expert_task_preds.append(outputs.expert_task_preds)
            # expert_class_preds.append(outputs.expert_class_preds)

            progress_bar.update(1)
        progress_bar.close()

        logger.info("\n" + metrics.classification_report(golds, preds))
        acc = metrics.accuracy_score(golds, preds)
        hit_acc = metrics.accuracy_score(gold_indices, pred_indices)
        logger.info("Acc {}".format(acc))
        logger.info("Hit Acc {}".format(hit_acc))

        if not oracle:
            # expert_task_preds = torch.cat(expert_task_preds, dim=0).tolist()
            # expert_class_preds = torch.cat(expert_class_preds, dim=0).tolist()
            save_data = {
                "preds": preds,
                "golds": golds,
                "pred_indices": pred_indices,
                "gold_indices": gold_indices,
                # "expert_task_preds": expert_task_preds,
                # "expert_class_preds": expert_class_preds,
            }
            # save information
            save_file = f"{self.cur_seed}_{self.task_idx}.pickle"
            save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            with open(save_dir + "/" + save_file, 'wb') as file:
                pickle.dump(save_data, file)

        return acc, hit_acc

    def statistic(self, model, dataset, data_collator):
        un_mean, un_cov, un_task_mean, un_task_cov = self.get_mean_and_cov(model, dataset, data_collator, False, self.task_idx)
        model.new_statistic_uninstructed_representation(un_mean, un_cov, un_task_mean, un_task_cov, self.task_idx)
        
        in_mean, in_cov, in_task_mean, in_task_cov = self.get_mean_and_cov(model, dataset, data_collator, True, self.task_idx)
        model.new_statistic_instructed_representation(in_mean, in_cov, in_task_mean, in_task_cov, self.task_idx)

    @torch.no_grad()
    def get_mean_and_cov(self, model, dataset, data_collator, instructed_representation, expert_id=0):
        loader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )
        model.eval()

        prelogits = []
        labels = []

        for step, inputs in enumerate(loader):
            label = inputs.pop('labels')
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            if instructed_representation:
                inputs.update({"instructed_representation": True})
            inputs.update({"return_hidden_states": True})
            inputs.update({"task_idx": expert_id})

            prelogit = model(**inputs)

            prelogits.extend(prelogit.tolist())
            labels.extend(label.tolist())

        prelogits = torch.tensor(prelogits)
        labels = torch.tensor(labels)
        labels_space = torch.unique(labels)

        task_mean = prelogits.mean(dim=0)
        task_cov = torch.cov((prelogits - task_mean).T)

        mean_over_classes = []
        cov_over_classes = []
        for c in labels_space:
            embeds = prelogits[labels == c]
            if embeds.numel() > 0:
                mean = embeds.mean(dim=0)
                cov = torch.cov((embeds - mean).T)
            else:
                mean = task_mean
                cov = task_cov
            mean_over_classes.append(mean)
            cov_over_classes.append(cov)

        mean_over_classes = torch.stack(mean_over_classes)
        shared_cov = torch.stack(cov_over_classes).mean(dim=0)

        return mean_over_classes, shared_cov, task_mean, task_cov
