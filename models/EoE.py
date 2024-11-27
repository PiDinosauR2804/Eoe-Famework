import copy
import math
import json
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple
import google.generativeai as genai
import wandb_logger as loggerdb



import re
import string

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import PeftFeatureExtractor
from utils import mahalanobis
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


class EoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.class_per_task = config.class_per_task
        self.default_expert = config.default_expert
        self.peft_type = config.peft_type
        self.query_mode = config.query_mode
        self.max_expert = config.max_expert if config.max_expert != -1 else float("inf")
        self.tau = 0.8
        self.feature_extractor = PeftFeatureExtractor(config)
        
        self.num_old_labels = 0
        self.num_labels = 0
        self.num_tasks = -1
        # self.overlap_label2task = None

        self.classifier_hidden_size = self.feature_extractor.bert.config.hidden_size
        self.query_size = self.feature_extractor.bert.config.hidden_size
        if config.task_name == "RelationExtraction":
            self.classifier_hidden_size = 2 * self.feature_extractor.bert.config.hidden_size
            self.query_size = 2 * self.feature_extractor.bert.config.hidden_size

        self.dropout = nn.Dropout(self.feature_extractor.bert.config.hidden_dropout_prob)
        self.n_layer = self.feature_extractor.bert.config.num_hidden_layers
        self.n_head = self.feature_extractor.bert.config.num_attention_heads
        self.n_embd = self.feature_extractor.bert.config.hidden_size // self.feature_extractor.bert.config.num_attention_heads
        self.hidden_size = self.feature_extractor.bert.config.hidden_size

        # 0-bert 1-10 task
        # self.expert_distribution = [
        #     {
        #         "class_mean": [],
        #         "accumulate_cov": torch.zeros(self.query_size, self.query_size),
        #         "cov_inv": torch.ones(self.query_size, self.query_size),
        #     }
        # ]
        self.expert_distribution = {
                "class_mean": [],
                "accumulate_cov": torch.zeros(self.query_size, self.query_size),
                "cov_inv": torch.ones(self.query_size, self.query_size),
            }
        self.label_description = {}
        self.label_description_ids = {}
        self.number_description = 3
        self.classifier = nn.ParameterList()
        self.classifier_only_bert = nn.ParameterList()
        
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    def generate_description_genai(self, label, dataset_name, tokenizer):
        if dataset_name.lower() == 'fewrel':
            file_path = 'datasets/FewRel/pid2name.json'
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
        
        label_name = data[label][0]
        
        genai.configure(api_key="AIzaSyBkNokklYsVbymqhuS15YCiM-XrMjyz9CE")
        model = genai.GenerativeModel("gemini-1.5-flash")
        pool = []
        for i in range(self.number_description):
            prompt = f"Describe the label '{label_name}' in your own words, maximum length is 50 words, focusing on originality: "
            response = model.generate_content(prompt)
            pool.append(response.text)
        
        # Lưu mô tả nhãn vào label_description        
        self.label_description[label] = [self.preprocess_text(desc) for desc in pool]
        self.label_description_ids[label] = [self.preprocess_tokenize_desciption(desc, tokenizer) for desc in self.label_description[label]]

    def generate_description(self, label, dataset_name, tokenizer):
        if dataset_name.lower() == 'fewrel':
            file_path = 'datasets/FewRel/pid2name.json'
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
        
        label_name = data[label][0]
        model_name = "gpt2"  # Bạn có thể thay thế bằng một mô hình ngôn ngữ mã nguồn mở khác
        tokenizer1 = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer1)
        
        prompt = f"Describe the label '{label_name}' in a simple and detailed way: "
        descriptions = generator(prompt, 
                                 
                                 max_length=50, num_return_sequences=1)
        
        # Lưu mô tả nhãn vào label_description        
        self.label_description[label] = [self.preprocess_text(desc['generated_text'].replace(prompt, '').strip()) for desc in descriptions]
        self.label_description_ids[label] = [self.preprocess_tokenize_desciption(desc, tokenizer) for desc in self.label_description[label]]

    def generate_description_from_file(self, label, dataset_name, tokenizer):
        if dataset_name.lower() == 'fewrel':
            file_path = 'datasets/FewRel/pid2name.json'
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                        
        # Lưu mô tả nhãn vào label_description
        # self.label_description_ids[label] = [self.preprocess_desciption(desc, tokenizer) for desc in data[label]]
        # self.label_description[label] = [desc for desc in data[label]]
        
        
        self.label_description_ids[label] = [self.preprocess_tokenize_desciption(data[label][-1], tokenizer)]
        self.label_description[label] = [data[label][-1]]
        
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9.,?!()\s]', '', text)
        text = text.strip()
        
        return text    
        
    def preprocess_tokenize_desciption(self, raw_text, tokenizer):
        result = tokenizer(raw_text)
        return result['input_ids']
        
    def get_description(self, labels):
        pool = {}
        for label in labels:
            pool[label] = copy.deepcopy(self.label_description[label])
        return pool
    
    def get_description_ids(self, labels):
        pool = {}
        for label in labels:
            if label in self.label_description_ids.keys():
                pool[label] = copy.deepcopy(self.label_description_ids[label])
            else:
                print("Not Found")
        return pool

    def load_expert_model(self, expert_model):
        ckpt = torch.load(expert_model)
        self.feature_extractor.bert.load_state_dict(ckpt["model"])
        num_class = self.classifier[0].weight.shape[0]
        self.classifier[0].weight.data = ckpt["linear"]["weight"].data[:num_class].clone()
        self.classifier[0].bias.data = ckpt["linear"]["bias"].data[:num_class].clone()

    def new_task(self, num_labels):
        self.num_tasks += 1
        self.num_labels += num_labels
        if self.num_tasks > 0:
            self.num_old_labels += self.class_per_task
        # freeze previous classifier and add new classifier for new task
        for param in self.classifier.parameters():
            param.requires_grad = False
            
        for param in self.classifier_only_bert.parameters():
            param.requires_grad = False
            
        new_output_size = self.num_old_labels + num_labels
        new_classifier = nn.Linear(self.classifier_hidden_size, new_output_size, device=self.device)
        # new_classifier = nn.Linear(self.classifier_hidden_size, num_labels, device=self.device)
        
        new_classifier_only_bert = nn.Linear(self.classifier_hidden_size, new_output_size, device=self.device)
        
        if self.num_tasks > 0:
            with torch.no_grad():
                # Copy old weights to the new classifier
                
                new_classifier.weight[:self.num_old_labels, :] = self.classifier[self.num_tasks-1].weight
                new_classifier.bias[:self.num_old_labels] = self.classifier[self.num_tasks-1].bias
                
                new_classifier_only_bert.weight[:self.num_old_labels, :] = self.classifier_only_bert[self.num_tasks-1].weight
                new_classifier_only_bert.bias[:self.num_old_labels] = self.classifier_only_bert[self.num_tasks-1].bias
        
        self.classifier.append(new_classifier)
        
        self.classifier_only_bert.append(new_classifier_only_bert)

        self.feature_extractor.add_adapter(self.num_tasks)

        # calculate distribution for each class with all expert model
        # self.expert_distribution.append({
        #     "class_mean": [torch.zeros(self.class_per_task, self.query_size).to(self.device) for _ in
        #                    range(self.num_tasks)],
        #     "accumulate_cov": torch.zeros(self.query_size, self.query_size),
        #     "cov_inv": torch.ones(self.query_size, self.query_size),
        # })

    def save_classifier(self, idx, save_dir):
        state_dict = self.classifier[idx].state_dict()
        torch.save({
            f"classifier": state_dict
        }, f"{save_dir}/classifier-{idx}.pth")

    def load_classifier(self, idx, save_dir):
        ckpt = torch.load(f"{save_dir}/classifier-{idx}.pth")
        self.classifier[idx].load_state_dict(ckpt["classifier"])
        
    def save_classifier_only_bert(self, idx, save_dir):
        state_dict = self.classifier_only_bert[idx].state_dict()
        torch.save({
            f"classifier_only_bert": state_dict
        }, f"{save_dir}/classifier_only_bert-{idx}.pth")

    def load_classifier_only_bert(self, idx, save_dir):
        ckpt = torch.load(f"{save_dir}/classifier_only_bert-{idx}.pth")
        self.classifier_only_bert[idx].load_state_dict(ckpt["classifier_only_bert"])

    def new_statistic(self, mean, cov, task_mean, task_cov, expert_id=0):
        expert_id = self.shift_expert_id(expert_id)
        if expert_id == 0 or expert_id == 1:
            length = self.num_tasks + 1
        else:
            length = self.num_tasks - expert_id + 2
        # self.expert_distribution[expert_id]["class_mean"].append(mean.cuda())
        # self.expert_distribution[expert_id]["accumulate_cov"] += cov
        # avg_cov = self.expert_distribution[expert_id]["accumulate_cov"].cuda() / length
        # self.expert_distribution[expert_id]["cov_inv"] = torch.linalg.pinv(avg_cov, hermitian=True)
        
        self.expert_distribution["class_mean"].extend(mean.cuda())
        # self.expert_distribution["accumulate_cov"] = cov
        avg_cov = self.expert_distribution["accumulate_cov"].cuda()
        self.expert_distribution["cov_inv"] = torch.linalg.pinv(avg_cov, hermitian=True)
        
    def shift_expert_id(self, expert_id):
        return expert_id + 1

    def get_prompt_indices(self, prelogits, expert_id=0):
        expert_id = self.shift_expert_id(expert_id)
        task_means_over_classes = self.expert_distribution[expert_id]["class_mean"]
        cov_inv = self.expert_distribution[expert_id]["cov_inv"]

        scores_over_tasks = []
        class_indices_over_tasks = []
        # for each task
        for idx, mean_over_classes in enumerate(task_means_over_classes):
            num_labels, _ = mean_over_classes.shape
            score_over_classes = []
            # for each label in task
            for c in range(num_labels):
                if self.query_mode == "cosine":
                    score = - F.cosine_similarity(prelogits, mean_over_classes[c])
                elif self.query_mode == "euclidean":
                    score = torch.cdist(prelogits, mean_over_classes[c].unsqueeze(0)).squeeze(1)
                elif self.query_mode == "mahalanobis":
                    score = mahalanobis(prelogits, mean_over_classes[c], cov_inv, norm=2)
                elif self.query_mode == "maha_ft":
                    score = mahalanobis(prelogits[idx], mean_over_classes[c], cov_inv, norm=2)
                else:
                    raise NotImplementedError
                score_over_classes.append(score)
            # [num_labels, n]
            score_over_classes = torch.stack(score_over_classes)
            score, class_indices = score_over_classes.min(dim=0)
            # min score of labels as task score
            scores_over_tasks.append(score)
            class_indices_over_tasks.append(class_indices + idx * num_labels)
        # [task_num, n]
        scores_over_tasks = torch.stack(scores_over_tasks, dim=0)
        class_indices_over_tasks = torch.stack(class_indices_over_tasks, dim=0)
        _, indices = torch.min(scores_over_tasks, dim=0)

        return indices, scores_over_tasks, class_indices_over_tasks

    def forward(self, input_ids, attention_mask=None, labels=None, oracle=False, **kwargs):

        batch_size, _ = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids != 0

        if self.training:
            indices = torch.LongTensor([self.num_tasks] * batch_size).to(self.device)
        else:
            if "return_hidden_states" in kwargs and kwargs["return_hidden_states"]:
                # input task idx 0-9 -1:bert
                hidden_states = self.feature_extractor(
                    input_ids=input_ids,
                    attention_mask=(input_ids!=0),
                    indices=None,
                    **kwargs
                )
                if "extract_mode" in kwargs:
                    del kwargs["extract_mode"]
                return hidden_states
            
            hidden_states = self.feature_extractor(
                input_ids=input_ids,
                attention_mask=attention_mask,
                indices=None,
                **kwargs
            )

            logits = self.classifier_only_bert[-1](hidden_states)
            # print("--------classifier_only_bert-----")
            # print(logits)
            pred = logits.argmax(dim=1)
            indices = pred.to(self.device)
            # print("Ground truth Task")
            # print(labels)
            # print("Predict Task Task")
            # print(indices)
            
            
            if oracle:
                task_idx = kwargs["task_idx"]
                indices_task_id = torch.LongTensor([task_idx] * batch_size).to(self.device)
            else:
                indices_task_id = indices // 8
                indices_task_id = torch.tensor(indices_task_id, dtype=torch.long, device=self.device)
                
            # print("Predict Task indices")
            # print(indices_task_id) 
                
            hidden_states_final = self.feature_extractor(
                input_ids=input_ids,
                attention_mask=attention_mask,
                indices=indices_task_id,
                **kwargs
            )
            
            logits_list = []
            for i in range(batch_size):
                if oracle:
                    idx = kwargs["task_idx"]
                else:
                    idx = -1
                classifier = self.classifier[idx]
                hs = hidden_states_final[i].unsqueeze(0)
                logits = classifier(hs)
                print(logits)
                logits_list.append(logits)
            # print("-------------Classifier MLP 1--------------------")
            # print(logits)
            # Kết hợp logits từ tất cả các mẫu
            logits_final = torch.cat(logits_list, dim=0)

            # Lấy dự đoán cuối cùng
            preds = logits_final.argmax(dim=-1)
            # print("Grouth Truth Class")
            # print(labels)
            # print("Predict Label")
            # print(preds)
            
            preds = preds.cpu().numpy()
            indices_task_id = indices_task_id.cpu().numpy()
            # print("-----------------------------------")
            # print(preds)
            # print(indices)
            
            # Chuyển đổi indices thành 
            return ExpertOutput(
                preds=preds,
                indices=indices_task_id
            )

        if "mlp2" in kwargs and kwargs["mlp2"]:
            hidden_states = input_ids
            # print("train--------------MLP2")
            # print(self.num_tasks)
            # print(len(self.classifier_only_bert))
            logits = self.classifier_only_bert[self.num_tasks](hidden_states)
            # print("-------------Training Classifier MLP 2--------------------")
            # print(logits)
            if self.training:
                offset_label = labels.to(dtype=torch.long)
                loss = F.cross_entropy(logits, offset_label)
            
            logits = logits[:, :]
            preds = logits.max(dim=-1)[1]
            
            
            loggerdb.log_metrics({"train/mlp2_{self.num_tasks}": loss.item()})
            
            indices = indices.tolist() if isinstance(indices, torch.Tensor) else indices
            return ExpertOutput(
                loss=loss,
                preds=preds,
                hidden_states=hidden_states,
                indices=indices,
            )
        
        # only for training
        
        loss = None
        
        if self.training:
            if "mlp1_term2" in kwargs:
                            
                anchor_hidden_states = self.feature_extractor(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    indices=indices,
                    extract_mode="cls",
                    **kwargs
                )
                
                logits = self.classifier[self.num_tasks](anchor_hidden_states)
                
                offset_label = labels
                loss = F.cross_entropy(logits, offset_label)
                preds = logits.max(dim=-1)[1]
                
                numerator_list = []
                for class_mean in self.expert_distribution["class_mean"]:
                    numerator_list.append(torch.exp(torch.matmul(anchor_hidden_states, class_mean.unsqueeze(1)) / self.tau))
                # numerator = torch.sum(torch.stack(numerator_list))
                
                # Compute denominator: sum(exp(h · h' / τ)) + sum(exp(h · μ_c / τ))
                denominator_list = []
                
                stack_u_c = []
                for label in labels:
                    u_c = self.expert_distribution["class_mean"][label]
                    stack_u_c.append(u_c)
                stack_u_c = torch.stack(stack_u_c)
                
                denominator_list.append(torch.exp((anchor_hidden_states * stack_u_c).sum(dim=1, keepdim=True) / self.tau))
                denominator_list.extend(numerator_list)  # Add numerator terms for μ_c
                denominator = torch.sum(torch.stack(denominator_list))

                # Compute log term
                log_term = torch.zeros(batch_size, 1, device=self.device)
                for numerator in numerator_list:
                    log_term += torch.log(numerator / denominator)
                    
                loss += (log_term.mean() / self.num_labels).item()
                # print('------@@@@Term2-------')
                # print(loss)
                
                loggerdb.log_metrics({"train/loss_mlp1_term2_{self.num_tasks}": loss.item()})
                
                indices = indices.tolist() if isinstance(indices, torch.Tensor) else indices
                return ExpertOutput(
                    loss=loss,
                    hidden_states=anchor_hidden_states,
                    preds=preds,
                    indices=indices,
                )
        
        hidden_states = self.feature_extractor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            indices=indices,
            **kwargs
        )
            
        logits = self.classifier[self.num_tasks](hidden_states)
        # print("-------------Training Classifier--------------------")
        # print(logits)
        if self.training:
            offset_label = labels
            loss = F.cross_entropy(logits, offset_label) 
            loggerdb.log_metrics({"train/loss_cross_entropy": loss.item()})
            anchor_hidden_states = hidden_states
            # print("1")
            description_ids_list = {k: v for k, v in kwargs.items() if k.startswith('description_ids_')}
            total_log_term = torch.zeros(1, device=self.device)
            for k, v in description_ids_list.items():
                # print("2")
                description_hidden_states = self.feature_extractor(
                    input_ids=v,
                    attention_mask=(v != 0),
                    indices=indices,
                    extract_mode="cls",
                    **kwargs
                )
                # print("description_hidden_states")
                # print(description_hidden_states)
                # info_nce_loss_value = self.info_nce_loss(anchor_hidden_states, description_hidden_states)
                
                # contrastive regularization Loss
                # print("3")
                # Compute numerator: exp(h · μ_c / τ)
                numerator_list = []
                for class_mean in self.expert_distribution["class_mean"]:
                    # print(class_mean)
                    # print(anchor_hidden_states)
                    # print(class_mean.shape)
                    # print(anchor_hidden_states.shape)
                    numerator_list.append(torch.exp(torch.matmul(anchor_hidden_states, class_mean.unsqueeze(1)) / self.tau))
                # numerator = torch.sum(torch.stack(numerator_list))
                # print("4")
                # Compute denominator: sum(exp(h · h' / τ)) + sum(exp(h · μ_c / τ))
                denominator_list = []                
                
                denominator_list.append(torch.exp((anchor_hidden_states * description_hidden_states).sum(dim=1, keepdim=True) / self.tau))
                denominator_list.extend(numerator_list)  # Add numerator terms for μ_c
                denominator = torch.sum(torch.stack(denominator_list))
                # print("5")
                # Compute log term
                log_term = torch.zeros(batch_size, 1, device=self.device)
                for numerator in numerator_list:
                    log_term += torch.log(numerator / denominator)
                    # print("------------------")
                    # print(numerator)
                    # print(denominator)
                # loss += log_term.mean()
                # print("6")
                print(self.num_labels)
                total_log_term += (log_term.mean() / self.num_labels)
            # print("7")
            # print("----@@@@@@@@-------")
            # print(total_log_term / len(description_ids_list))
            loss += (total_log_term / len(description_ids_list)).item()
        # logger.log_metrics({"train/loss": loss})
        loggerdb.log_metrics({"train/cr_loss": (total_log_term / len(description_ids_list)).item()})
        loggerdb.log_metrics({"train/total_loss": loss.item()})
        preds = logits.max(dim=-1)[1]
                
        indices = indices.tolist() if isinstance(indices, torch.Tensor) else indices
        return ExpertOutput(
            loss=loss,
            preds=preds,
            hidden_states=hidden_states,
            indices=indices,
        )


@dataclass
class ExpertOutput:
    loss: Optional[torch.FloatTensor] = None
    preds: Optional[torch.LongTensor] = None
    logits: Optional[torch.FloatTensor] = None
    expert_task_preds: Optional[torch.LongTensor] = None
    expert_class_preds: Optional[torch.LongTensor] = None
    indices: Optional[torch.LongTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
