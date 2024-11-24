import copy
import math
import json
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple

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
        self.classifier = nn.ParameterList()
        self.classifier_only_bert = nn.ParameterList()
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    def generate_description(self, label, dataset_name, tokenizer):
        if dataset_name.lower() == 'fewrel':
            file_path = 'datasets/FewRel/pid2name.json'
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
        
        label_name = data[label][0]
        model_name = "gpt2"  # Bạn có thể thay thế bằng một mô hình ngôn ngữ mã nguồn mở khác
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
        prompt = f"Describe the label '{label_name}' in a simple and detailed way: "
        descriptions = generator(prompt, max_length=50, num_return_sequences=1)
        
        # Lưu mô tả nhãn vào label_description
        self.label_description_ids[label] = [self.preprocess_desciption(desc['generated_text'].replace(prompt, '').strip(), tokenizer) for desc in descriptions]
        self.label_description[label] = [desc['generated_text'].replace(prompt, '').strip() for desc in descriptions]

    def generate_description_from_file(self, label, dataset_name, tokenizer):
        if dataset_name.lower() == 'fewrel':
            file_path = 'datasets/FewRel/pid2name.json'
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                        
        # Lưu mô tả nhãn vào label_description
        # self.label_description_ids[label] = [self.preprocess_desciption(desc, tokenizer) for desc in data[label]]
        # self.label_description[label] = [desc for desc in data[label]]
        
        self.label_description_ids[label] = [self.preprocess_desciption(data[label][-1], tokenizer)]
        self.label_description[label] = [data[label][-1]]
        
    def preprocess_desciption(self, raw_text, tokenizer):
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
            pool[label] = copy.deepcopy(self.label_description_ids[label])
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
        print("-------------------------------")
        print(self.num_tasks)
        # if self.num_tasks > 0:
        #     with torch.no_grad():
        #         # Copy old weights to the new classifier
        #         new_classifier.weight[:self.num_old_labels, :] = self.classifier.weight
        #         new_classifier.bias[:self.num_old_labels] = self.classifier.bias
                
        #         new_classifier_only_bert.weight[:self.num_old_labels, :] = self.classifier_only_bert.weight
        #         new_classifier_only_bert.bias[:self.num_old_labels] = self.classifier_only_bert.bias
        
        self.classifier.append(new_classifier)
        
        self.classifier_only_bert.append(new_classifier_only_bert)

        self.feature_extractor.add_adapter(self.num_tasks)

        # calculate distribution for each class with all expert model
        self.expert_distribution.append({
            "class_mean": [torch.zeros(self.class_per_task, self.query_size).to(self.device) for _ in
                           range(self.num_tasks)],
            "accumulate_cov": torch.zeros(self.query_size, self.query_size),
            "cov_inv": torch.ones(self.query_size, self.query_size),
        })

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
        self.classifier_only_bert[idx].load_state_dict(ckpt["classifier"])

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
        
        self.expert_distribution["class_mean"].append(mean.cuda())
        self.expert_distribution[expert_id]["accumulate_cov"] += cov
        avg_cov = self.expert_distribution[expert_id]["accumulate_cov"].cuda() / length
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

    def info_nce_loss(self, anchor, positive, negative, temperature=0.07):
        # anchor: [batch_size, dim]
        # positive: [batch_size, dim]
        # negatives: [batch_size, num_negatives, dim]

        # Normalize embeddings
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)

        # Positive logits
        pos_logits = torch.sum(anchor * positive, dim=1, keepdim=True) / temperature  # [batch_size, 1]

        # Negative logits
        neg_logits = torch.sum(anchor * negative, dim=1, keepdim=True) / temperature  # [batch_size, 1]

        # Combine logits
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [batch_size, 1 + num_negatives]

        labels = torch.zeros(anchor.size(0), dtype=torch.long).to(anchor.device)

        # Compute loss
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(self, input_ids, attention_mask=None, labels=None, oracle=False, **kwargs):

        batch_size, _ = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids != 0

        if self.training:
            indices = torch.LongTensor([self.num_tasks] * batch_size).to(self.device)
        else:
            if "return_hidden_states" in kwargs and kwargs["return_hidden_states"]:
                # input task idx 0-9 -1:bert
                if kwargs["task_idx"] == -1:  # origin bert
                    indices = None
                    use_origin = True
                    kwargs.update({"extract_mode": "entity"})
                elif kwargs["task_idx"] == 0:  # first task model
                    indices = None
                    use_origin = False
                else:
                    indices = [kwargs["task_idx"]] * batch_size
                    use_origin = False
                hidden_states = self.feature_extractor(
                    input_ids=input_ids if kwargs["task_idx"] != -1 else kwargs["input_ids_without_marker"],
                    indices=indices,
                    use_origin=use_origin,
                    **kwargs
                )
                if "extract_mode" in kwargs:
                    del kwargs["extract_mode"]
                return hidden_states

            all_score_over_task = []
            all_score_over_class = []
            all_logits = []
            for e_id in range(-1, self.num_tasks + 1):
                if e_id == -1:
                    indices = None
                    use_origin = True
                    kwargs.update({"extract_mode": "entity"})
                elif e_id == 0:
                    indices = None
                    use_origin = False
                else:
                    indices = [e_id] * batch_size
                    use_origin = False
                hidden_states = self.feature_extractor(
                    input_ids=input_ids if e_id != -1 else kwargs["input_ids_without_marker"],
                    indices=indices,
                    use_origin=use_origin,
                    **kwargs
                )
                if "extract_mode" in kwargs:
                    del kwargs["extract_mode"]
                _, scores_over_tasks, scores_over_classes = self.get_prompt_indices(hidden_states, expert_id=e_id)
                scores_over_tasks = scores_over_tasks.transpose(-1, -2)
                scores_over_classes = scores_over_classes.transpose(-1, -2)
                if e_id != -1:
                    scores_over_tasks[:, :e_id] = float('inf')  # no seen task
                    logits = self.classifier[e_id](hidden_states)[:, :self.class_per_task]
                    all_logits.append(logits)
                all_score_over_task.append(scores_over_tasks)
                all_score_over_class.append(scores_over_classes)
            all_score_over_task = torch.stack(all_score_over_task, dim=1)  # (batch, expert_num, task_num)
            all_score_over_class = torch.stack(all_score_over_class, dim=1)  # (batch, expert_num, task_num)
            all_logits = torch.stack(all_logits, dim=1)
            indices = []
            # expert0_score_over_task = all_score_over_task[:, 0, :]  # (batch, task_num)
            expert_values, expert_indices = torch.topk(all_score_over_task, dim=-1, k=all_score_over_task.shape[-1],
                                                       largest=False)
            expert_values = expert_values.tolist()
            expert_indices = expert_indices.tolist()
            for i in range(batch_size):
                bert_indices = expert_indices[i][0]
                task_indices = expert_indices[i][1]
                if self.default_expert == "bert":
                    default_indices = copy.deepcopy(bert_indices)
                else:
                    default_indices = copy.deepcopy(task_indices)
                min_task = min(bert_indices[0], task_indices[0])
                max_task = max(bert_indices[0], task_indices[0])
                # valid_task_id = [min_task, max_task]
                cur_min_expert = self.shift_expert_id(min_task)
                if bert_indices[0] != task_indices[0] and cur_min_expert > 1:
                    cur_ans = []
                    for j in range(0, cur_min_expert + 1):
                        if j <= self.max_expert:  # self.max_expert==1 --> default expert
                            for k in expert_indices[i][j]:
                                if k >= min_task:
                                    cur_ans.append(k)
                                    break
                    cur_count = Counter(cur_ans)
                    most_common_element = cur_count.most_common(1)
                    if most_common_element[0][1] == cur_ans.count(default_indices[0]):
                        indices.append(default_indices[0])
                    else:
                        indices.append(most_common_element[0][0])
                else:
                    indices.append(default_indices[0])
                # indices.append(expert_indices[i][1][0])
            indices = torch.LongTensor(indices).to(self.device)
            if oracle:
                task_idx = kwargs["task_idx"]
                indices = torch.LongTensor([task_idx] * batch_size).to(self.device)
            idx = torch.arange(batch_size).to(self.device)
            all_logits = all_logits[idx, indices]
            preds = all_logits.max(dim=-1)[1] + self.class_per_task * indices
            indices = indices.tolist() if isinstance(indices, torch.Tensor) else indices
            return ExpertOutput(
                preds=preds,
                indices=indices,
                expert_task_preds=all_score_over_task,
                expert_class_preds=all_score_over_class,
            )
            
        # only for training
        hidden_states = self.feature_extractor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            indices=indices,
            attribute="anchor",
            **kwargs
        )
        logits = self.classifier[self.num_tasks](hidden_states)

        loss = None
        if self.training:
            offset_label = labels
            loss = F.cross_entropy(logits, offset_label)
            
            anchor_hidden_states = hidden_states
            
            # print("First------------------")
            # print(input_ids.size())
            # print(positive_input_ids.size())
            # print(negative_input_ids.size())
            description_ids_list = {k: v for k, v in kwargs.items() if k.startswith('description_ids_')}

            for k, v in description_ids_list.items():
                description_hidden_states = self.feature_extractor(
                    input_ids=v,
                    attention_mask=(v != 0),
                    indices=indices,
                    extract_mode="cls",
                    **kwargs
                )
                info_nce_loss_value = self.info_nce_loss(anchor_hidden_states, description_hidden_states)
                
                # contrastive regularization Loss
                
                # Compute numerator: exp(h · μ_c / τ)
                numerator_list = []
                for expert in self.expert_distribution:
                    class_mean = expert["class_mean"]
                    print(class_mean)
                    print(anchor_hidden_states)
                    print(class_mean.shape)
                    print(anchor_hidden_states.shape)
                    numerator_list.append(torch.exp(torch.dot(anchor_hidden_states, class_mean) / self.tau))
                # numerator = torch.sum(torch.stack(numerator_list))
                
                # Compute denominator: sum(exp(h · h' / τ)) + sum(exp(h · μ_c / τ))
                denominator_list = []
                denominator_list.append(torch.exp(torch.dot(anchor_hidden_states, description_hidden_states) / self.tau))
                denominator_list.extend(numerator_list)  # Add numerator terms for μ_c
                denominator = torch.sum(torch.stack(denominator_list))

                # Compute log term
                for numerator in numerator_list:
                    log_term = torch.log(numerator / denominator)
                    loss += log_term
                    
                loss += info_nce_loss_value            

        # logits = logits[:, :self.class_per_task]
        # preds = logits.max(dim=-1)[1] + self.class_per_task * indices
        
        logits = logits[:, :]
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
