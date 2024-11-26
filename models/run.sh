#!/bin/bash

# Install necessary packages
pip install hydra-core

# Clone the repository
git clone https://github.com/PiDinosauR2804/Eoe-Famework.git

# Navigate into the project directory
cd Eoe-Famework

# Install project dependencies
pip install -r requirements.txt
pip install peft
pip install -q -U google-generativeai

# Run the first training command
python main.py \
  +task_args='FewRel' \
  +training_args=Expert \
  task_args.model_name_or_path='bert-base-uncased' \
  task_args.config_name='configs/task_args/FewRel.yaml' \
  task_args.tokenizer_name='bert-base-uncased'

# Run the second training command
python main.py \
  +task_args='FewRel' \
  +training_args=EoE \
  task_args.model_name_or_path='bert-base-uncased' \
  task_args.config_name='configs/task_args/FewRel.yaml' \
  task_args.tokenizer_name='bert-base-uncased'
