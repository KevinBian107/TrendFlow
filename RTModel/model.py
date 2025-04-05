import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from peft import LoraConfig, get_peft_model

class ScenarioModel(nn.Module):
    def __init__(self, args, tokenizer, target_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_setup(args)
        self.target_size = target_size

        # task1: add necessary class variables as you wish.
        self.encoder_out_features = 768
        self.projection = nn.Linear(self.encoder_out_features,args.embed_dim)
        
        # task2: initilize the dropout and classify layers done
        self.dropout = nn.Dropout(args.drop_rate)
        self.classify = Classifier(args,target_size)
    
    def model_setup(self, args):
        print(f"Setting up {args.model} model")
        # task1: get a pretrained model of 'bert-base-uncased' done
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

    def forward(self, inputs, targets):
        """
        1. feeding the input to the encoder, 
        2. take the last_hidden_state's <CLS> token as output of the encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
        3. feed the output of the dropout layer to the Classifier which is provided for you.
        """
        # encoder_outputs shape: ((batch_size, sequence_length, hidden_size),(batch_size, hidden_size))
        encoder_outputs = self.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        output = encoder_outputs.last_hidden_state[:, 0, :]
        output = self.dropout(output)
        logits = self.classify(output)
        return logits
  
class Classifier(nn.Module):
    def __init__(self, args, target_size):
        super().__init__()
        input_dim = args.embed_dim
        self.top = nn.Linear(input_dim, args.hidden_dim)
        self.relu = nn.ReLU()
        self.bottom = nn.Linear(args.hidden_dim, target_size)

    def forward(self, hidden):
        middle = self.relu(self.top(hidden))
        logit = self.bottom(middle)
        return logit


class CustomModel(ScenarioModel):                                             
    def __init__(self, args, tokenizer, target_size):
        super().__init__(args, tokenizer, target_size)

        # Layer-wise Learning Rate Decay (LLRD)
        self.lr_multiplier = getattr(args, "lr_multiplier", 0.95)  # Decay factor for deeper layers

    def reinitialize_layers(self, n=0):
        """
        Re-initialize the last `n` layers of BERT to help adaptation.
        """
        # Jack's implementation
        for layer in self.encoder.encoder.layer[-n:]:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=self.encoder.config.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)

        print('Check top layer weights:', self.encoder.encoder.layer[-1].output.dense.weight)

    def setup_optimizer(self, args):
        """
        LLRD and configures optimizer with different learning rates for each layer.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []
        lr = args.learning_rate

        for i, layer in enumerate(self.encoder.encoder.layer):
            layer_lr = lr * (self.lr_multiplier ** (len(self.encoder.encoder.layer) - 1 - i))
            optimizer_grouped_parameters.extend([
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                    "lr": layer_lr
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": layer_lr
                }
            ])

        optimizer_grouped_parameters.extend([
            {"params": [p for n, p in self.classify.named_parameters()], "lr": lr}
        ])

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=args.weight_decay)
        return optimizer

    def setup_scheduler(self, optimizer, train_dataloader, args):
        """
        Implements Warm-up steps and learning rate scheduling.
        """
        num_training_steps = len(train_dataloader) * args.n_epochs
        num_warmup_steps = len(train_dataloader) * 1  # Warm-up for 2 epochs

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=num_warmup_steps, 
                                                    num_training_steps=num_training_steps)
        return scheduler


    def setup_swa_model(self, device):
        """
        Sets up Stochastic Weight Averaging model wrapper.
        """
        swa_model = torch.optim.swa_utils.AveragedModel(self)
        swa_model.to(device)
        return swa_model

    def setup_swa_scheduler(self, args, optimizer):
        """
        Sets up the SWA learning rate scheduler.
        """
        # First, update all parameter groups to use the SWA learning rate
        for group in optimizer.param_groups:
            group['initial_lr'] = args.swa_lr

        # Create the SWA scheduler with the specified parameters
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer,
            swa_lr=args.swa_lr,
            anneal_strategy=args.swa_anneal_strategy,
            anneal_epochs=args.swa_anneal_epochs
        )
        return swa_scheduler


class LoRAModel(ScenarioModel):                                             
    def __init__(self, args, tokenizer, target_size, r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__(args, tokenizer, target_size)

        self.config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self._set_lora()

    def _set_lora(self):
        print('LoRA fine-tunning!')
        self.encoder = get_peft_model(self.encoder, self.config)
        self.encoder.print_trainable_parameters()