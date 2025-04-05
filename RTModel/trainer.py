import os, sys, pdb
import numpy as np
import random
import torch
import math
from tqdm import tqdm as progress_bar
from utils import set_seed, setup_gpus, check_directories
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from model import ScenarioModel, CustomModel
from torch import nn
import wandb
from uuid import uuid4
import json
from pathlib import Path


def baseline_train(args, model, datasets, tokenizer, logger, resume_id = False):
    if not resume_id:
        [f.unlink() for f in Path("checkpoints/").glob("*") if f.is_file()] 
        train_id = str(uuid4())
    else:
        train_id = resume_id
    logger.info(f"run_id:{train_id}")
    checkpoint_path = f"checkpoints/{train_id}.pth"
    
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args,datasets,split="train")
    criterion = nn.CrossEntropyLoss()

    # task2: setup model's optimizer_scheduler if you have
    optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    
    best_avg_loss = float("inf")
    if resume_id:
        wandb.init(project="TrendFlow-RTModel", config=vars(args), id = resume_id, resume="must")
    else:
        wandb.init(project="TrendFlow-RTModel", config=vars(args),id = train_id)
    # task3: write a training loop
    start_epoch = 0
    #if there are wandb can be restore
    if resume_id:
        logger.info(f"Resume from the previous checkpoint: {train_id}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  
        logger.info(f"Resuming training from epoch {start_epoch}")
    
    for epoch_count in range(start_epoch,args.n_epochs):
        losses = 0
        model.train()
        acc = 0
        #FIXME
        pbar = progress_bar(enumerate(train_dataloader),total = len(train_dataloader))
        #for step, batch in progress_bar(train_dataloader)
        for idx, batch in pbar:
            inputs, labels = prepare_inputs(batch = batch, use_text = False)
            logits = model(inputs,labels)
            loss = criterion(logits,labels)
            pbar.set_description(f"train_loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()  # backprop to update the weights
            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()
            model.zero_grad()
            losses += loss.item()
        scheduler.step()
        acc = acc/len(datasets['train'])
        avg_train_loss = losses/len(train_dataloader)
        logger.info(f'epoch {epoch_count} | avg_train_loss: {losses/len(train_dataloader)}')
        val_acc, avg_val_loss = run_eval(args, model, datasets, criterion, tokenizer, logger, split='validation')
        checkpoint = {
        'epoch': epoch_count,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        }
        wandb.log({
                "Train Accuracy": acc,
                "Validation Accuracy":val_acc,
                "Train Loss": avg_train_loss,
                "Validation Loss": avg_val_loss,
                "Epoch": epoch_count + 1,
                "Learning Rate": scheduler.get_last_lr()[0],
                "Best Validation Loss": best_avg_loss
            })
        torch.save(checkpoint,checkpoint_path)
        #wandb.save(checkpoint_path)
        logger.info(f"Epoch {epoch_count} completed, checkpoint saved.")
    wandb.finish()
    test_acc, test_loss = run_eval(args, model, datasets, criterion, tokenizer, logger, split='test')
    save_model(args,model,test_acc,test_loss,logger)


def custom_train(args, model, datasets, tokenizer, logger, resume_id=False):
    """
    Custom training loop with early stopping.
    """

    if not resume_id:
        checkpoint_dir = Path("checkpoints/")
        if checkpoint_dir.exists():
            [f.unlink() for f in checkpoint_dir.glob("*") if f.is_file()]
        train_id = str(uuid4())
    else:
        train_id = resume_id

    logger.info(f"Custom training run_id: {train_id}")
    checkpoint_path = Path("checkpoints") / f"custom_{train_id}.pth"
    logger.info(f"Checkpoint will be saved to: {checkpoint_path}")

    if resume_id:
        wandb.init(project="TrendFlow-RTModel", config=vars(args), id=resume_id, resume="must")
    else:
        wandb.init(project="TrendFlow-RTModel", config=vars(args), id=train_id)

    criterion = torch.nn.CrossEntropyLoss()
    train_dataloader = get_dataloader(args, datasets, split="train")

    if getattr(args, 'use_reinit', False):
        n_layers = getattr(args, 'n_reinit_layers', 2)
        logger.info(f"Re-initializing top {n_layers} layers of BERT.")
        model.reinitialize_layers(n_layers)
    else:
        logger.info("No layer re-initialization is applied.")

    if getattr(args, 'use_llrd', False):
        logger.info("Using layer-wise learning rate decay (LLRD).")
        optimizer = model.setup_optimizer(args)
    else:
        logger.info("Using a standard AdamW optimizer (no LLRD).")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if getattr(args, 'use_warmup', False):
        logger.info("Using linear schedule with warmup.")
        scheduler = model.setup_scheduler(optimizer, train_dataloader, args)
    else:
        logger.info("Using a default MultiStepLR or no scheduler at all.")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[30, 80], 
            gamma=0.1
        )


    start_epoch = 0
    if resume_id and checkpoint_path.exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming training from epoch {start_epoch}")
    else:
        logger.info("No valid checkpoint found, starting training from scratch.")


    best_avg_loss = float('inf')
    no_improvement_count = 0
    patience = getattr(args, "early_stop_patience", 3)

    for epoch in range(start_epoch, args.n_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        pbar = progress_bar(enumerate(train_dataloader),total = len(train_dataloader))
        for idx, batch in pbar:
            inputs, labels = prepare_inputs(batch, use_text=False)

            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            
            pbar.set_description(f"train_loss: {loss.item():.4f}")
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

        # Scheduler step (once per epoch)
        scheduler.step()

        avg_train_loss = epoch_loss / len(train_dataloader)
        train_acc = correct / len(datasets['train'])

        logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_acc, avg_val_loss = run_eval(args, model, datasets, criterion, tokenizer, logger, split='validation')

        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.learning_rate
        wandb.log({
            'Epoch': epoch + 1,
            'Train Loss': avg_train_loss,
            'Train Accuracy': train_acc,
            'Validation Loss': avg_val_loss,
            'Validation Accuracy': val_acc,
            'Learning Rate': current_lr
        })

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Epoch {epoch} completed, checkpoint saved to {checkpoint_path}")

        if avg_val_loss < best_avg_loss:
            best_avg_loss = avg_val_loss

            best_checkpoint_path = Path("checkpoints") / f"custom_{train_id}_best.pth"
            torch.save(checkpoint_data, best_checkpoint_path)
            logger.info(f"New best model saved to {best_checkpoint_path}")
            no_improvement_count=0
        else:
            no_improvement_count += 1
            logger.info(f"No improvement in validation loss for {no_improvement_count} epochs.")
            if no_improvement_count >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs with no improvement.")
                break

    wandb.finish()


def save_model(args,model,test_acc,test_loss,logger):
    logger.info("Saving Models")

    if not os.path.exists(f'{args.output_dir}/{args.task}/best_model.json'): #if it is the first model, 
        logger.info(f"Model model saved to {args.output_dir}/{args.task}/best_model.pt")
        torch.save(model,f'{args.output_dir}/{args.task}/best_model.pt')
        with open(f'{args.output_dir}/{args.task}/best_model.json',"w") as f:
            f.write(json.dumps({**vars(args),'test_acc':test_acc,'test_loss':test_loss}))

    else: #if it is not the first model, compare to find the best one
        with open(f'{args.output_dir}/{args.task}/best_model.json','r+') as f:
            historical_best_model_dict = json.load(f)

        if historical_best_model_dict["test_acc"] < test_acc:
            logger.info(f"New best accuracy achieved: {test_acc} (previous best: {historical_best_model_dict['test_acc']})")
            logger.info(f"Best model saved to {args.output_dir}/{args.task}/best_model.pt")
            torch.save(model,f'{args.output_dir}/{args.task}/best_model.pt')
            with open(f'{args.output_dir}/{args.task}/best_model.json','w') as f:
                f.write(json.dumps({**vars(args),'test_acc':test_acc,'test_loss':test_loss}))
        else:
            logger.info(f"Current accuracy {test_acc} did not exceed best accuracy {historical_best_model_dict['test_acc']}")
    #save as newest model
    with open(f'{args.output_dir}/{args.task}/newest_model.json','w') as f:
        f.write(json.dumps({**vars(args),'test_acc':test_acc,'test_loss':test_loss}))
    torch.save(model,f'{args.output_dir}/{args.task}/newest_model.pt')
    logger.info(f"Result Saved into f'{args.output_dir}/{args.task}")


def run_eval(args, model, datasets, criterion, tokenizer, logger, split='validation'):
    '''Evaluation function for  all models'''
    
    model.eval()
    acc = 0
    losses = 0
    dataloader = get_dataloader(args, datasets, split)
    pbar = progress_bar(enumerate(dataloader), total=len(dataloader))
    for step, batch in pbar:
        inputs,labels = prepare_inputs(batch,use_text=False)
        logits = model(inputs, labels)

        loss = criterion(logits,labels)
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()

        losses += loss.item()
        pbar.set_description(f"{split}_loss: {loss.item():.4f}")

    avg_loss = losses/len(dataloader)
    
    if acc is not None:
        acc = acc/len(datasets[split])
        logger.info(f'{split} acc:{acc} | avg_{split}_loss:{avg_loss}')
    else:
        logger.info(f'{split} contrastive loss: {avg_loss}')
    
    return acc, avg_loss