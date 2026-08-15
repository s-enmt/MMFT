import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import gc

# Heavy deps (open_clip -> timm/transformers, torchvision via dataset_utils) are
# imported lazily inside main()/train_epoch so an already-completed run can exit
# via setup_experiment_dir() without paying that import cost.
from utils import experiment_utils


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CLIP Fine-tuning with Caption Data')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='Pet',
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, default='../dataset',
                       help='Data root directory')
    
    # Model settings
    parser.add_argument('--model_name', type=str, default='ViT-B-32',
                       help='CLIP model architecture')
    parser.add_argument('--pretrained', type=str, default='openai',
                       help='Pretrained weights')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=256,
                       help='Batch size for validation and test')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--min_lr_ratio', type=float, default=0.0,
                       help='Minimum learning rate ratio for cosine annealing')
    parser.add_argument('--no_amp', action='store_true',
                       help='Disable AMP and train in fp32 (no autocast, no GradScaler). '
                            'Removes loss-scaling artifacts entirely, at the cost of speed')
    parser.add_argument('--init_scale', type=float, default=65536.0,
                       help='Initial AMP GradScaler scale (torch default 65536.0). '
                            'Lower it so short few-shot runs do not lose their first '
                            'optimizer steps to overflow-induced step skipping')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory')
    
    # MMFT settings
    parser.add_argument('--captions', type=str, default='./captions/Pet_captions.json',
                       help='Image caption JSON file for training')
    parser.add_argument('--add_class_template', action='store_true',
                   help='Add "a photo of a {class_name}. " prefix to caption')
    parser.add_argument('--w', type=float, default=0.1,
                   help='Loss weight')
    parser.add_argument('--prompt_ensemble', action='store_true',
                   help='FLYP-style 80-prompt ensemble: random template at training, '
                        'averaged embeddings at inference. Only effective with --captions class_label')

    return parser.parse_args()


# Alternative simpler implementation based on SupCon-style
def sup_ft_loss(logits_per_image, logits_per_text, class_labels):
    device = logits_per_image.device
    
    # Create masks
    class_labels = class_labels.contiguous().view(-1, 1)
    same_class_mask = (class_labels == class_labels.T).float().to(device)
    
    # Image-to-Text loss
    loss_i2t = supcon_style_loss(logits_per_image, same_class_mask)
    
    # Text-to-Image loss
    loss_t2i = supcon_style_loss(logits_per_text, same_class_mask)
    
    return (loss_i2t + loss_t2i) / 2


def supcon_style_loss(logits, same_class_mask):
    """SupCon-style loss computation"""
    batch_size = logits.shape[0]
    
    # Remove diagonal (self-pairs)
    mask = same_class_mask - torch.eye(batch_size, device=logits.device)
    
    # Compute exp logits
    exp_logits = torch.exp(logits)
    
    # Compute log probabilities
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
    
    # Compute mean log probability over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    
    # Average over batch, only considering samples that have positives
    valid_samples = mask.sum(dim=1) > 0
    if valid_samples.sum() > 0:
        loss = -mean_log_prob_pos[valid_samples].mean()
    else:
        # Fallback to standard CLIP loss
        labels = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
    
    return loss


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch, args):
    """Train for one epoch with contrastive learning"""
    import open_clip
    from utils import model_utils
    model.train()
    total_loss = 0
    tokenizer = open_clip.get_tokenizer(args.model_name)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, captions, labels in pbar:
        images = images.to(device)
        
        # Tokenize captions
        texts = tokenizer(captions).to(device)
        
        optimizer.zero_grad()
        
        # Use autocast for forward pass
        with autocast(enabled=not args.no_amp):
            logits_per_image, logits_per_text = model(images, texts)

            sup_loss = sup_ft_loss(logits_per_image, logits_per_text, labels)
            clip_loss = model_utils.contrastive_loss(logits_per_image, logits_per_text)
            loss = args.w * sup_loss + (1 - args.w) * clip_loss
            
        # Scale the loss and backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update learning rate
        scheduler.step()
        
        total_loss += loss.item()
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'LR': f'{current_lr:.2e}'
        })
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, epoch=None, amp=True):
    """Validation using classification"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    desc = f'Epoch {epoch} [Val]' if epoch is not None else 'Test'
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Use autocast for forward pass
            with autocast(enabled=amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    return total_loss / len(dataloader), 100. * correct / total


def main():
    args = parse_args()
    print(args)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create experiment directory (exits here if this run is already completed,
    # before the heavy imports below are paid).
    exp_dir = experiment_utils.setup_experiment_dir(args)
    print(f"Experiment directory: {exp_dir}")

    # Heavy imports, only reached for runs that actually execute.
    import open_clip
    from utils import dataset_utils, model_utils

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    amp_enabled = device.type == 'cuda' and not args.no_amp
    if amp_enabled:
        print("AMP (Automatic Mixed Precision) enabled")
    elif args.no_amp:
        print("AMP disabled by --no_amp: training in fp32")
    else:
        print("Warning: CUDA not available, AMP will not be effective")

    torch.cuda.empty_cache()
    gc.collect()
    
    # Load CLIP model
    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, device=device
    )
    
    # Get datasets using dataset_utils
    train_dataset, val_dataset, test_dataset, num_classes, class_names = dataset_utils.get_dataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        val_split_ratio=args.val_split,
        seed=args.seed,
        train_transform=preprocess_train,
        val_transform=preprocess_val
    )
    
    # Load caption data and determine caption type
    caption_data = dataset_utils.load_caption_data(args.captions)
    caption_train_dataset = dataset_utils.AugMultiCaptionDataset(
        train_dataset,
        caption_data,
        class_names,
        add_class_template=args.add_class_template,
        prompt_ensemble=args.prompt_ensemble
    )
    filename_to_class = caption_train_dataset.filename_to_class

    print(f"Dataset: {args.dataset}")
    print(f"Training with caption data")
    print(f"Train samples: {len(caption_train_dataset)} (with captions)")
    print(f"Val samples: {len(val_dataset)} (with labels)")
    print(f"Test samples: {len(test_dataset)} (with labels)")
    print(f"Number of classes: {num_classes}")
    print(f"Add class template: {args.add_class_template}")
     
    # Create DataLoaders
    train_loader = DataLoader(caption_train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, 
                           shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, 
                            shuffle=False, num_workers=4)
    
    # Create models
    contrastive_model = model_utils.CLIPContrastiveModel(clip_model)
        
    # Optimizer and scheduler
    optimizer = optim.AdamW(contrastive_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs
    min_lr = args.lr * args.min_lr_ratio
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
    
    # Initialize GradScaler for AMP
    scaler = GradScaler(init_scale=args.init_scale, enabled=not args.no_amp)
    
    print(f"Total training steps: {total_steps}")
    print(f"Initial LR: {args.lr:.2e}, Min LR: {min_lr:.2e}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0
    best_epoch = 0
    best_model_path = exp_dir / 'best_model.pth'

    training_mode = f"contrastive_learning_with_captions"
    if args.add_class_template:
        training_mode += "_with_class_template"
        
    print(f"Starting contrastive learning for {args.epochs} epochs...")
    print("Training: Image-Caption contrastive learning")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Training with contrastive loss on caption data
        train_loss = train_epoch(
            contrastive_model, train_loader, optimizer, scheduler, scaler, device, epoch, args
        )
        
        # Validation with classification on label data
        if caption_data is None:  # FLYP / class_label
            if args.prompt_ensemble:
                eval_model = model_utils.CLIPClassifierEnsemble(
                    clip_model, args.model_name, class_names, device)
            else:
                eval_model = model_utils.CLIPClassifierOriginal(
                    clip_model, args.model_name, class_names, device)
        else:
            eval_model = model_utils.CLIPClassifier(clip_model, args.model_name, class_names, device,
                            caption_data=caption_data,
                            add_class_template=args.add_class_template,
                            filename_to_class=filename_to_class,
                            )
        val_loss, val_acc = validate(eval_model, val_loader, nn.CrossEntropyLoss(), device, epoch,
                                     amp=amp_enabled)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(scheduler.get_last_lr()[0])
        
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train Loss (Contrastive): {train_loss:.4f}")
        print(f"Val Loss (Classification): {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(clip_model.state_dict(), best_model_path)
            print(f"Best model saved with val acc: {best_val_acc:.2f}% at epoch {best_epoch}")
        
        print("-" * 60)
    
    # Load best model for final evaluation
    print("Loading best model for final evaluation...")
    clip_model.load_state_dict(torch.load(best_model_path))
    if caption_data is None:  # FLYP / class_label
        if args.prompt_ensemble:
            final_eval_model = model_utils.CLIPClassifierEnsemble(
                clip_model, args.model_name, class_names, device)
        else:
            final_eval_model = model_utils.CLIPClassifierOriginal(
                clip_model, args.model_name, class_names, device)
    else:
        final_eval_model = model_utils.CLIPClassifier(clip_model, args.model_name, class_names, device,
                            caption_data=caption_data,
                            add_class_template=args.add_class_template,
                            filename_to_class=filename_to_class,
                            )
    
    # Test evaluation
    test_loss, test_acc = validate(final_eval_model, test_loader, nn.CrossEntropyLoss(), device,
                                   amp=amp_enabled)
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Best model was saved at epoch {best_epoch} with validation accuracy: {best_val_acc:.2f}%")
    
    # Save results
    results_path = exp_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final training info
    final_info = {
        'dataset': args.dataset,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'final_test_acc': test_acc,
        'total_epochs': args.epochs,
        'total_steps': total_steps,
        'final_lr': scheduler.get_last_lr()[0],
        'min_lr': min_lr,
        'amp_enabled': amp_enabled,
        'training_mode': training_mode,
        'model_name': args.model_name,
        'pretrained': args.pretrained,
        'caption_samples': len(caption_train_dataset),
        'add_class_template': args.add_class_template,
    }
    
    info_path = exp_dir / 'training_info.json'
    with open(info_path, 'w') as f:
        json.dump(final_info, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    print(f"Training info saved to: {info_path}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == '__main__':
    main()