import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import pandas as pd
from tqdm import tqdm
import config
import utils
from data_loader import create_dataloaders
from model import BrainTumorCNN

def train(args):
    utils.seed_everything()
    device = utils.get_device()
    print(f"Using device: {device}")
    train_loader, val_loader, _, class_names = create_dataloaders()
    model = BrainTumorCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    start_epoch = 0
    if args.resume and os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Resuming from {config.MODEL_SAVE_PATH}")
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
       
    log_file = os.path.join(config.LOGS_DIR, 'training_log.csv')
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    best_val_acc = 0.0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f}\n")
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, config.MODEL_SAVE_PATH)
            print(f"Saved best model with Val Acc: {val_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brain Tumor CNN")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    config.BATCH_SIZE = args.batch_size
    train(args)
