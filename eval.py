import torch
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import config
import utils
from data_loader import create_dataloaders
from model import BrainTumorCNN

def evaluate(args):
    device = utils.get_device()
    print(f"Using device: {device}")
    _, _, test_loader, class_names = create_dataloaders()
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("Model file not found. Please train the model first.")
        return
    model = BrainTumorCNN(num_classes=len(class_names)).to(device)
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_preds = []
    all_labels = []
    print("Evaluating on Test Set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(config.OUTPUTS_DIR, 'classification_report.csv'))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(config.OUTPUTS_DIR, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(config.OUTPUTS_DIR, 'confusion_matrix.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor CNN")
    args = parser.parse_args()
    evaluate(args)
