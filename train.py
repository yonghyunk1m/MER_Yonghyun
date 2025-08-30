# train.py (수정된 버전)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
import importlib

from dataset import PMEmoDataset
from utils import CCCLoss

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        filter_bank = batch["filter_bank"].to(device)
        handcrafted = batch["handcrafted"].to(device)
        style = batch["style"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        predictions = model(filter_bank, handcrafted, style)
        
        loss_v = criterion(predictions[:, 0], labels[:, 0])
        loss_a = criterion(predictions[:, 1], labels[:, 1])
        loss = (loss_v + loss_a) / 2
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            filter_bank = batch["filter_bank"].to(device)
            handcrafted = batch["handcrafted"].to(device)
            style = batch["style"].to(device)
            labels = batch["labels"].to(device)
            
            predictions = model(filter_bank, handcrafted, style)
            loss_v = criterion(predictions[:, 0], labels[:, 0])
            loss_a = criterion(predictions[:, 1], labels[:, 1])
            loss = (loss_v + loss_a) / 2
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MusicEmotionDataset(args.annotations_file, args.audio_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Loading model: {args.model}")
    try:
        model_module = importlib.import_module(f"models.{args.model}.model")
        ModelClass = getattr(model_module, args.model)
        model = ModelClass(num_styles=8).to(device)
    except (ImportError, AttributeError) as e:
        print(f"Error loading model '{args.model}'. Make sure 'models/{args.model}/model.py' exists and contains a class named '{args.model}'.")
        print(e)
        return

    criterion = CCCLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    
    model_save_dir = os.path.join(args.save_dir, args.model)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{model_save_dir}/best_model.pth")
            print(f"Best model saved to {model_save_dir}/best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Music Emotion Recognition models")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to train (e.g., MCAN).")
    parser.add_argument("--annotations_file", type=str, required=True, help="Path to the annotations CSV file.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to the directory containing audio files.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for Adam optimizer.")
    args = parser.parse_args()
    main(args)
    
"""
python train.py \
    --model MCAN \
    --annotations_file ./data/DEAM/annotations.csv \
    --audio_dir ./data/DEAM/audio \
    --save_dir ./checkpoints \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001
"""