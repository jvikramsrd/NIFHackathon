import argparse
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as CFG
from data.dataset import split_dataset, split_clf_dataset
from models import Stage1Module, RooftopClassifier, InfrastructureDetector
from utils.core import setup, get_logger, atomic_torch_save
from utils.metrics import SegmentationMetrics

log = get_logger("train")

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = model.loss(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def val_epoch_seg(model, loader, device, num_classes, class_names):
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics(num_classes, class_names)
    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = model.loss(logits, labels)
        total_loss += loss.item()
        
        preds = logits.argmax(1)
        metrics.update(preds.cpu().numpy(), labels.cpu().numpy())
    
    return total_loss / len(loader), metrics.compute()

@torch.no_grad()
def val_epoch_cls(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images, labels)
        loss = model.loss(logits, labels)
        total_loss += loss.item()
        
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / len(loader), correct / total

def train_stage1():
    cfg = CFG.STAGE1
    device = setup(cfg["seed"])
    
    train_ds, val_ds = split_dataset(
        str(CFG.PATCH_DIR), str(CFG.MASK_DIR),
        cfg["val_fraction"], cfg["seed"],
        cfg["num_classes"], cfg["patch_size"], cfg.get("patch_sizes"),
    )
    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=CFG.NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"]//2, shuffle=False, num_workers=CFG.NUM_WORKERS)
    
    model = Stage1Module(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    
    best_iou = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        log.info(f"Epoch {epoch}/{cfg['epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, metrics = val_epoch_seg(model, val_loader, device, cfg["num_classes"], cfg["class_names"])
        
        miou = metrics["mean_iou"]
        log.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {miou:.4f}")
        
        if miou > best_iou:
            best_iou = miou
            atomic_torch_save(model.state_dict(), str(CFG.CKPT_DIR / "stage1_best.pth"))
            log.info("Saved new best model!")

def train_stage2a():
    cfg = CFG.STAGE2A
    device = setup(CFG.STAGE1["seed"])
    
    train_ds, val_ds = split_clf_dataset(
        str(CFG.CROP_DIR),
        cfg["class_names"],
        val_fraction=CFG.STAGE1["val_fraction"],
        seed=CFG.STAGE1["seed"],
        crop_size=cfg["crop_size"]
    )
    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=CFG.NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=CFG.NUM_WORKERS)
    
    model = RooftopClassifier(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    
    best_acc = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        log.info(f"Epoch {epoch}/{cfg['epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, acc = val_epoch_cls(model, val_loader, device)
        
        log.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            atomic_torch_save(model.state_dict(), str(CFG.CKPT_DIR / "stage2a_best.pth"))
            log.info("Saved new best model!")

def train_stage2b():
    cfg = CFG.STAGE2B
    detector = InfrastructureDetector(cfg, str(CFG.CKPT_DIR))
    yaml_path = str(CFG.YOLO_DIR / "dataset.yaml")
    log.info(f"Training Stage 2B on {yaml_path}")
    detector.train(yaml_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True, choices=["1", "2a", "2b"])
    args = parser.parse_args()
    
    if args.stage == "1":
        train_stage1()
    elif args.stage == "2a":
        train_stage2a()
    elif args.stage == "2b":
        train_stage2b()
