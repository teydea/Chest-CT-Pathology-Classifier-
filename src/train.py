import torch
import torch.nn as nn
import torch.optim as optim
import gc
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt


def validate_model(
    model, 
    val_loader, 
    device,
    criterion=None
):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_val_loss = 0.0

    with torch.no_grad():
        for data, labels in tqdm(val_loader):
            data = data.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            logits = model(data)
            
            if (criterion is not None):
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels_np = labels.cpu().numpy()

            all_labels.extend(labels_np)
            all_preds.extend(preds)
            all_probs.extend(probs)

    avg_val_loss = total_val_loss / len(val_loader)
    
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

    return f1, precision, recall, auc, avg_val_loss

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=10,
    lr=1e-3,
    wd=5e-4,
    patience=2,
    save_path='best_slice_clf_model.pth'
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=patience,
        verbose=True,
        min_lr=1e-6
    )

    history = {
        'train_loss' : [],
        'val_loss' : [],
        'val_f1' : [],
        'val_precision' : [],
        'val_recall' : [],
        'val_roc_auc' : [],
    }

    best_val_loss = float('inf')
    best_f1 = 0.0

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0.0

        for data, labels in train_loader:
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)

        val_f1, val_precision, val_recall, val_auc, avg_val_loss = validate_model(model, val_loader, device, criterion)

        print(f"Epoch {epoch+1}")
        print(f"Train loss = {avg_train_loss:.4f}")
        print(f"Val metrics: loss = {avg_val_loss:.4f}, f1-score = {val_f1:.4f}, precision = {val_precision:.4f}, recall = {val_recall:.4f}, ROC-AUC = {val_auc:.4f}")

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_roc_auc'].append(val_auc)

        if (val_f1 > best_f1):
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with F1 = {val_f1:.4f}")
            
        if (avg_val_loss < best_val_loss):
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        scheduler.step(avg_val_loss)

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return history

def plot_training_history(history):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0,0].plot(history['train_loss'], label='Train')
    axes[0,0].plot(history['val_loss'], label='Val')
    axes[0,0].set_title('Loss')
    axes[0,0].legend()
    
    axes[0,1].plot(history['val_f1'], label='Val F1', color='green')
    if 'train_f1' in history:
        axes[0,1].plot(history['train_f1'], label='Train F1', color='lightgreen')
    axes[0,1].set_title('F1 Score')
    axes[0,1].legend()
    
    axes[0,2].plot(history['val_precision'], label='Val Precision', color='blue')
    axes[0,2].plot(history['val_recall'], label='Val Recall', color='red')
    axes[0,2].set_title('Precision & Recall')
    axes[0,2].legend()
    
    axes[1,0].plot(history['val_auc'], label='Val AUC', color='purple')
    axes[1,0].set_title('ROC-AUC')
    
    axes[1,1].plot(history.get('lr', []), label='Learning Rate', color='orange')
    axes[1,1].set_title('Learning Rate')
    
    if 'val_accuracy' in history:
        axes[1,2].plot(history['val_accuracy'], label='Val Accuracy')
        axes[1,2].set_title('Accuracy')
    
    plt.tight_layout()
    return fig