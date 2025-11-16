import torch
from torch import nn, optim
from dataloader import ASLDataModule
from model import ASL_BiLSTM
import config
from analysis import predict, plot_learning_curves
from sklearn.metrics import ConfusionMatrixDisplay

def main():
    # Device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU:", torch.cuda.get_device_name(0))
        print("GPU Memory Allocated:", torch.cuda.memory_allocated(0)/1024**2, "MB")
        print("GPU Memory Cached:   ", torch.cuda.memory_reserved(0)/1024**2, "MB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Data
    dm = ASLDataModule()
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()

    # Model, Loss, Optimizer
    model = ASL_BiLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 10

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("Starting training...")

    # Training Loop
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.float().to(device), y.long().to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.float().to(device), y.long().to(device)
                logits = model(x)
                loss = criterion(logits, y)

                val_running_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += y.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs)
    
    return model, val_loader, device


if __name__ == "__main__":
    model, val_loader, device = main()

    val_metrics = predict(model, val_loader, device)
    print("Val accuracy:", val_metrics["accuracy"])
    print(val_metrics["report"])

    cfm = ConfusionMatrixDisplay(
        confusion_matrix=val_metrics["confusion_matrix"],
        display_labels=[0, 1, 2, 3, 4, 5]
    )
    cfm.plot()