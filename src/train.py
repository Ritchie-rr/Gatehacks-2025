import torch
from torch import nn, optim
from dataloader import ASLDataModule
from model import ASL_BiLSTM
import config
from analysis import predict, plot_learning_curves
from sklearn.metrics import ConfusionMatrixDisplay

def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    dm = ASLDataModule()
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()

    model = ASL_BiLSTM().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 10

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("Starting training...")

    # Training Loop
    for epoch in range(1, config.EPOCHS + 1):
        # ----- TRAIN -----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

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

        # ----- VALIDATION -----
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
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

        # ----- EARLY STOPPING -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "asl_best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")
    plot_learning_curves(train_losses, val_losses, train_acc, val_acc)
    
    return model, val_loader, device


if __name__ == "__main__":
    model, val_loader, device = main()
    #validation stats from model on training
    val_metrics = predict(model, val_loader, device)
    print("Val accuracy:", val_metrics["accuracy"])
    print(val_metrics["report"])
    cfm = ConfusionMatrixDisplay(
        confusion_matrix=val_metrics["confusion_matrix"],
        display_labels=[0, 1, 2, 3]
    )  # Citation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
    cfm.plot()