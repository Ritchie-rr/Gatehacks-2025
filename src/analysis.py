import torch
from model import ASL_BiLSTM
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from dataloader import ASLDataModule

def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    # TODO: Use this function to plot learning curve
    """Plot learning curves for loss and accuracy"""
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def predict(model, X_test_loader, device):
        """
        Evaluate model on test data and return comprehensive metrics
        """
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in X_test_loader:
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100 * correct / total
        
        # TODO: Add more evaluation metrics
        # Hint: Use classification_report, confusion_matrix from sklearn.metrics
        # Print or return precision, recall, F1-score for each class
        # metrics
        report_text = classification_report(all_targets, all_predictions, digits=3)
        cm = confusion_matrix(all_targets, all_predictions, labels=[0, 1, 2, 3])
        rep_dict = classification_report(all_targets, all_predictions, output_dict=True)
        macro_f1 = rep_dict["macro avg"]["f1-score"]
        weighted_f1 = rep_dict["weighted avg"]["f1-score"]

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "report": report_text,
            "confusion_matrix": cm,
        }

if __name__ == "__main__":
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

    # Run training/validation; also get objects needed for final prediction
    model = ASL_BiLSTM().to(device)
    # load the best model
    model.load_state_dict(torch.load("best.pt"))

    dm = ASLDataModule()
    dm.setup()
    test_loader = dm.test_dataloader()

    test_metrics = predict(model, test_loader, device)
    print("Test accuracy:", test_metrics["accuracy"])
    print(test_metrics["report"])
    cfm_test = ConfusionMatrixDisplay(
        confusion_matrix=test_metrics["confusion_matrix"],
        display_labels=[0, 1, 2, 3]
    )
    cfm_test.plot()
    plt.show()