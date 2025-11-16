import torch
from torch import nn
from dataloader import ASLDataModule
from model import ASL_BiLSTM
import numpy as np

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
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    # Run training/validation; also get objects needed for final prediction
    model = ASL_BiLSTM().to(device)
    # load the best model
    model.load_state_dict(torch.load("best_model.pt"))

    #validation stats from model on training
    val_metrics = predict(model, val_loader, device)
    print("Val accuracy:", val_metrics["accuracy"])
    print(val_metrics["report"])
    cfm = ConfusionMatrixDisplay(
        confusion_matrix=val_metrics["confusion_matrix"],
        display_labels=[0, 1, 2, 3]
    )  # Citation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
    cfm.plot()


    test_df = 

    #Citation for .copy(): https://www.w3schools.com/python/pandas/ref_df_copy.asp
    X_test = test_df[feature_cols].copy()
    y_test = test_df["price_range"].copy()
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    # build test loader and evaluate
    test_loader = DataLoader(MobilePriceDataset(X_test, y_test), batch_size=64, shuffle=False)
    test_metrics = predict(model, test_loader, device)
    print("Test accuracy:", test_metrics["accuracy"])
    print(test_metrics["report"])
    cfm_test = ConfusionMatrixDisplay(
        confusion_matrix=test_metrics["confusion_matrix"],
        display_labels=[0, 1, 2, 3]
    )
    cfm_test.plot()