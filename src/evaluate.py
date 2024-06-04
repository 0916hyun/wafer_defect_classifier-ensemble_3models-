import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_best_model(model_class, model_name, num_classes, device):
    model = model_class(num_classes)
    model.load_state_dict(torch.load(f'save_best_model/{model_name}_best_model.pth'))
    model.to(device)
    return model

def voting_ensemble_predict(models, dataloader, device):
    all_outputs = []
    for model in models:
        model.eval()
        outputs = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                output = model(inputs)
                outputs.append(output.cpu().numpy())
        all_outputs.append(np.concatenate(outputs, axis=0))

    avg_outputs = np.mean(all_outputs, axis=0)
    ensemble_preds = np.argmax(avg_outputs, axis=1)
    return ensemble_preds

def evaluate_voting_ensemble(models, dataloader, device):
    all_targets = []
    for _, labels in dataloader:
        all_targets.extend(labels.numpy())

    ensemble_preds = voting_ensemble_predict(models, dataloader, device)
    cm = confusion_matrix(all_targets, ensemble_preds)
    val_accuracy = 100 * np.sum(ensemble_preds == np.array(all_targets)) / len(all_targets)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Label')
    plt.title('Voting Ensemble Confusion Matrix')
    plt.show()

    print(f'Voting Ensemble Validation Accuracy: {val_accuracy:.2f}%')
    return val_accuracy
