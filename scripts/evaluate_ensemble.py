import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import WaferDataset
from src.models import DenseNetModel, ResNetModel, EfficientNetModel
from src.evaluate import load_best_model, evaluate_voting_ensemble

# Define transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
df_val = pd.read_pickle("../data/dataset_val.pickle")
val_dataset = WaferDataset(df_val, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Number of classes
num_classes = 7

# Load best models
densenet_best_model = load_best_model(DenseNetModel, 'DenseNetModel', num_classes, device)
resnet_best_model = load_best_model(ResNetModel, 'ResNetModel', num_classes, device)
efficientnet_best_model = load_best_model(EfficientNetModel, 'EfficientNetModel', num_classes, device)

# Evaluate ensemble
models = [densenet_best_model, resnet_best_model, efficientnet_best_model]
evaluate_voting_ensemble(models, val_loader, device)
