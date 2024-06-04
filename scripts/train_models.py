import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import WaferDataset
from src.models import DenseNetModel, ResNetModel, EfficientNetModel
from src.train import train_model

# Define transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load datasets
df_train = pd.read_pickle("../data/dataset_train.pickle")
df_val = pd.read_pickle("../data/dataset_val.pickle")

train_dataset = WaferDataset(df_train, transform=transform)
val_dataset = WaferDataset(df_val, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Number of classes
num_classes = 7

# Initialize models
densenet_model = DenseNetModel(num_classes).to(device)
resnet_model = ResNetModel(num_classes).to(device)
efficientnet_model = EfficientNetModel(num_classes).to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer_densenet = torch.optim.Adam(densenet_model.parameters(), lr=0.0001)
optimizer_resnet = torch.optim.Adam(resnet_model.parameters(), lr=0.0001)
optimizer_efficientnet = torch.optim.Adam(efficientnet_model.parameters(), lr=0.0001)

# Train models
train_model(densenet_model, train_loader, val_loader, criterion, optimizer_densenet, device, num_epochs=20)
train_model(resnet_model, train_loader, val_loader, criterion, optimizer_resnet, device, num_epochs=20)
train_model(efficientnet_model, train_loader, val_loader, criterion, optimizer_efficientnet, device, num_epochs=20)
