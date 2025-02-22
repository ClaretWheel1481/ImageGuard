import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from PIL import ImageFile, Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 自定义ImageFolder类以跳过无效图像
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        while True:
            try:
                sample, target = super(CustomImageFolder, self).__getitem__(index)
                if sample is not None:
                    return sample, target
            except Exception as e:
                print(f"Failed to load image {self.imgs[index][0]}: {e}")
                index = (index + 1) % len(self.imgs)

# 加载图片数据方法
def pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        print(f"Failed to load image {path}: {e}")
        return None

# 数据预处理
# ImageNet标准归一化参数
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# 加载数据集
train_dataset = CustomImageFolder('dataset', transform=data_transforms['train'], loader=pil_loader)
val_dataset = CustomImageFolder('data/validation', transform=data_transforms['val'], loader=pil_loader)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = train_dataset.classes
print(class_names)

# 构建模型(EfficientNet)
class ImageGuard(nn.Module):
    def __init__(self, num_classes=2):
        super(ImageGuard, self).__init__()

        self.base_model = EfficientNet.from_pretrained('efficientnet-b3')

        # 获取原始分类层的输入特征数
        in_features = self.base_model._fc.in_features

        # 替换全连接层
        self.base_model._fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.SiLU(),  # EfficientNet常用Swish激活（SiLU是PyTorch实现）
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

model = ImageGuard()

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# 学习率衰减
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练模型
def train_model(model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(enumerate(train_loader),total=len(train_loader))

        for step, data in loop:
            images,labels = data
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # 获取当前batch中各类别样本的数量
            unique_labels, counts = torch.unique(labels, return_counts=True)
            batch_classes = ", ".join([f"{class_names[label.item()]}: {count.item()}"
                                       for label, count in zip(unique_labels, counts)])

            loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
            loop.set_postfix(loss=running_loss / dataset_sizes['train'], classes=batch_classes)

        validate_model(model)

    return model

# 评估性能
def validate_model(model):
    model.eval()
    corrects = 0

    for data in val_loader:
        images,labels = data
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            corrects += torch.sum(preds == labels.data)

    print(f"correct: {corrects/len(val_dataset)}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练模式:{device}")
    model = model.to(device)

    # 训练
    print("开始训练")
    print("-" * 20)
    # 训练轮数
    epochs = 20
    model = train_model(model, criterion, optimizer, epochs)
    if not os.path.exists('model'):
        os.makedirs('model')
    torch.save(model.state_dict(), 'model/image_guard_v2.pth')
    print("Model saved.")
