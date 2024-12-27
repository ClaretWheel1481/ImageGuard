import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from PIL import ImageFile, Image
from tqdm import tqdm
from torch.utils.data import DataLoader
ImageFile.LOAD_TRUNCATED_IMAGES = True

# FIXME：目前存在严重的数据集不平衡问题，待解决!!!
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

# TODO: 计算图片标准差、均值

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(299),
        transforms.RandomResizedCrop(299, scale=(0.8, 1.0), ratio=(3/4, 4/3)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
    ]),
}

# 加载数据集
train_dataset = CustomImageFolder('dataset', transform=data_transforms['train'], loader=pil_loader)
val_dataset = CustomImageFolder('data/validation', transform=data_transforms['val'], loader=pil_loader)

# TODO: 根据显存大小调整Batch_size
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = train_dataset.classes
print(class_names)

# 构建模型
class ImageGuard(nn.Module):
    def __init__(self):
        super(ImageGuard, self).__init__()

        # RESNET50
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            # TODO: 目前只有两种分类，不使用Softmax
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.base_model(x)

model = ImageGuard()

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
# TODO: Adam优化器相对更稳定，RMSprop需要控制学习率，未来调优
optimizer = optim.Adam(model.parameters(), lr=0.002)
# optimizer = optim.RMSprop(model.parameters(), lr=0.002, weight_decay=1e-5)

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

            with torch.set_grad_enabled(True):
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)

            loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
            loop.set_postfix(loss=running_loss / dataset_sizes['train'])

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
torch.save(model.state_dict(), 'model/image_guard_v1.pth')
print("Model saved.")