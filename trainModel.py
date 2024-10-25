import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        # 做标准化、归一化
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据集
train_dataset = datasets.ImageFolder('dataset/', transform=data_transforms['train'], loader=pil_loader)
val_dataset = datasets.ImageFolder('data/validation', transform=data_transforms['val'], loader=pil_loader)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = train_dataset.classes
print(class_names)

# 构建模型
class ImageGuard(nn.Module):
    def __init__(self):
        super(ImageGuard, self).__init__()
        self.base_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.base_model.aux_logits = False
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 4),  # 4 classes
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.base_model(x)

model = ImageGuard()

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.RMSprop(model.parameters(), lr=0.0001)

# 训练模型
def train_model(model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in (train_loader if phase == 'train' else val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 训练20轮
model = train_model(model, criterion, optimizer, num_epochs=20)
torch.save(model.state_dict(), 'model/image_guard_v1.pth')
print("Model saved.")

# 图像测试预测
def predict_image(image_path, model, class_names):
    model.eval()
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    print(f'预测结果: {class_names[predicted[0]]}')

model.load_state_dict(torch.load('model/image_guard_v1.pth', weights_only=True))
predict_image('data/validation/porn/[www.google.com][10382].jpg', model, class_names)
