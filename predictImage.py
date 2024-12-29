import torch
from torchvision import transforms
from PIL import Image
from trainModel import class_names, ImageGuard  # 确保导入 class_names 和 ImageGuard

# 图像测试预测函数
def predict_image(image_path, model, classes):
    model.eval()  # 设置模型为评估模式
    img = Image.open(image_path)

    # 图像预处理
    transform = transforms.Compose([
        # transforms.Resize(299),
        # transforms.CenterCrop(299),
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)  # 添加批量维度

    # 将图像移动到计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    img = img.to(device)

    # 执行推理
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    print(f'预测结果: {classes[predicted[0]]}')

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageGuard()  # 实例化模型
model.load_state_dict(torch.load('model/image_guard_v2.pth', map_location=device,weights_only=True))  # 加载权重
model = model.to(device)

# 调用预测函数
image_path = 'data/validation/porn/[www.google.com][18585].jpg'
predict_image(image_path, model, class_names)
