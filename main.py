import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from trainModel import ImageGuard, class_names

app = Flask(__name__)

# ImageNet 标准归一化参数
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 图像预处理变换（与训练时使用的 val 预处理一致）
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageGuard()
model.load_state_dict(torch.load('model/image_guard_v2.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval()

def predict_image_from_pil(img):
    """
    接收 PIL.Image 图像，执行预处理、推理，返回预测类别名称
    """
    img = img.convert("RGB")
    tensor_img = preprocess(img).unsqueeze(0)  # 增加 batch 维度
    tensor_img = tensor_img.to(device)
    with torch.no_grad():
        outputs = model(tensor_img)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted[0].item()]

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """
    接口说明：
    - 请求方式：POST
    - 参数：file（上传的图像文件）
    - 返回：JSON 格式的预测结果，例如 {"prediction": "porn"} 或 {"prediction": "neutral"}
    """
    if 'file' not in request.files:
        return jsonify({'error': '未检测到文件上传'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '上传的文件名为空'}), 400
    try:
        # 使用 PIL 打开上传的图像文件
        img = Image.open(file)
        prediction = predict_image_from_pil(img)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 启动 Flask 服务
    app.run(
        port=37882,
        debug=False,
        threaded=False,
        host='127.0.0.1'
    )
