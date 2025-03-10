import threading
import time

import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from trainModel import ImageGuard, class_names
from consul import Consul

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

@app.route('/api/v1/image/predict', methods=['POST'])
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

@app.route('/api/v1/image/predict', methods=['GET'])
def ping():
    return jsonify({'status': 'ok'})

def register_service_with_consul():
    """
    注册 Flask 服务到 Consul
    """
    consul = Consul()

    service_id = "ImageGuard-1"
    service_name = "ImageGuard"
    service_port = 37887
    service_address = "127.0.0.1"
    health_check_url = f"http://{service_address}:{service_port}/api/v1/image/predict"

    consul.agent.service.register(
        service_name,
        service_id=service_id,
        port=service_port,
        address=service_address,
        tags=["flask", "predictor"],
        check={
            "http": health_check_url,
            "interval": "20s",
            "timeout": "5s"
        }
    )
    print(f"Service {service_name} registered with Consul")

    try:
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        consul.agent.service.deregister(service_id)
        print(f"Service {service_name} deregistered from Consul")

if __name__ == '__main__':
    # 若不需要consul服务，则注释掉以下代码
    consul_thread = threading.Thread(target=register_service_with_consul)
    consul_thread.daemon = True
    consul_thread.start()
    ##########################################
    app.run(
        port=37887,
        debug=False,
        threaded=False,
        host='127.0.0.1'
    )
