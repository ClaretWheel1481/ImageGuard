# 图像测试预测
# def predict_image(image_path, model, class_names):
#     model.eval()
#     img = Image.open(image_path)
#     transform = transforms.Compose([
#         transforms.Resize((299, 299)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     img = transform(img).unsqueeze(0)
#     img = img.to(device)
#
#     with torch.no_grad():
#         outputs = model(img)
#         _, predicted = torch.max(outputs, 1)
#     print(f'预测结果: {class_names[predicted[0]]}')
#
# model.load_state_dict(torch.load('model/image_guard_v1.pth', weights_only=True))
# predict_image('data/validation/porn/[www.google.com][10382].jpg', model, class_names)
