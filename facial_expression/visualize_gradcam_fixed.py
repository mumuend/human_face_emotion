import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================================
# 1. 基础配置
# ==========================================
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
DEVICE = torch.device("cpu")


# ==========================================
# 2. 手动实现 Grad-CAM 核心提取器
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None

        self.target_layer.register_forward_hook(self.save_feature)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_feature(self, module, input, output):
        self.features = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, class_idx]
        loss.backward()

        weights = torch.mean(self.gradients, dim=(2, 3))[0]
        heatmap = torch.zeros(self.features.shape[2:], dtype=torch.float32).to(DEVICE)

        for i, w in enumerate(weights):
            heatmap += w * self.features[0, i, :, :]

        heatmap = torch.clamp(heatmap, min=0)
        heatmap = heatmap / torch.max(heatmap)
        return heatmap.cpu().numpy()


# ==========================================
# 3. 核心执行逻辑 (批量处理架构)
# ==========================================
if __name__ == '__main__':
    # 【优化点1】：在循环外部加载模型，只需加载1次，极大节省时间
    print("正在初始化计算引擎，加载 ResNet18 模型...")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load('best_emotion_model.pth', map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    # 初始化 OpenCV 级联分类器（也只加载1次）
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    target_layer = model.layer4
    cam = GradCAM(model, target_layer)

    print("\n🚀 开始批量自动化检测任务...")

    # 【优化点2】：使用 for 循环遍历 1 到 9
    for i in range(1, 10):
        # 动态构建输入和输出路径
        image_path = fr'.\result\visualize_gradcam_fixed_input\test{i}.jpg'
        output_path = fr'.\result\visualize_gradcam_fixed_output\result{i}.jpg'

        # 检查当前数字的图片是否存在，不存在则跳过，不让程序崩溃
        if not os.path.exists(image_path):
            print(f"⏭️ 跳过: 找不到图片 {image_path}")
            continue

        print(f"--- 正在处理第 {i} 张图片: test{i}.jpg ---")

        orig_img = cv2.imread(image_path)
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

        # 扫描人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))

        if len(faces) == 0:
            print("  ⚠️ 警告：未检测到人脸，将使用整图分析。")
            face_img = orig_img.copy()
        else:
            # 提取紧凑人脸 ROI
            x, y, w, h = faces[0]
            pad_w = int(w * 0.15)
            pad_h = int(h * 0.15)

            y1, y2 = max(0, y - pad_h), min(orig_img.shape[0], y + h + pad_h)
            x1, x2 = max(0, x - pad_w), min(orig_img.shape[1], x + w + pad_w)

            face_img = orig_img[y1:y2, x1:x2]

        # 预处理与预测
        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            preds = model(input_tensor)
            predicted_class_idx = torch.argmax(preds, dim=1).item()
            predicted_emotion = class_names[predicted_class_idx]
            print(f"  📊 预测情绪为: {predicted_emotion}")

        # 生成热力图
        input_tensor.requires_grad = True
        model.train()
        heatmap = cam.generate_heatmap(input_tensor, predicted_class_idx)
        model.eval()  # 生成完热力图后立刻切回评估模式，保持规范

        # 后处理与图像融合
        heatmap_resized = cv2.resize(heatmap, (face_img.shape[1], face_img.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        blended_img = cv2.addWeighted(face_img, 0.6, heatmap_color, 0.4, 0)

        # 绘制黑底绿字的双层文字标签
        text = f"{predicted_emotion}"
        position = (15, 35)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        cv2.putText(blended_img, text, position, font, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(blended_img, text, position, font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)

        # 确保输出文件夹存在并保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, blended_img)
        print(f"  ✅ 成功保存至: result{i}.jpg")

    print("\n🎉 批量自动化处理全部完成！请前往文件夹查看结果。")