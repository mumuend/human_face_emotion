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

        # 挂载钩子 (Hooks) 来捕获前向传播的特征图和反向传播的梯度
        self.target_layer.register_forward_hook(self.save_feature)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_feature(self, module, input, output):
        self.features = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx):
        # 1. 前向传播
        output = self.model(input_tensor)

        # 2. 清空梯度并针对指定类别进行反向传播
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        # 3. 计算通道权重 (对梯度的全球平均池化)
        weights = torch.mean(self.gradients, dim=(2, 3))[0]

        # 4. 将权重与特征图线性组合
        heatmap = torch.zeros(self.features.shape[2:], dtype=torch.float32).to(DEVICE)
        for i, w in enumerate(weights):
            heatmap += w * self.features[0, i, :, :]

        # 5. ReLU 激活（只保留对分类有正向贡献的特征区域）
        heatmap = torch.clamp(heatmap, min=0)

        # 6. 归一化处理
        heatmap = heatmap / torch.max(heatmap)
        return heatmap.cpu().numpy()


# ==========================================
# 3. 核心执行逻辑 (批量处理架构)
# ==========================================
if __name__ == '__main__':
    print("正在初始化计算引擎，加载 ResNet18 模型...")

    # 1. 载入模型（放在循环外部，只需加载 1 次）
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 7)

    if not os.path.exists('best_emotion_model.pth'):
        raise FileNotFoundError("找不到 best_emotion_model.pth，请检查文件位置！")

    # 添加了 weights_only=True 消除警告
    model.load_state_dict(torch.load('best_emotion_model.pth', map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    # 2. 预处理定义（只需定义 1 次）
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 指定目标层并初始化 GradCAM（只需初始化 1 次）
    target_layer = model.layer4
    cam = GradCAM(model, target_layer)

    print("\n🚀 开始批量生成 Grad-CAM 热力图...")

    # 4. 开始批量循环 (处理 test1 到 test9)
    for i in range(1, 10):
        # 动态构建输入输出路径 (修复了输入路径可能拼写错误的文件夹名)
        IMAGE_PATH = fr'.\result\visualize_gradcam_input\test{i}.jpg'
        OUTPUT_PATH = fr'.\result\visualize_gradcam_output\result{i}.jpg'

        # 如果图片不存在，跳过当前循环，防止程序崩溃
        if not os.path.exists(IMAGE_PATH):
            print(f"⏭️ 跳过: 找不到图片 {IMAGE_PATH}")
            continue

        print(f"\n--- 正在处理: {IMAGE_PATH} ---")

        # 读取原图
        orig_img = cv2.imread(IMAGE_PATH)
        if orig_img is None:
            print(f"⚠️ 无法解析图片 {IMAGE_PATH}")
            continue

        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        # 先预测该图属于哪个类别 (使用 eval 模式)
        model.eval()
        with torch.no_grad():
            preds = model(input_tensor)
            predicted_class_idx = torch.argmax(preds, dim=1).item()
            emotion = class_names[predicted_class_idx]
            print(f"  📊 模型预测情绪为: {emotion}")

        # 生成该预测类别的热力图
        input_tensor.requires_grad = True
        model.train()  # 临时切回 train 模式以允许计算梯度
        heatmap = cam.generate_heatmap(input_tensor, predicted_class_idx)
        model.eval()  # 计算完毕后立刻切回评估模式，保证下一张图不受影响

        # 后处理：调整热力图大小并应用伪彩色
        heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # 融合热力图与原图
        blended_img = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)

        # 💡 [加分项]: 在热力图左上角动态标注预测的情绪文本（黑底绿字抗干扰）
        text = f"{emotion}"
        cv2.putText(blended_img, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(blended_img, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # 确保输出文件夹存在
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        # 保存图片
        cv2.imwrite(OUTPUT_PATH, blended_img)
        print(f"  ✅ 热力图已保存至: result{i}.jpg")

    print("\n🎉 批量 Grad-CAM 分析任务全部完成！请前往输出文件夹查看。")