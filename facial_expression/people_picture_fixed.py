import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from facenet_pytorch import MTCNN  # 💡 引入地表最强人脸检测神经网络

# ==========================================
# 1. 基础配置
# ==========================================
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
DEVICE = torch.device("cpu")


# ==========================================
# 2. 载入在训练阶段保存的情绪识别模型
# ==========================================
def load_saved_model():
    print("正在加载 ResNet18 情绪识别模型...")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 7)

    if not os.path.exists('best_emotion_model.pth'):
        raise FileNotFoundError("找不到 best_emotion_model.pth，请先运行 train.py 训练模型！")

    model.load_state_dict(torch.load('best_emotion_model.pth', map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


emotion_model = load_saved_model()

# ==========================================
# 3. 图像预处理流程
# ==========================================
data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 4. 💡 初始化 MTCNN 深度学习人脸检测器
# ==========================================
print("正在加载 MTCNN 深度学习人脸检测引擎...")
# keep_all=True 表示保留图像中的所有检测到的人脸（专为多人合影设计）
mtcnn = MTCNN(keep_all=True, device=DEVICE)

# ==========================================
# 5. 核心逻辑：批量遍历处理 test1 到 test9
# ==========================================
print("\n🚀 开始执行基于纯深度学习的端到端情绪分析任务...")

for i in range(1, 10):
    IMAGE_PATH = fr'.\result\people_picture_fixed_input\test{i}.jpg'
    OUTPUT_PATH = fr'.\result\people_picture_fixed_output\result{i}.jpg'

    if not os.path.exists(IMAGE_PATH):
        print(f"⏭️ 跳过: 找不到图片 {IMAGE_PATH}")
        continue

    print(f"\n--- 正在读取图片: {IMAGE_PATH} ---")
    frame = cv2.imread(IMAGE_PATH)

    if frame is None:
        print(f"⚠️ 错误: 无法解析图片 {IMAGE_PATH}")
        continue

    # MTCNN 需要 RGB 格式的图像
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    # 💡 使用神经网络进行人脸检测！
    # boxes: 人脸框坐标 [x1, y1, x2, y2]，probs: 置信度概率
    boxes, probs = mtcnn.detect(pil_img)

    if boxes is None:
        print("  ⚠️ 警告: MTCNN 未能检测到任何人脸！原图将直接保存。")
    else:
        print(f"  ✅ MTCNN 成功锁定 {len(boxes)} 张人脸，正在进行情绪预测...")

        # 遍历神经网络找到的每一个脸框
        for box, prob in zip(boxes, probs):
            # 获取坐标并防止越界 (MTCNN 返回的是浮点数，且有可能是负数边界)
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            # 如果框因为边界问题变成无效的，直接跳过
            if x2 <= x1 or y2 <= y1:
                continue

            # 1. 在原图上画出耀眼的绿色人脸框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 2. 裁剪出人脸区域 (ROI)
            roi_rgb = rgb_frame[y1:y2, x1:x2]
            roi_pil = Image.fromarray(roi_rgb)

            # 3. 预处理与模型预测
            input_tensor = data_transform(roi_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = emotion_model(input_tensor)
                _, preds = torch.max(outputs, 1)
                emotion = class_names[preds.item()]

            print(f"    -> 🎯 [置信度: {prob:.2f}] 预测结果: {emotion}")

            # 4. 动态排版文字 (防越界)
            text_y = y1 - 10 if y1 > 30 else y1 + 30

            # 双层文字渲染
            cv2.putText(frame, emotion, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(frame, emotion, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # ==========================================
    # 6. 保存每一张的处理结果
    # ==========================================
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, frame)
    print(f"  💾 处理完成！结果已保存为: result{i}.jpg")

print("\n🎉 端到端深度学习群像情绪分析全部完成！请去验收成果！")