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
# 2. 载入在训练阶段保存的模型 (放在循环外)
# ==========================================
def load_saved_model():
    print("正在加载模型权重...")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 7)

    if not os.path.exists('best_emotion_model.pth'):
        raise FileNotFoundError("找不到 best_emotion_model.pth，请先运行 train.py 训练模型！")

    model.load_state_dict(torch.load('best_emotion_model.pth', map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


model = load_saved_model()

# ==========================================
# 3. 定义图像预处理流程
# ==========================================
data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 4. 初始化人脸检测器 (放在循环外)
# ==========================================
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# ==========================================
# 5. 核心逻辑：批量遍历处理 test1 到 test9
# ==========================================
print("\n🚀 开始批量自动化多人情绪检测任务...")

for i in range(1, 10):
    IMAGE_PATH = fr'.\result\people_picture_input\test{i}.jpg'
    OUTPUT_PATH = fr'.\result\people_picture_output\result{i}.jpg'

    if not os.path.exists(IMAGE_PATH):
        print(f"⏭️ 跳过: 找不到图片 {IMAGE_PATH}")
        continue

    print(f"\n--- 正在读取图片: {IMAGE_PATH} ---")
    frame = cv2.imread(IMAGE_PATH)

    if frame is None:
        print(f"⚠️ 错误: 无法解析图片 {IMAGE_PATH}")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 💡【核心修复】换回严谨的检测参数，完美剔除衣服和背景上的假阳性！
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) == 0:
        print("  ⚠️ 没有在图片中检测到人脸！原图将直接保存。")
    else:
        print(f"  ✅ 成功检测到 {len(faces)} 张人脸，正在进行群像情绪分析...")

    # 遍历检测到的每一张人脸
    for (x, y, w, h) in faces:
        # 1. 在原图上画出绿色的人脸框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 2. 裁剪人脸 ROI
        roi_bgr = frame[y:y + h, x:x + w]

        # 防止因图像边缘问题裁剪出空矩阵
        if roi_bgr.size == 0:
            continue

        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)

        # 3. 预处理与模型预测
        input_tensor = data_transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            emotion = class_names[preds.item()]

        print(f"    -> 找到人脸 | 预测结果: {emotion}")

        # 💡【动态排版保护】如果人脸在画面最顶部，把字写在框里面，防止越界看不见
        text_y = y - 10 if y > 30 else y + 30

        # 双层抗干扰文字渲染
        cv2.putText(frame, emotion, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(frame, emotion, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 确保输出文件夹存在
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 保存图片
    cv2.imwrite(OUTPUT_PATH, frame)
    print(f"  💾 处理完成！结果已保存为: result{i}.jpg")

print("\n🎉 批量多人情绪分析任务全部完成！请前往输出文件夹查看。")