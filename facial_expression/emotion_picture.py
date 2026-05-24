import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ==========================================
# 1. 基础配置
# ==========================================
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
DEVICE = torch.device("cpu")


# ==========================================
# 2. 载入在训练阶段保存的模型 (只加载一次)
# ==========================================
def load_saved_model():
    print("正在初始化计算引擎，加载模型权重...")
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
# 4. 初始化人脸检测器 (只加载一次)
# ==========================================
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# ==========================================
# 5. 核心循环：批量读取并识别
# ==========================================
print("\n🚀 开始批量自动化情绪识别与画框任务...")

for i in range(1, 10):
    # 动态构建输入和输出路径
    image_path = fr'.\result\emotion_picture_input\test{i}.jpg'
    output_path = fr'.\result\emotion_picture_output\result{i}.jpg'

    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"⏭️ 跳过: 找不到图片 {image_path}")
        continue

    print(f"\n--- 正在处理第 {i} 张图片: test{i}.jpg ---")
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"⚠️ 读取失败: 无法解析图片 {image_path}")
        continue

    # 将彩色图转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测图片中的所有人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("  ⚠️ 警告: 没有在图片中检测到人脸！原图将直接保存。")
    else:
        print(f"  ✅ 成功检测到 {len(faces)} 张人脸，正在分析...")

    # 遍历检测到的每一张人脸
    for (x, y, w, h) in faces:
        # 画出绿色的人脸框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 裁剪出人脸区域并转为 RGB
        roi_bgr = frame[y:y + h, x:x + w]
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)

        # 图像预处理与预测
        input_tensor = data_transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            emotion = class_names[preds.item()]

        print(f"    -> 预测结果: {emotion}")

        # 在人脸框上方写上双层标签（黑底绿字，防止背景太亮看不清）
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 确保输出文件夹存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存结果
    cv2.imwrite(output_path, frame)
    print(f"  💾 结果已保存为: result{i}.jpg")

print("\n🎉 批量情绪画框识别全部完成！请前往输出文件夹查看结果。")