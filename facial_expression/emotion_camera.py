import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. 类别映射与设备配置
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
DEVICE = torch.device("cpu")


# 2. 载入在训练阶段保存的 ResNet18 模型架构与权重
def load_saved_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 7)  # 与训练时的分类数保持一致
    model.load_state_dict(torch.load('best_emotion_model.pth', map_location=DEVICE, weights_only=True))
    model.to(DEVICE)  # 确保模型完全转移到 CPU 内存
    model.eval()
    return model


model = load_saved_model()

# 3. 定义与验证集完全相同的图像预处理流程
data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. 初始化 OpenCV 人脸检测器 (Haar 级联分类器)
# 注意：你需要确保本地或当前环境有这个 xml 文件，通常 OpenCV 安装时自带
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 5. 打开摄像头捕获视频流
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("错误：无法打开摄像头。")
    exit()

print("实时人脸情绪识别已启动。按 'q' 键退出退出。")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转为灰度图用于人脸位置检测（提高速度和准确度）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # 在原图上框出人脸
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 裁剪出人脸区域 (ROI)
        roi_bgr = frame[y:y + h, x:x + w]
        # OpenCV 默认是 BGR，需要转为 PyTorch 训练时要求的 RGB 格式
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        # 将 NumPy 矩阵转换为 PIL Image 以匹配 torchvision 的 transform
        pil_img = Image.fromarray(roi_rgb)
        input_tensor = data_transform(pil_img).unsqueeze(0).to(DEVICE)  # 增加 Batch 维度

        # 送入网络进行推理
        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            emotion = class_names[preds.item()]

        # 将预测的情绪文本印在人脸框上方
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 实时显示视频窗口
    cv2.imshow('Face Emotion Recognition Demo', frame)

    # 监听键盘，按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()