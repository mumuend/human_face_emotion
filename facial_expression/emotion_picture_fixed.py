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
# 4. 初始化人脸检测器
# ==========================================
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# ==========================================
# 5. 核心逻辑：批量遍历处理 test1 到 test9
# ==========================================
print("\n🚀 开始批量自动化检测任务...")

for i in range(1, 10):
    IMAGE_PATH = fr'.\result\emotion_picture_fixed_input\test{i}.jpg'
    OUTPUT_PATH = fr'.\result\emotion_picture_fixed_output\result{i}.jpg'

    if not os.path.exists(IMAGE_PATH):
        print(f"⏭️ 跳过: 找不到图片 {IMAGE_PATH}")
        continue

    print(f"\n--- 正在读取图片: {IMAGE_PATH} ---")
    frame = cv2.imread(IMAGE_PATH)

    if frame is None:
        print(f"⚠️ 错误: 无法解析图片 {IMAGE_PATH}")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 尝试使用 OpenCV 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 💡【核心修复区：保底机制】
    if len(faces) == 0:
        print("  ⚠️ 警告: 传统算法未能检测到人脸！(可能存在剧烈姿态变化或遮挡)")
        print("  🔄 触发保底机制：将整张图片作为人脸区域强制进行情绪分析...")
        # 获取图片的宽和高
        h_img, w_img = frame.shape[:2]
        # 强制制造一个覆盖整图的虚拟人脸框 [x=0, y=0, width=图片宽, height=图片高]
        faces = [(0, 0, w_img, h_img)]
    else:
        print(f"  ✅ 成功检测到 {len(faces)} 张人脸，正在进行情绪分析...")

    # 遍历检测到的（或我们伪造的）每一张人脸
    for (x, y, w, h) in faces:
        # 画出绿色的人脸框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 裁剪出人脸区域
        roi_bgr = frame[y:y + h, x:x + w]
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)

        input_tensor = data_transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            emotion = class_names[preds.item()]

        print(f"    -> 预测结果: {emotion}")

        # 💡【动态文字防越界处理】
        # 如果人脸框太靠图片顶部（y < 30），文字就画在框里面往下一点，否则画在框上面
        text_y = y - 10 if y > 30 else y + 30

        cv2.putText(frame, emotion, (x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(frame, emotion, (x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # ==========================================
    # 6. 保存每一张的处理结果
    # ==========================================
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, frame)
    print(f"  💾 处理完成！结果已保存为: result{i}.jpg")

print("\n🎉 批量情绪分析画框任务全部完成！请前往文件夹查看 result1 到 result9。")