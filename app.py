import gradio as gr
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from moviepy import VideoFileClip

# ==========================================
# 1. 基础配置与模型加载
# ==========================================
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
DEVICE = torch.device("cpu")


def load_model():
    print("正在加载 ResNet18 模型权重...")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load('best_emotion_model.pth', map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)


# ==========================================
# 2. 🔥 Grad-CAM 核心提取器类
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


cam = GradCAM(model, model.layer4)


# ==========================================
# 3. 核心图像处理单元
# ==========================================
def process_single_frame(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))

    for (x, y, w, h) in faces:
        roi_rgb = image_rgb[y:y + h, x:x + w]
        pil_img = Image.fromarray(roi_rgb)
        input_tensor = data_transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            emotion = class_names[preds.item()]

        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_y = y - 10 if y > 30 else y + 30
        cv2.putText(image_rgb, emotion, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(image_rgb, emotion, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image_rgb


def process_gradcam_frame(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))

    if len(faces) == 0:
        face_img_rgb = image_rgb.copy()
    else:
        x, y, w, h = faces[0]
        pad_w = int(w * 0.15)
        pad_h = int(h * 0.15)
        y1, y2 = max(0, y - pad_h), min(image_rgb.shape[0], y + h + pad_h)
        x1, x2 = max(0, x - pad_w), min(image_rgb.shape[1], x + w + pad_w)
        face_img_rgb = image_rgb[y1:y2, x1:x2]

    pil_img = Image.fromarray(face_img_rgb)
    input_tensor = data_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(input_tensor)
        predicted_class_idx = torch.argmax(preds, dim=1).item()
        predicted_emotion = class_names[predicted_class_idx]

    input_tensor.requires_grad = True
    model.train()
    heatmap = cam.generate_heatmap(input_tensor, predicted_class_idx)
    model.eval()

    heatmap_resized = cv2.resize(heatmap, (face_img_rgb.shape[1], face_img_rgb.shape[0]))
    heatmap_color_bgr = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

    blended_rgb = cv2.addWeighted(face_img_rgb, 0.6, heatmap_color_rgb, 0.4, 0)

    text = f"Attention: {predicted_emotion}"
    cv2.putText(blended_rgb, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(blended_rgb, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return blended_rgb


# ==========================================
# 4. 各模块专属调度函数
# ==========================================
def handle_image(image):
    if image is None: return None
    return process_single_frame(image.copy())


def handle_gradcam(image):
    if image is None: return None
    return process_gradcam_frame(image.copy())


def handle_video(video_path):
    if video_path is None: return None

    temp_out_path = "temp_output.mp4"
    final_web_path = "web_ready_video.mp4"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_out_path, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_rgb = process_single_frame(frame_rgb)
        processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        out.write(processed_bgr)

    cap.release()
    out.release()

    try:
        clip = VideoFileClip(temp_out_path)
        clip.write_videofile(final_web_path, codec="libx264", audio=False, logger=None)
        clip.close()
        return final_web_path
    except Exception as e:
        print(f"转码遇到问题: {e}")
        return temp_out_path


def handle_folder(files):
    if not files: return []
    processed_images = []
    for file_obj in files:
        ext = os.path.splitext(file_obj.name)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            img_bgr = cv2.imread(file_obj.name)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                processed_rgb = process_single_frame(img_rgb)
                processed_images.append(processed_rgb)
    return processed_images


# ==========================================
# 5. 构建高级企业级界面 (Gradio Blocks)
# ==========================================
# 💡 【修复警告】去掉了此处的 theme 参数，适应 Gradio 6.0 标准
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🎭 全能 AI 人脸情绪识别与可解释性工作站
        本系统基于 ResNet18 构建，提供四大核心模块：单图分析、视频流追踪、海量文件清洗，以及**基于 Grad-CAM 的神经网络视觉可解释性分析**。
        """
    )

    with gr.Tabs():
        # 标签页 1：图片与拍照
        with gr.TabItem("🖼️ 基础群像识别 / 📷 拍照"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(sources=["upload", "webcam"], label="上传多人合影或使用摄像头拍照")
                with gr.Column():
                    img_output = gr.Image(label="情绪矩阵追踪结果")
            # 💡 【修复报错】将按钮放在图片框下方，此时 img_output 已被定义
            with gr.Row():
                img_btn = gr.Button("🔍 启动识别引擎", variant="primary")
                img_clear = gr.ClearButton(components=[img_input, img_output], value="🗑️ 清空内容")

            img_btn.click(fn=handle_image, inputs=img_input, outputs=img_output)

        # 标签页 2：视觉热力图
        with gr.TabItem("🔥 视觉可解释性 (Grad-CAM)"):
            gr.Markdown(
                "**学术级探测仪：** 此模块通过捕获网络反向传播梯度，逆向生成注意力热力图。建议上传单人面部特写获得最佳分析效果。")
            with gr.Row():
                with gr.Column():
                    cam_input = gr.Image(sources=["upload", "webcam"], label="上传待剖析的人脸特写")
                with gr.Column():
                    cam_output = gr.Image(label="Grad-CAM 视觉可解释性热力图")
            with gr.Row():
                cam_btn = gr.Button("🧠 捕获神经网络注意力", variant="primary")
                cam_clear = gr.ClearButton(components=[cam_input, cam_output], value="🗑️ 清空内容")

            cam_btn.click(fn=handle_gradcam, inputs=cam_input, outputs=cam_output)

        # 标签页 3：视频分析
        with gr.TabItem("🎥 动态视频帧分析"):
            gr.Markdown("**⚠️ 算力提示：** 视频按帧处理运算量极大，请上传 5-10 秒内的测试片段。")
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="拖拽上传 MP4 短视频")
                with gr.Column():
                    vid_output = gr.Video(label="多目标追踪渲染结果")
            with gr.Row():
                vid_btn = gr.Button("🎬 启动逐帧算力分析 (附 H.264 编译)", variant="primary")
                vid_clear = gr.ClearButton(components=[vid_input, vid_output], value="🗑️ 清空内容")

            vid_btn.click(fn=handle_video, inputs=vid_input, outputs=vid_output)

        # 标签页 4：文件夹批量处理
        with gr.TabItem("📁 海量文件夹流批处理"):
            gr.Markdown("将包含多张图片的 **文件夹** 拖拽到下方区域，系统将自动过滤非图片文件并构建结果画廊。")
            with gr.Row():
                with gr.Column():
                    folder_input = gr.File(file_count="directory", label="注入数据集文件夹")
                with gr.Column():
                    folder_output = gr.Gallery(label="全局数据清洗与识别画廊", columns=3, height="auto")
            with gr.Row():
                folder_btn = gr.Button("🚀 触发批量吞吐", variant="primary")
                folder_clear = gr.ClearButton(components=[folder_input, folder_output], value="🗑️ 清空内容")

            folder_btn.click(fn=handle_folder, inputs=folder_input, outputs=folder_output)

if __name__ == "__main__":
    # 💡 【修复警告】在 Gradio 6.0 中，将主题参数转移到 launch 函数中
    demo.launch(theme=gr.themes.Soft())