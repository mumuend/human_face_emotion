# 🎭 Human Face Emotion Identify (人脸情绪识别系统)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-lightgrey.svg)

本项目是一个基于深度学习的人脸情绪识别与视觉可解释性系统。通过迁移学习微调 ResNet18 模型，系统能够精准捕捉并分析人脸微表情。项目涵盖了从算法训练、多目标检测、动态视频流处理到神经网络注意力可视化的一整套工业级 AI 解决方案。

## ✨ 核心功能 (Features)

本项目实现了以下五大核心能力：
1. **🖼️ 单幅图像情绪识别：** 支持多种图片格式上传，对人脸进行七类情绪预测。
2. **👨‍👩‍👧‍👦 多目标群像检测：** 集成 OpenCV Haar 与深度学习检测引擎，支持多人合影场景下的独立情绪追踪。
3. **🎥 动态视频帧分析：** 支持上传 MP4 视频，实现全帧逐帧人脸框选与情绪标注，并自动转码以供网页端播放。
4. **🔥 视觉可解释性 (Grad-CAM)：** 引入梯度加权类激活映射，生成神经网络关注区域热力图，实现“AI 决策过程可视化”。
5. **💻 全能 Web 交互界面：** 基于 Gradio 构建的 SaaS 级工作站，提供实时拍照、海量文件夹批量处理及一键清空重置等功能。

## 🧠 模型与数据集 (Dataset & Models)

* **数据集：** 使用 FER-2013 公共数据集 (包含 Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral 七类情绪)。
* **基线模型 (Baseline)：** 搭建基础卷积神经网络 (CNN) 作为验证模型。
* **主力模型 (Core Engine)：** 采用迁移学习 (Transfer Learning) 策略，通过微调预训练的 **ResNet18** 网络，实现了在复杂野外场景下的高鲁棒性。

## 🛠️ 安装说明 (Installation)

在终端运行以下命令：

**1. 克隆项目**
```bash
git clone [https://github.com/YourUsername/human_face_emotion_identify.git](https://github.com/YourUsername/human_face_emotion_identify.git)
cd human_face_emotion_identify
**2. 安装核心依赖库

Bash
pip install torch torchvision
pip install opencv-python
pip install gradio
pip install moviepy
pip install numpy Pillow
