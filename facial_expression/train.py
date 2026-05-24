import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ==========================================
# 模型定义与辅助函数 (必须放在最外层)
# ==========================================
def get_advanced_model(num_classes=7):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def train_model(model, criterion, optimizer, loader, device):
    model.train()
    running_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
    return running_loss / len(loader.dataset), correct.double() / len(loader.dataset)


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


# ==========================================
# 核心执行逻辑 (严格包裹在 main 保护块内)
# ==========================================
if __name__ == '__main__':
    # 1. 参数配置
    torch.set_num_threads(8)
    BATCH_SIZE = 64
    EPOCHS = 15
    LEARNING_RATE = 0.001
    DATA_DIR = './data'
    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("cpu")
    print(f"当前训练设备: {DEVICE}")

    # 2. 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 检查数据集路径是否存在
    if not os.path.exists(os.path.join(DATA_DIR, 'train')):
        raise FileNotFoundError(f"未找到数据集，请确保当前目录下存在 {os.path.join(DATA_DIR, 'train')}")

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=val_transform)

    # 🛠️ 终极修复：num_workers 必须严格设为 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    print(f"检测到情绪类别: {class_names}")

    # 3. 初始化模型
    model = get_advanced_model(num_classes=7).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- 开始训练高级模型 (ResNet18) ---")

    # 4. 训练循环
    for epoch in range(EPOCHS):
        loss, acc = train_model(model, criterion, optimizer, train_loader, DEVICE)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss:.4f} - Acc: {acc:.4f}")

    # 5. 保存模型与生成图表
    torch.save(model.state_dict(), 'best_emotion_model.pth')
    print("模型已成功保存为 'best_emotion_model.pth'")

    preds, targets = evaluate_model(model, val_loader, DEVICE)
    print("\n--- 详细分类报告 ---")
    print(classification_report(targets, preds, target_names=class_names))

    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("混淆矩阵已生成并保存为 confusion_matrix.png")