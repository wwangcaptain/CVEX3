import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 仿射变换函数（仅旋转和平移）
def apply_affine_transform(image, angle=0, scale=1.0, translation=(0, 0)):
    # 计算旋转矩阵
    center = (image.shape[1] / 2, image.shape[0] / 2)  # 图像中心
    M = cv2.getRotationMatrix2D(center, angle, scale)  # 旋转矩阵
    M[0, 2] += translation[0]
    M[1, 2] += translation[1]
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

# 加载数据
def load_data(dataset_path):
    images = []
    labels = []
    person_ids = []
    for i in range(1, 176):  # 175个人
        for j in range(1, 8):  # 每个人7张照片
            for variant in ['original', 'noisy', 'blurred']:
                img_path = os.path.join(dataset_path, f'{i:03d}_{j:02d}_{variant}.bmp')
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # 应用中值滤波来减少椒盐噪声的影响
                    if 'noisy' in variant:
                        img = cv2.medianBlur(img, 3)  # 使用3x3的核进行中值滤波

                    # 仿射变换
                    img = apply_affine_transform(img, angle=np.random.uniform(-5, 5), scale=1.0,
                                                 translation=(np.random.randint(-1, 1), np.random.randint(-2, 2)))
                    images.append(img)
                    labels.append(i - 1)  # 标签从0开始
                    person_ids.append(i)
    return np.array(images), np.array(labels), person_ids

# 创建LBPH人脸识别器
def train_lbph_model(X_train, y_train):
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # 创建LBPH识别器
    recognizer.train(X_train, y_train)  # 训练模型
    return recognizer

# 测试LBPH模型
def test_lbph_model(recognizer, X_val, y_val):
    predictions = []
    for img in X_val:
        label, confidence = recognizer.predict(img)  # 预测标签和置信度
        predictions.append(label)
    return np.array(predictions)

# 加载数据集
dataset_path = 'feret_improved'  # FERET数据集路径
images, labels, person_ids = load_data(dataset_path)

# 随机划分训练集与验证集
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.5, random_state=42)

# 训练LBPH模型
recognizer = train_lbph_model(X_train, y_train)

# 在验证集上进行预测
y_pred = test_lbph_model(recognizer, X_val, y_val)

# 输出分类报告
print(classification_report(y_val, y_pred))

# 混淆矩阵可视化
cm = confusion_matrix(y_val, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()