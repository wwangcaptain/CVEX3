import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 加载数据
def load_data(dataset_path):
    images = []
    labels = []
    for i in range(1, 121):  # 120个人
        person_images = []
        for j in range(1, 27):  # 每个人26张
            img_path = os.path.join(dataset_path, f'AR{str(i).zfill(3)}-{j}.tif')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 以灰度图形式加载
            img = cv2.resize(img, (100, 100))  # 统一尺寸
            person_images.append(img)
        images.extend(person_images)
        labels.extend([i-1] * 26)  # 标签为人物编号（从0开始）
    return np.array(images), np.array(labels)

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
dataset_path = 'AR_k120_s26_w80_h100'  # AR数据集路径
images, labels = load_data(dataset_path)

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
