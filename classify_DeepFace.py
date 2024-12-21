import os
import numpy as np
import cv2
from deepface import DeepFace
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# 加载数据并提取深度特征
def load_data_and_extract_features(dataset_path):
    features = []
    labels = []
    for i in range(1, 176):  # 175个人，编号从1到175
        for j in range(1, 8):  # 每个人7张原始照片
            for variant in ['original', 'blurred', 'noisy']:  # 增强后的图像类型
                img_path = os.path.join(dataset_path, f'{i:03d}_{j:02d}_{variant}.bmp')
                if not os.path.exists(img_path):
                    print(f"Warning: File not found {img_path}")
                    continue

                # 加载灰度图像
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Error: Failed to load image {img_path}")
                    continue

                img = cv2.resize(img, (224, 224))  # DeepFace要求输入图像尺寸为224x224

                # 将灰度图像转换为3通道彩色图像
                img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # 使用DeepFace提取图像的特征
                try:
                    #  'VGG-Face', 'Facenet', 'ArcFace'
                    result = DeepFace.represent(img_colored, model_name='VGG-Face', enforce_detection=False)
                    features.append(result[0]['embedding'])  # 提取embedding特征
                    labels.append(i)  # 标签为人物编号（从0开始）
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
    return np.array(features), np.array(labels)


# 创建一个KNN分类器（或其他分类器）
from sklearn.neighbors import KNeighborsClassifier

# 加载数据集并提取特征
dataset_path = 'feret_improved'  # 数据集路径
features, labels = load_data_and_extract_features(dataset_path)

# 检查是否成功加载了所有特征
if len(features) == 0:
    raise ValueError("No features were extracted. Please check the dataset path and image format.")

# 随机划分训练集与验证集
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.5, random_state=42)

# 创建并训练KNN模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 在验证集上进行预测
y_pred = knn.predict(X_val)

# 输出分类报告
print(classification_report(y_val, y_pred))

# 混淆矩阵可视化
cm = confusion_matrix(y_val, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()