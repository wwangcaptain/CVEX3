import os
import pickle
import cv2
from deepface import DeepFace
from mtcnn import MTCNN
import numpy as np

# 数据集路径
dataset_path = "video_Data"

# 初始化MTCNN人脸检测器
detector = MTCNN()

# 训练函数：提取训练集的特征并保存为模型文件
def train_recognition_model():
    print("开始提取人脸特征，请稍候...")
    model = DeepFace.build_model("VGG-Face")  # 使用VGG-Face模型
    face_db = {}  # 用于存储每个人的特征向量

    # 遍历每个人的文件夹
    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)
        if os.path.isdir(person_path):
            person_name = person_folder  # 使用文件夹名作为标识符
            print(f"正在处理: {person_name}")

            # 获取 `train` 文件夹路径
            train_path = os.path.join(person_path, "train")
            if not os.path.isdir(train_path):
                print(f"跳过无效目录: {train_path}")
                continue

            embeddings = []  # 用于存储该人的所有人脸特征
            for image_file in os.listdir(train_path):
                image_path = os.path.join(train_path, image_file)
                try:
                    # 读取图片并检测人脸
                    image = cv2.imread(image_path)
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    faces = detector.detect_faces(rgb_image)

                    # 如果检测到人脸，提取最大人脸的区域
                    if faces:
                        x, y, w, h = faces[0]['box']  # 默认提取第一个检测到的人脸
                        face_image = rgb_image[y:y + h, x:x + w]

                        # 提取特征向量
                        embedding = DeepFace.represent(img_path=face_image, model_name="VGG-Face", enforce_detection=False)
                        embeddings.append(embedding[0]["embedding"])  # 提取第一个特征向量
                except Exception as e:
                    print(f"跳过无法处理的图片: {image_path}, 错误信息: {e}")

            if embeddings:
                # 计算该人的特征向量的平均值
                avg_embedding = np.mean(embeddings, axis=0)  # 计算平均值
                face_db[person_name] = avg_embedding

    # 保存特征数据库到文件
    model_file = "face_recognition_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(face_db, f)
    print(f"人脸识别模型已保存到 {model_file}")

if __name__ == "__main__":
    train_recognition_model()