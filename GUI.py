import os
import pickle
import cv2
import tkinter as tk
from tkinter import filedialog  # 用于文件对话框
from PIL import Image, ImageTk  # 用于处理和显示图像
from deepface import DeepFace
from mtcnn import MTCNN
from scipy.spatial.distance import cosine

# 加载训练好的模型（特征数据库）
model_file = "face_recognition_model.pkl"
with open(model_file, "rb") as f:
    face_db = pickle.load(f)

# 打印加载的数据库
print(f"Loaded face database: {face_db.keys()}")  # 查看数据库中存储的所有身份

# 初始化MTCNN人脸检测器
detector = MTCNN()

# 人脸识别函数
def recognize_face(face_image):
    # 将图像转换为RGB格式
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # 保存图像为临时文件，并提取特征
    temp_file_path = "temp_face.jpg"
    cv2.imwrite(temp_file_path, face_image)

    # 使用DeepFace提取特征
    model_name = "VGG-Face"  # 使用VGG-Face模型
    embeddings = DeepFace.represent(img_path=temp_file_path, model_name=model_name, enforce_detection=False)

    # 打印提取的嵌入特征
    print(f"Extracted embeddings: {embeddings}")  # 打印提取的嵌入特征

    # 找到最相似的人
    min_distance = float("inf")
    identity = "Unknown"
    for person_name, person_embedding in face_db.items():
        # 计算余弦距离
        distance = cosine(embeddings[0]['embedding'], person_embedding)  # 获取第一个嵌入向量
        print(f"Comparing with {person_name}: distance={distance}")  # 打印与每个数据库特征的距离
        if distance < min_distance:
            min_distance = distance
            identity = person_name

    # 判断是否为已知身份
    if min_distance > 0.4:  # 可根据实际情况调整阈值
        identity = "Unknown"

    # 打印识别结果到终端
    print(f"识别结果：{identity}")

    return identity

# GUI显示图像
def show_image(image, panel):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panel.config(image=image)
    panel.image = image

# 上传按钮回调函数
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg")])
    if not file_path:
        return  # 如果用户取消选择，什么都不做
    image = cv2.imread(file_path)

    # 在左侧显示上传的图片
    show_image(image, left_panel)

    # 人脸检测
    faces = detector.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for face in faces:
        x, y, w, h = face['box']
        face_image = image[y:y + h, x:x + w]
        identity = recognize_face(face_image)

        # 在右侧显示检测到的人脸图，并标注识别结果
        image_with_identity = image.copy()  # 创建图像副本，用于右侧显示
        cv2.putText(image_with_identity, identity, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        show_image(image_with_identity, right_panel)

    # GUI中的识别结果
    result_label.config(text=f"识别完成：{identity}")
    # 终端显示的识别结果
    print(f"识别完成：{identity}")

# 结束按钮回调函数
def end_process():
    root.quit()

# GUI界面设置
root = tk.Tk()
root.title("人脸识别系统")

# 左右两个图像框
left_frame = tk.Frame(root)
left_frame.pack(side="left", padx=10, pady=10)
left_panel = tk.Label(left_frame, width=100, height=100, bg="lightgray")  # 左侧矩形框
left_panel.pack()

right_frame = tk.Frame(root)
right_frame.pack(side="left", padx=10, pady=10)
right_panel = tk.Label(right_frame, width=100, height=100, bg="lightgray")  # 右侧矩形框
right_panel.pack()

# 结果标签
result_label = tk.Label(root, text="识别结果：", font=("Arial", 12))
result_label.pack(pady=10)

# 上传按钮
upload_button = tk.Button(root, text="上传", command=upload_image)
upload_button.pack(side="left", padx=10)

# 结束按钮
end_button = tk.Button(root, text="结束", command=end_process)
end_button.pack(side="left", padx=10)

root.mainloop()