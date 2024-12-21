import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from deepface import DeepFace
from mtcnn import MTCNN

# 初始化MTCNN人脸检测器
detector = MTCNN()

# ID到人名的映射
id_to_name = {
    "1_BlurFace": "BlurFace",
    "2_ClifBar": "ClifBar",
    "3_David": "David",
    "4_Dudek": "Dudek",
    "5_FaceOcc1": "FaceOcc1",
    "6_FaceOcc2": "FaceOcc2",
    "7_FleetFace": "FleetFace",
    "8_Girl": "Girl",
    "9_Jumping": "Jumping",
    "10_Mhyang": "Mhyang"
}

# 图像处理函数
def detect_face(image_path):
    image = cv2.imread(image_path)
    # 转换为RGB格式，因为MTCNN需要RGB输入
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_image)

    face_regions = []
    for face in faces:
        x, y, w, h = face['box']
        face_regions.append((x, y, w, h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, face_regions

def recognize_face(image, faces):
    results = []
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        # 使用DeepFace进行VGG-Face识别
        try:
            result = DeepFace.find(face, db_path="video_Data")  # db_path为你的人脸数据库路径
            if not result.empty:
                # 获取识别出的文件名并解析编号
                identity = result.iloc[0]['identity']
                person_id = identity.split("/")[-2]  # 获取文件夹名（对应编号）
                person_name = id_to_name.get(person_id, "Unknown")
            else:
                person_name = "未识别"
        except Exception as e:
            person_name = "识别失败"
        results.append(person_name)
        cv2.putText(image, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return image, results

# 显示图像在Tkinter界面
def show_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panel.config(image=image)
    panel.image = image

# 上传按钮的回调函数
def upload_image():
    file_path = filedialog.askopenfilename()
    image, faces = detect_face(file_path)
    image, results = recognize_face(image, faces)
    show_image(image)
    # 显示识别结果
    if results:
        result_label.config(text=f"识别结果: {', '.join(results)}")
    else:
        result_label.config(text="识别结果: 未检测到人脸")

# 结束按钮的回调函数
def end_process():
    root.quit()

# GUI界面设置
root = tk.Tk()
root.title("人脸识别系统")

# 左右两个图像框
left_frame = tk.Frame(root)
left_frame.pack(side="left")
panel = tk.Label(left_frame)
panel.pack()

# 右边显示识别结果
right_frame = tk.Frame(root)
right_frame.pack(side="right")
result_label = tk.Label(right_frame, text="识别结果：", font=("Arial", 12))
result_label.pack()

# 上传按钮
upload_button = tk.Button(root, text="上传", command=upload_image)
upload_button.pack(side="left")

# 结束按钮
end_button = tk.Button(root, text="结束", command=end_process)
end_button.pack(side="right")

root.mainloop()