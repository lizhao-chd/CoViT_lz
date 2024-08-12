# opencv-python
import cv2
# mediapipe人工智能工具包
import matplotlib.pyplot as plt
import mediapipe as mp
# 进度条库
from tqdm import tqdm
# 时间库
import time
import math
import numpy as np
import os


mp_face_mesh = mp.solutions.face_mesh
model = mp_face_mesh.FaceMesh(
        static_image_mode=False,      # 是静态图片还是连续视频帧，摄像头画面为连续视频帧，此处选False
        refine_landmarks=True,       # 使用Attention Mesh模型，对嘴唇、眼睛、瞳孔周围的关键点精细定位
        max_num_faces=1,              # 最多检测几张脸
        min_detection_confidence=0.01, # 置信度阈值
        min_tracking_confidence=0.01,  # 追踪阈值
)

# 导入可视化函数和可视化样式
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_frame(img):
    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取预测结果
    results = model.process(img_RGB)
    # 绘制人脸曲面和重点区域轮廓线
    if results.multi_face_landmarks:  # 如果检测出人脸
        for face_landmarks in results.multi_face_landmarks:  # 遍历每一张脸
            face_aoi=eyes_fun(face_landmarks,img_RGB)
            # 绘制人脸网格
    else:
        pass
    return face_aoi

def eyes_fun(facelms,img):
    lmList = []
    x_list=[]
    y_list=[]
    for index, lm in enumerate(facelms.landmark):
        h, w, c = img.shape  # 分别存放图像长\宽\通道数
        cx, cy = int(lm.x * w), int(lm.y * h)  # 比例坐标x乘以宽度得像素坐标
        lmList.append([index, cx, cy])
        x_list.append(cx)
        y_list.append(cy)
        # 在21个关键点上换个圈，img画板，坐标(cx,cy)，半径5，蓝色填充
    face_aoi=img_invert(cv2.resize(np.array(img[min(y_list)+50:max(y_list)-150, min(x_list):max(x_list) ]),(512,224)))
    return face_aoi



def img_invert(img):
    b, g, r = cv2.split(img)
    img = cv2.merge((r, g, b))
    return img




def img_imshow(img,name):
    cv2.imshow(name, img)
    cv2.waitKey(1)


if __name__ == '__main__':
    # 获取摄像头，传入0表示获取系统默认摄像头
    cap = cv2.VideoCapture(r'D:\all_data_sets\car_following_test\luyisi_40.mp4')

    i=1
    while cap.isOpened():
        # 获取画面
        success, frame = cap.read()
        frame = np.array(frame[25:700, 200:1100])
        # cv2.imshow('video', frame)
        # cv2.waitKey(0)

        if i%2==0:
            try:
                print("第%d帧"%i)
                face= process_frame(frame )
                img_imshow(face,"face")


            except:
                pass
        i+=1

    # 关闭摄像头
    cap.release()
    # 关闭图像窗口
    cv2.destroyAllWindows()
