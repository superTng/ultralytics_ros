import cv2
from ultralytics import YOLO
import os

# 加载训练好的YOLO模型
model = YOLO(r"F:\001_WorkProjects1\YoLoProjects\ultralytics_ros\runs\detect\train\weights\best.pt")

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否正常打开
if not cap.isOpened():
    print("无法打开摄像头！")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        break

    # 使用YOLO模型进行目标检测
    results = model(frame)

    # 在检测到的目标上绘制边界框
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 绘制边界框和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            conf = float(box.conf)
            cls = int(box.cls)
            label = f'{result.names[cls]} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示识别结果
    cv2.imshow("Camera Feed", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
