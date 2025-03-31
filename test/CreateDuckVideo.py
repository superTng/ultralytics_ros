import cv2
from ultralytics import YOLO
import os

# 加载 YOLO 模型
model = YOLO(r"F:\001_WorkProjects1\YoLoProjects\ultralytics_ros\runs\detect\train15\weights\best.pt")

# 设定输入视频路径
video_path = r"F:\001_WorkProjects1\YoLoProjects\ultralytics_ros\assets\duck.mp4"

# 打开视频文件
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ 无法打开视频文件:", video_path)
    exit()

# 获取视频属性
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设定输出视频路径
output_dir = r"F:\001_WorkProjects1\YoLoProjects\ultralytics_ros\results"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'result.mp4')

# 设置视频编码格式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 逐帧处理
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 目标检测
    results = model.predict(frame, device="cuda")  # 强制使用 GPU

    # 处理检测结果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # 确保转换为整数
            conf = float(box.conf)
            cls = int(box.cls)

            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 添加标签
            label = f"{result.names[cls]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存并显示处理后的视频
    out.write(frame)
    cv2.imshow('Detection Result', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ 目标检测完成，结果已保存至 {output_path}")
