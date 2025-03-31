from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("F:/001_WorkProjects1/YoLoProjects/ultralytics_ros/runs/detect/train15/weights/best.pt")

# 进行目标检测
results = model.predict(source="F:/001_WorkProjects1/YoLoProjects/ultralytics_ros/assets/005.png", save=True)
