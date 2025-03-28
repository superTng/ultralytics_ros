from ultralytics import YOLO
yolo = YOLO("./yolov8n.pt", task="detect")
result = yolo(source="F:/001_WorkProjects1\YoLoProjects/ultralytics_ros/assets/123.png", save=True)