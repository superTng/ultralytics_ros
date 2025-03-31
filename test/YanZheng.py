from ultralytics import YOLO

model = YOLO("F:/001_WorkProjects1/YoLoProjects/ultralytics_ros/test/yolov8n.pt")
model.train(
    data="F:/001_WorkProjects1/YoLoProjects/ultralytics_ros/datasets/Model_Duck/data.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    device="cuda",
    workers=0  # 正确的写法
)


