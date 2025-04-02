from ultralytics import YOLO

def train_model():
    print("sss")
    model = YOLO("F:/001_WorkProjects1/YoLoProjects/ultralytics_ros/test/yolov8n.pt")
    model.train(
        data="F:/001_WorkProjects1/YoLoProjects/ultralytics_ros/datasets/Model_Helmet/data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        device="cuda",
        workers=8  # 正确的写法
    )

if __name__=='__name__':
    train_model()