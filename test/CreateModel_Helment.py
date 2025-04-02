from ultralytics import YOLO

def train_model():
    print("开始训练模型...")
    model = YOLO("F:/001_WorkProjects1/YoLoProjects/ultralytics_ros/test/yolov8n.pt")
    model.train(
        data="F:/001_WorkProjects1/YoLoProjects/ultralytics_ros/datasets/Model_Helmet/data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        device="cuda",  # 设为 "cuda" 使用 GPU，如果不支持 GPU，可改为 "cpu"
        workers=8  # 如果报错，可尝试 workers=4 或 workers=8
    )

if __name__ == '__main__':  # 修正错误
    train_model()
