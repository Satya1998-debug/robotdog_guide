from ultralytics import YOLO

def train():
    # Load a model
    model = YOLO('yolo11m.pt')  # load an official model (will auto-download if not present)

    # train the model
    model.train(data='object_detection/door_detector.yaml', 
                device=-1,
                epochs=100, imgsz=640, batch=32, pretrained=True,
                workers=4, lr0=0.01, patience=100, save_period=10, seed=42,
                name='yolo11m_door_detector', resume=False, save=True)
    

if __name__ == "__main__":
    from clearml import Task
    task = Task.init(project_name="robotdog_guide", task_name="train_door_detector_yolo11m")
    train()