if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0   
    )
