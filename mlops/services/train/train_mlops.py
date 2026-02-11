import os
import time
import glob
import mlflow
from ultralytics import YOLO

def pick_latest_train_dir(base="runs/detect"):
    candidates = sorted(glob.glob(os.path.join(base, "train*")), key=os.path.getmtime)
    return candidates[-1] if candidates else None

if __name__ == "__main__":
    # ===== Config via ENV =====
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    experiment = os.getenv("EXPERIMENT_NAME", "yolo11-cpu")
    run_name = os.getenv("RUN_NAME", f"train_{int(time.time())}")

    data_yaml = os.getenv("DATA_YAML", "dataset/data.yaml")
    base_model = os.getenv("BASE_MODEL", "yolo11n.pt")

    epochs = int(os.getenv("EPOCHS", "50"))
    imgsz = int(os.getenv("IMGSZ", "640"))
    batch = int(os.getenv("BATCH", "4"))
    workers = int(os.getenv("WORKERS", "4"))

    
    device = os.getenv("DEVICE", "cpu")

    # ===== MLflow =====
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)

    model = YOLO(base_model)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "data": data_yaml,
            "base_model": base_model,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "workers": workers,
            "device": device,
        })

        # ===== Train =====
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
        )

        # ===== Log artifacts =====
        train_dir = pick_latest_train_dir()
        if train_dir and os.path.exists(train_dir):
            mlflow.log_artifacts(train_dir, artifact_path="ultralytics_run")

        
        if train_dir:
            weights = os.path.join(train_dir, "weights", "best.pt")
            if os.path.exists(weights):
                with open("best_path.txt", "w", encoding="utf-8") as f:
                    f.write(weights)
                mlflow.log_artifact("best_path.txt", artifact_path="meta")
