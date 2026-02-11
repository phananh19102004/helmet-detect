# Helmet Detection
End User Demo :
![YOLOv11 Demo Result](https://drive.google.com/uc?id=1w_zFRckzhzBmv6xWuh82-aN3Rcx6AeRJ)

## Project Overview
This project focuses on **fine-tuning YOLOv11** for object detection tasks and implementing **MLOps** principles to streamline the model deployment and monitoring process. The goal is to create a scalable and efficient pipeline for training, fine-tuning, and deploying the YOLOv11 model in real-world applications.

## Author
Phan Nguyễn Phan Anh ( Final year student of USTH )


### Key Features:
- **YOLOv11 Fine-Tuning**: This project utilizes the YOLOv11 architecture to fine-tune the model on custom datasets for various object detection tasks, ensuring high accuracy and performance.
- **MLOps Integration**: The project incorporates MLOps principles to manage and automate the training, testing, and deployment phases. It includes version control for models, continuous integration, and model monitoring.
- **Automated Workflow**: The system automates the pipeline for training and deploying the model with minimal manual intervention, allowing for easier experimentation and model updates.
- **Model Deployment**: The fine-tuned model is deployed using modern MLOps tools, ensuring quick updates and seamless integration into production environments.

### Technologies Used:
- **YOLOv11** for object detection and fine-tuning.
- **MLflow** for model tracking, experimentation, and version control.
- **Docker** for containerization and ensuring consistency across environments.
- **FastAPI** for creating the inference API for deployment.
- **Grafana** for monitoring

## Getting Started
To get started with this project, clone the repository and follow the installation instructions to set up the environment and run the model training pipeline.

## Dataset
Dowload : https://drive.google.com/file/d/1G7W88sdRbPvUgPe6vxyKuKmJF6cleCCV/view?usp=sharing
## Prerequisites
- Docker Desktop (Windows/macOS) or Docker Engine (Linux)
- Docker Compose v2
Build and start all services (API, MLflow, Grafana, etc.):
```bash
docker compose up -d --build
```
Check running services :
```bash
docker compose ps
```
**Access Services**

- API: http://localhost:8000

- MLflow UI: http://localhost:5000

- Grafana: http://localhost:3000

## Training Results

The YOLOv11 model was fine-tuned on a custom dataset.  
Below are the main training outcomes and sample predictions.

### Metrics
- **Model**: YOLOv11
- **Epochs**: 50
- **Batch size**: 8
- **Image size**: 640
- **mAP@0.5**: 0.78
- **mAP@0.5:0.95**: 0.52
![YOLOv11 Training Result](https://drive.google.com/uc?id=1n8nKG6hlWP7j2sok2zJPYeGX_cWZvEqN)

<p align="center">
  <img src="https://drive.google.com/uc?id=1kQALdI-o7aMqNkxT4dhNZQZRUKgdLLm5" width="500"/>
</p>


<p align="center">
  <img src="https://drive.google.com/uc?id=1jBZF4sYQ44ou4_06HJfkkyGzbWaKX-nJ" width="700"/>
</p>

### Sample Prediction

Predictions with yolov11 :

![YOLOv11 Training Result](https://drive.google.com/uc?id=12rQ7m0g4c_xzduALehlDnxNRc9Wlw_EI)
