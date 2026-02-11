FROM python:3.10-slim

WORKDIR /work


RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_mlops.py .

CMD ["python", "train_mlops.py"]
