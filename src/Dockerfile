FROM python:3.12-slim
WORKDIR /src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install torch==2.6.0+cu131 torchvision --extra-index-url https://download.pytorch.org/whl/cu131
COPY . .
CMD ["python", "-m", "split-learning"]