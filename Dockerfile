FROM python:3.8.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN pip install torch-scatter==2.0.9 torch-sparse==0.6.12