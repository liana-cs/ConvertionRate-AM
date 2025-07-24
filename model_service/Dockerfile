FROM python:3.12.4

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .
COPY data/ ./data/
COPY mlruns/ ./mlruns/

CMD ["python", "train.py"]