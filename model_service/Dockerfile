FROM python:3.12.4

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .
COPY data/ ./data/
COPY mlruns/ ./mlruns/
COPY app.py .

RUN mkdir -p mlruns data

EXPOSE 5001

CMD ["python", "app.py"]