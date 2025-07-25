FROM python:3.12.4

WORKDIR /app

COPY model_service/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001

CMD ["python", "app.py"]
