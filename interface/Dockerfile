FROM python:3.12.4

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads templates static

EXPOSE 5000

CMD ["python", "app.py"]