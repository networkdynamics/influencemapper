FROM python:3.10
LABEL authors="blodstone"

WORKDIR /app

COPY app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "listener.py"]