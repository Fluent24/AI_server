FROM python:3.10

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip 
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r /app/requirements.txt

COPY ./server /app/server/

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "10100"]

# CMD ["gunicorn", "--bind", "0:8000", "server.main:app","--workers", "3", "--worker-class", "uvicorn.workers.UvicornWorker"]