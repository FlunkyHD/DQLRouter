FROM python:alpine3.8

COPY requirements.txt service.py /

RUN pip install --upgrade pip && pip install -r requirements.txt 

ENTRYPOINT ["python", "service.py"]