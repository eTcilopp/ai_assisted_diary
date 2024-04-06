
# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

RUN apt-get update \
    && apt-get install -y pkg-config default-libmysqlclient-dev gcc \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /ai_assisted_diary

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python3", "main.py"]
