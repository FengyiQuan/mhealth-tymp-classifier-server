# syntax=docker/dockerfile:1.4
FROM python:3.11 AS builder

WORKDIR /app

COPY requirements.txt /app
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

COPY . /app

# ENTRYPOINT ["python3"]
CMD ["gunicorn", "-b", "0.0.0.0", "app:app"]
EXPOSE 8000
