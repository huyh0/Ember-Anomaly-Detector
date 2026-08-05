FROM python:3.14.6

WORKDIR /code

COPY ./app/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./dataset/test_ember_2018_v2_features.parquet /code/dataset
COPY ./app /code/app

CMD ["fastapi", "run", "app/main.py", "--port", "80"]