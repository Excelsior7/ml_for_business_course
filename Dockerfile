FROM python:3.12.7

WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./app /code/app

COPY ./models /code/models

ENV PYTHONPATH="${PYTHONPATH}:/code/app"

CMD ["uvicorn", "app.main:app"]