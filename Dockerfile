FROM python:3.12-slim 

RUN pip install pipenv 

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"] 

RUN pipenv install --system --deploy 

COPY ["predict.py", "converted_model.tflite", "./"] 

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]  
# Use Gunicorn to serve the app, binding it to all interfaces on port 9696.
