FROM python:3.12
WORKDIR /app

# install the application requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy in the source code
COPY src ./src
EXPOSE 8080

# setup an app user container
RUN useradd app
USER app

# runt the container
CMD ['python','src/predict.py', "--host", "0.0.0.0", "--port", "8080"]