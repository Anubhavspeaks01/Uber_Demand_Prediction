# set up the base image
FROM python:3.12.7

# set the working directory
WORKDIR /app

# copy requirements first (better caching)
COPY requirements-docker.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# install dvc (with http support for DagsHub)
RUN pip install --no-cache-dir dvc[http]

# copy entire project (including .dvc files)
COPY . .

# pull data from DVC remote
RUN dvc pull

# expose the port
EXPOSE 8000

# run the streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]