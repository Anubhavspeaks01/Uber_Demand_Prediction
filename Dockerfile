# set up the base image
FROM python:3.12.7

# set the working directory
WORKDIR /app/

# copy the requirements file
COPY requirements-docker.txt .

# install dependencies
RUN pip install -r requirements-docker.txt

# install dvc
RUN pip install dvc

# copy entire project (including .dvc files)
COPY . .

# pull data from DVC remote
RUN dvc pull

# copy the data files
COPY ./data/external/plot_data.csv ./data/external/plot_data.csv
COPY ./data/processed/test.csv ./data/processed/test.csv

# copy the models
COPY ./models/ ./models/

# copy the code file
COPY ./app.py ./app.py

# expose the port
EXPOSE 8000

# run the streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]