#!/bin/bash

# Log everything
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."

aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 696637901268.dkr.ecr.eu-north-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 696637901268.dkr.ecr.eu-north-1.amazonaws.com/uber-demand-prediction:latest

echo "Checking existing container..."

if [ "$(docker ps -q -f name=uber-demand)" ]; then
  echo "Stopping existing container..."
  docker stop uber-demand
fi

if [ "$(docker ps -aq -f name=uber-demand)" ]; then
  echo "Removing existing container..."
  docker rm uber-demand
fi

echo "Starting new container..."

docker run -d \
  --name uber-demand \
  -p 80:8000 \
  696637901268.dkr.ecr.eu-north-1.amazonaws.com/uber-demand-prediction:latest

echo "Container started successfully."