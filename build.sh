#!/bin/bash

echo "Pulling latest..."
git pull

echo "Building image..."
docker build -t fheonix/mercedes-chat:0.0.1 .

echo "Pushing image..."
docker push fheonix/mercedes-chat:0.0.1

echo "Done."


