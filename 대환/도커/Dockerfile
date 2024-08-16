FROM python:3.8-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y git apt-utils

# Install pip and necessary Python packages
RUN pip install --upgrade pip
COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install -r /opt/ml/code/requirements.txt --no-warn-script-location

# Install segment-anything directly from GitHub
RUN pip install git+https://github.com/facebookresearch/segment-anything.git --no-warn-script-location

# Copy the model and inference script
COPY model /opt/ml/model
COPY inference.py /opt/ml/code/inference.py
COPY serve.py /opt/ml/code/serve.py

ENV PYTHONUNBUFFERED=TRUE

# Define the entrypoint for the Docker container
ENTRYPOINT ["python", "/opt/ml/code/serve.py"]
