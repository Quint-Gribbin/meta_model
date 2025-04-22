FROM gcr.io/deeplearning-platform-release/xgboost-cpu:latest

# Install additional Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from current directory to container
COPY . .

# Set the default entry point for the container
ENTRYPOINT ["python", "arg_train.py"]