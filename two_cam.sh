#!/bin/bash

# Check if base_env exists, if not create it with requirements_base.txt
if [ ! -d "base_env" ]; then
    echo "Creating base_env from requirements_base.txt..."
    python3 -m venv base_env
    source base_env/bin/activate
    pip install -r requirements_base.txt
else
    echo "base_env already exists. Activating..."
    source base_env/bin/activate
fi

# Run the publisher and subscriber
echo "Running ros_publisher_double.py and ros_subscriber.py..."
python3 ros_publisher_double.py &
python3 ros_subscriber.py

wait