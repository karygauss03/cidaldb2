# Use the official Python base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port that the Streamlit app will run on
EXPOSE 7777

# Run the Streamlit app when the container starts
CMD ["streamlit", "run", "--server.port", "7777", "main.py"]