# Use a stable Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy only the requirements first (this helps with caching)
COPY requirements.txt .

# Install Python libraries
# We add --default-timeout=100 in case your internet is slow
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
