# We use the full python image because it already includes the build tools
FROM python:3.11

# Prevent Python from buffering logs (helps you see errors in Dokploy logs)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy and install requirements
# This avoids the 'apt-get' step that was failing
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Streamlit port
EXPOSE 8501

# Start command
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
