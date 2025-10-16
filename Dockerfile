FROM python:3.10-slim

# Create app directory
WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app

# Expose Gradio default port
EXPOSE 7860

CMD ["python", "app.py"]
