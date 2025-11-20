# DockerFile


FROM python:3.11-slim

# Prevent python from writing pyc files and buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED=1

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY fraud ./fraud
COPY api ./api

# Create logs dir iniside container
RUN mkdir -p logs

# Port
EXPOSE 8000

#Run api
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
