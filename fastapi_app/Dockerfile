FROM python:3.11.2-slim  
  

WORKDIR /app  
  

COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  
RUN python3 -m spacy download en_core_web_md
  

COPY . .  
  

EXPOSE 8080  
ENV  MLFLOW_TRACKING_URI=file:/app/mlruns 

# Run the application  
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]