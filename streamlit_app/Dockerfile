# Usa la imagen oficial de Python
FROM python:3.9

# Configura el directorio de trabajo
WORKDIR /app_stream

# Copia los archivos necesarios
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -m spacy download en_core_web_md

ENV  MLFLOW_TRACKING_URI=file:/app/mlruns 


COPY . .

EXPOSE 8501


CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]