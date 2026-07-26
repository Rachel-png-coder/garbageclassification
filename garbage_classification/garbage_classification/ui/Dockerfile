FROM python:3.10-slim

WORKDIR /app

COPY ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ui/ ./ui/

EXPOSE 8501

CMD streamlit run ui/app.py --server.port $PORT --server.address 0.0.0.0