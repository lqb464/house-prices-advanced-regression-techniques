FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.4.0" \
    && pip install --no-cache-dir -r requirements.txt
COPY backend/ /app/backend/
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY evaluation/production_report.json /app/evaluation/production_report.json
RUN mkdir -p /app/data /app/outputs/models
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
