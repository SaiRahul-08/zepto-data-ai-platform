FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch to avoid downloading CUDA/NVIDIA packages
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY docs ./support_assistance/docs
COPY src ./support_assistance/src
COPY build_index.py .

ENV MOCK_LLM=1
ENV PYTHONUNBUFFERED=1

RUN python build_index.py

EXPOSE 7860

CMD ["uvicorn", "support_assistance.src.api:app", "--host", "0.0.0.0", "--port", "7860"]