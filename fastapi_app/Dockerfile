

FROM python:3.12-slim  

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

# Install extra dependencies separately to avoid reinstalling everything if only one changes
RUN pip install --no-cache-dir python-multipart llama-index-embeddings-huggingface  

# Copy all other project files
COPY . .  

EXPOSE 8000 

CMD ["uvicorn", "sample:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
