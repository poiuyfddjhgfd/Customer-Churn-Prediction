# 1 . use the official lightweighted python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# 3. Copy only dependency file first(For Docker caching)
COPY requirements.txt .   

# 4.Install python dependencies (add curl if you use MLflow local tracking URLs)
RUN pip install --upgrade pip \
    && pip install -r requirements.txt\
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 5. Copy the entire project into the image    
COPY  . .

# Explicity copy model(in case .dockerignore excluded mlruns)
# NOTE: destination changed to /app/src/serving/model to match inference.py's path
COPY src/serving/model/app/serving/model

# Copy MLflow run(artifacts+metadata) to the flat /app/model convenience path
# Copy MLflow run (artifacts + metadata) to the flat /app/model convenience path
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/model /app/model
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/feature_columns.txt /app/model
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/preprocessing.pkl /app/model

# Make "serving "  and "app" important without the "src." prefix
# ensures logs are shown in real-time(no buffering).
# lets you import modules using from app... instead of from src.app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

#6.Expose FastAPI
EXPOSE 8000

#7. Run the FastAPI app using uvicorn (change path if needed)
CMD ["python","-m","uvicorn","src.app.main:app","--host","0.0.0.0","--port","8000"]

