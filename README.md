# Sentiment Analysis Service

A production-ready NLP service designed to classify social media text into three categories: **Negative**, **Neutral**, and **Positive**. This project demonstrates a full ML lifecycle, from data preprocessing and model comparison to containerized deployment.

## Project Structure
```text
.
├── src/
│   ├── training/           # Model training & data processing scripts
│   │   ├── data_processing.py
│   │   ├── train.py        # Baseline training
│   │   ├── load_data.py        # load data script
│   │   └── train_advanced.py # Transformer training
│   │   └── data/ #datasets used 
│   ├── api/                # FastAPI application
│   │   ├── main.py         # API routes & Lifespan management
│   │   ├── schemas.py      # Pydantic data models
│   │   └── predictors.py   # Inference logic & Model loading
├── models/                 # Saved model artifacts (.joblib, bin)
├── tests/                  # Unit and Integration tests
├── Dockerfile              # Container definition
├── docker-compose.yml      # Orchestration & Environment config
└── requirements.txt        # Project dependencies

```

## 1. Setup and Installation

**Prerequisites**
* Python 3.12+
* Docker & Docker Compose (Optional, for containerization)

**Local Environment Setup**
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
## 2. How to Train the Models
The project includes two modeling approaches to satisfy baseline and advanced requirements.

**Baseline Model (Logistic Regression)**
Uses TF-IDF vectorization and a Logistic Regression pipeline.

```bash
python src/training/train.py
```

**Advanced Model (DistilBERT)**
Fine-tunes a distilbert-base-uncased transformer model using the Hugging Face Trainer API.

```bash
python src/training/train_advanced.py
```

**Note: Both scripts save artifacts into the models/ directory for the API to consume.**

## 3. How to Run the API
**Running via Docker**
The easiest way to run the service is using Docker Compose, which handles environment variables and volume mounting automatically.

```bash
# Build and start the service
docker compose up --build
```
The API will be available at http://localhost:8000. You can access the interactive documentation (Swagger UI) at http://localhost:8000/docs.

**Running Locally**
The API dynamically loads the model specified in the `MODEL_TYPE` environment variable.

```bash
# To run the Advanced model (Default)
export MODEL_TYPE=advanced
python -m uvicorn src.api.main:app --reload

# To run the Baseline model
export MODEL_TYPE=baseline
python -m uvicorn src.api.main:app --reload
```
## 4. Model Choices and Results
**Architecture Decisions**
* **Baseline (Logistic Regression)**: Chosen for its interpretability and extremely low computational overhead. It serves as a benchmark for performance and latency.
* **Advanced (DistilBERT)**: Selected because it provides near-BERT performance while being 40% smaller and 60% faster. It utilizes self-attention mechanisms to capture the context of social media slang and sarcasm more effectively than n-gram methods.

**Results Summary**
Results based on the `tweet_eval` (sentiment) test set:

Model = Baseline                     
Accuracy = 0..5850 
F1-Score(Weighted) = 0.5759
Inference Latency = <5ms

Model = Advanced  
Accuracy = 0.6931     
F1-Score(Weighted) = 0.6913               
Inference Latency = ~50 ms (CPU)


## 5. Testing
The project includes a robust test suite covering data cleaning logic and API integration.
```bash
# Run all tests
python -m pytest
```

## 6. API Endpoints
* `GET /health`: Check model readiness and version.
* `POST /predict`: Single text sentiment classification.
* `POST /predict/batch` : Efficient classification for a list of strings.

