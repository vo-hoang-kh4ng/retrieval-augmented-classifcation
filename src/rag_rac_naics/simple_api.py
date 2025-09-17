"""Simple FastAPI server without heavy dependencies."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json

# Import our simple classifier
from .simple_main import SimpleClassifier, load_sample_data

app = FastAPI(
    title="Simple RAG RAC NAICS API",
    description="Lightweight business classification API",
    version="1.0.0"
)

# Global classifier
classifier = None

@app.on_event("startup")
async def startup_event():
    """Initialize classifier on startup."""
    global classifier
    print("Initializing Simple RAG RAC NAICS API...")
    
    # Load data and train classifier
    sample_data = load_sample_data()
    texts = [item["text"] for item in sample_data]
    labels = [item["label"] for item in sample_data]
    
    classifier = SimpleClassifier()
    classifier.fit(texts, labels)
    
    print(f"✓ Classifier initialized with {len(texts)} examples")

class ClassificationRequest(BaseModel):
    query: str

class ClassificationResponse(BaseModel):
    query: str
    prediction: str
    confidence: float
    examples: List[Dict[str, Any]]

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Simple RAG RAC NAICS Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "classifier_ready": classifier is not None,
        "training_examples": len(classifier.texts) if classifier else 0
    }

@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """Classify business text into NAICS codes."""
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    try:
        result = classifier.classify(request.query)
        
        return ClassificationResponse(
            query=request.query,
            prediction=result["prediction"],
            confidence=result["confidence"],
            examples=result["examples"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify/batch")
async def classify_batch(requests: List[ClassificationRequest]):
    """Classify multiple texts."""
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    results = []
    for req in requests:
        try:
            result = classifier.classify(req.query)
            results.append(ClassificationResponse(
                query=req.query,
                prediction=result["prediction"],
                confidence=result["confidence"],
                examples=result["examples"]
            ))
        except Exception as e:
            results.append({
                "query": req.query,
                "error": str(e)
            })
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
