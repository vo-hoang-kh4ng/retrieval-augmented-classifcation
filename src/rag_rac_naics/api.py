"""FastAPI web service for RAG RAC NAICS classification."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import json
import logging
from contextlib import asynccontextmanager

from .config import LLMClients, Settings
from .classifiers.domain_classifier import DomainClassifier
from .classifiers.rac_classifier import RACClassifier
from .retrieval.naics_retriever import NAICSRetriever
try:
    from .retrieval.vector_store import VectorStore
except ImportError:
    # Fallback to simple vector store if ChromaDB is not available
    from .retrieval.simple_vector_store import VectorStore
from .embeddings.naics_embeddings import NAICSEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for classifiers
domain_classifier = None
rac_classifier = None
naics_retriever = None
settings = None
clients = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize classifiers on startup."""
    global domain_classifier, rac_classifier, naics_retriever, settings, clients
    
    logger.info("Initializing RAG RAC NAICS API...")
    
    # Load configuration
    settings = Settings.from_file()
    clients = LLMClients.from_env()
    
    # Initialize domain classifier
    domain_classifier = DomainClassifier(clients)
    
    # Initialize RAC classifier
    rac_vector_store = VectorStore(
        db_path="data/processed/rac_chroma_db", 
        collection_name="rac_collection"
    )
    rac_classifier = RACClassifier(clients, rac_vector_store, k=settings.retrieval_top_k)
    
    # Initialize NAICS retriever
    naics_vector_store = VectorStore(
        db_path="data/processed/naics_chroma_db", 
        collection_name="naics_collection"
    )
    embeddings = NAICSEmbeddings(clients)
    naics_retriever = NAICSRetriever(naics_vector_store, embeddings)
    
    logger.info("✓ All classifiers initialized successfully")
    
    yield
    
    logger.info("Shutting down RAG RAC NAICS API...")

app = FastAPI(
    title="RAG RAC NAICS Classification API",
    description="A comprehensive API for business classification using RAG, RAC, and NAICS codes",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response Models
class ClassificationRequest(BaseModel):
    query: str = Field(..., description="Text to classify", min_length=1, max_length=1000)

class DomainResponse(BaseModel):
    query: str
    classification: str = Field(..., description="Either 'IN_DOMAIN' or 'OUT_OF_DOMAIN'")
    is_in_domain: bool

class RetrievedExample(BaseModel):
    text: str
    label: str
    score: float

class RACResponse(BaseModel):
    query: str
    prediction: str
    confidence: float
    method: str
    retrieved_examples: List[RetrievedExample]

class NAICSShortlistItem(BaseModel):
    code: str
    description: str
    score: float

class NAICSResponse(BaseModel):
    query: str
    prediction: str
    method: str
    shortlist: List[NAICSShortlistItem]
    chosen: Optional[NAICSShortlistItem] = None

class TrainingRequest(BaseModel):
    data: List[Dict[str, str]] = Field(..., description="Training data with 'text' and 'label' fields")
    retrain: bool = Field(default=False, description="Whether to clear existing data before training")

class TrainingResponse(BaseModel):
    message: str
    rac_count: int
    naics_count: int

class HealthResponse(BaseModel):
    status: str
    rac_ready: bool
    naics_ready: bool
    rac_count: int
    naics_count: int

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG RAC NAICS Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    rac_count = rac_classifier.vector_store.get_collection_count() if rac_classifier else 0
    naics_count = naics_retriever.vector_store.get_collection_count() if naics_retriever else 0
    
    return HealthResponse(
        status="healthy",
        rac_ready=rac_count > 0,
        naics_ready=naics_count > 0,
        rac_count=rac_count,
        naics_count=naics_count
    )

@app.post("/classify/domain", response_model=DomainResponse)
async def classify_domain(request: ClassificationRequest):
    """Classify if a query is in-domain or out-of-domain."""
    if not domain_classifier:
        raise HTTPException(status_code=503, detail="Domain classifier not initialized")
    
    try:
        classification = domain_classifier.classify(request.query)
        return DomainResponse(
            query=request.query,
            classification=classification,
            is_in_domain=classification == "IN_DOMAIN"
        )
    except Exception as e:
        logger.error(f"Domain classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify/rac", response_model=RACResponse)
async def classify_rac(request: ClassificationRequest):
    """Perform RAC (Retrieval-Augmented Classification)."""
    if not rac_classifier:
        raise HTTPException(status_code=503, detail="RAC classifier not initialized")
    
    # Check if RAC classifier has training data
    if rac_classifier.vector_store.get_collection_count() == 0:
        raise HTTPException(
            status_code=400, 
            detail="RAC classifier not trained. Use /train endpoint first."
        )
    
    try:
        result = rac_classifier.classify_with_retrieval(request.query, use_llm=True)
        
        retrieved_examples = [
            RetrievedExample(text=text, label=label, score=score)
            for text, label, score in result['retrieved_examples']
        ]
        
        return RACResponse(
            query=request.query,
            prediction=result['prediction'],
            confidence=result['confidence'],
            method=result['method'],
            retrieved_examples=retrieved_examples
        )
    except Exception as e:
        logger.error(f"RAC classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify/naics", response_model=NAICSResponse)
async def classify_naics(request: ClassificationRequest):
    """Perform NAICS industry classification."""
    if not naics_retriever:
        raise HTTPException(status_code=503, detail="NAICS retriever not initialized")
    
    # Check if NAICS retriever has training data
    if naics_retriever.vector_store.get_collection_count() == 0:
        raise HTTPException(
            status_code=400, 
            detail="NAICS retriever not trained. Use /train endpoint first."
        )
    
    try:
        # Get shortlist
        shortlist = naics_retriever.shortlist_codes(request.query, k=settings.naics_shortlist_k)
        
        # Get final classification
        result = naics_retriever.classify(request.query, shortlist)
        
        shortlist_items = [
            NAICSShortlistItem(code=item['code'], description=item['description'], score=item['score'])
            for item in shortlist
        ]
        
        chosen_item = None
        if result.get('chosen'):
            chosen = result['chosen']
            chosen_item = NAICSShortlistItem(
                code=chosen['code'], 
                description=chosen['description'], 
                score=chosen.get('score', 0.0)
            )
        
        return NAICSResponse(
            query=request.query,
            prediction=result['prediction'],
            method=result.get('from', 'unknown'),
            shortlist=shortlist_items,
            chosen=chosen_item
        )
    except Exception as e:
        logger.error(f"NAICS classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify/complete")
async def classify_complete(request: ClassificationRequest):
    """Complete classification pipeline: Domain -> RAC -> NAICS."""
    results = {}
    
    # Step 1: Domain classification
    try:
        domain_result = await classify_domain(request)
        results["domain"] = domain_result.dict()
        
        if not domain_result.is_in_domain:
            results["warning"] = "Query is out of domain - RAC/NAICS results may be unreliable"
    except Exception as e:
        results["domain"] = {"error": str(e)}
    
    # Step 2: RAC classification
    try:
        rac_result = await classify_rac(request)
        results["rac"] = rac_result.dict()
    except Exception as e:
        results["rac"] = {"error": str(e)}
    
    # Step 3: NAICS classification
    try:
        naics_result = await classify_naics(request)
        results["naics"] = naics_result.dict()
    except Exception as e:
        results["naics"] = {"error": str(e)}
    
    return results

@app.post("/train", response_model=TrainingResponse)
async def train_classifiers(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train both RAC and NAICS classifiers with provided data."""
    if not rac_classifier or not naics_retriever:
        raise HTTPException(status_code=503, detail="Classifiers not initialized")
    
    if len(request.data) == 0:
        raise HTTPException(status_code=400, detail="Training data cannot be empty")
    
    # Validate training data format
    for item in request.data:
        if "text" not in item or "label" not in item:
            raise HTTPException(
                status_code=400, 
                detail="Each training item must have 'text' and 'label' fields"
            )
    
    try:
        texts = [item["text"] for item in request.data]
        labels = [item["label"] for item in request.data]
        
        # Clear existing data if requested
        if request.retrain:
            rac_classifier.vector_store.clear_collection()
            naics_retriever.vector_store.clear_collection()
        
        # Train RAC classifier
        rac_classifier.fit(texts, labels)
        rac_count = rac_classifier.vector_store.get_collection_count()
        
        # Train NAICS retriever
        text_embeddings = naics_retriever.embeddings.get_embeddings(texts)
        naics_retriever.vector_store.add_texts(texts, text_embeddings, labels)
        naics_count = naics_retriever.vector_store.get_collection_count()
        
        logger.info(f"Training completed: RAC={rac_count}, NAICS={naics_count}")
        
        return TrainingResponse(
            message=f"Successfully trained classifiers with {len(texts)} examples",
            rac_count=rac_count,
            naics_count=naics_count
        )
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/train/sample")
async def train_with_sample_data():
    """Train classifiers with built-in sample data."""
    sample_data_path = "data/sample_data.json"
    
    if not os.path.exists(sample_data_path):
        raise HTTPException(status_code=404, detail="Sample data file not found")
    
    try:
        with open(sample_data_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        
        request = TrainingRequest(data=sample_data, retrain=True)
        return await train_classifiers(request, BackgroundTasks())
    except Exception as e:
        logger.error(f"Sample training error: {e}")
        raise HTTPException(status_code=500, detail=f"Sample training failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
