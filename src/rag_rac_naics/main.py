# Main entry point
import argparse
import os
import json
from typing import List, Dict, Any
from src.rag_rac_naics.classifiers.domain_classifier import DomainClassifier
from src.rag_rac_naics.classifiers.rac_classifier import RACClassifier
from src.rag_rac_naics.retrieval.naics_retriever import NAICSRetriever
try:
    from src.rag_rac_naics.retrieval.vector_store import VectorStore
except ImportError:
    # Fallback to simple vector store if ChromaDB is not available
    from src.rag_rac_naics.retrieval.simple_vector_store import VectorStore
from src.rag_rac_naics.embeddings.naics_embeddings import NAICSEmbeddings
from src.rag_rac_naics.config import LLMClients, Settings

def load_sample_data(data_path: str = "data/sample_data.json") -> List[Dict[str, Any]]:
    """Load sample training data"""
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Return sample data if file doesn't exist
        return [
            {"text": "Software development and programming services", "label": "541511"},
            {"text": "Web design and development", "label": "541511"},
            {"text": "Mobile app development", "label": "541511"},
            {"text": "Restaurant and food service", "label": "722513"},
            {"text": "Fast food restaurant", "label": "722513"},
            {"text": "Coffee shop and cafe", "label": "722515"},
            {"text": "Retail clothing store", "label": "448140"},
            {"text": "Online retail sales", "label": "454110"},
            {"text": "Consulting services", "label": "541611"},
            {"text": "Management consulting", "label": "541611"}
        ]

def main():
    parser = argparse.ArgumentParser(description="RAG with RAC and NAICS Classification")
    parser.add_argument("--mode", choices=["domain", "rac", "naics", "train"], required=True)
    parser.add_argument("--query", help="Query to process (required for domain/rac/naics modes)")
    parser.add_argument("--config", default="config/hyperparameters.json")
    parser.add_argument("--data", default="data/sample_data.json", help="Path to training data")
    parser.add_argument("--train-rac", action="store_true", help="Train RAC classifier before classification")
    
    args = parser.parse_args()
    
    # Load configuration
    settings = Settings.from_file(args.config)
    clients = LLMClients.from_env()
    
    # Create necessary directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    if args.mode == "domain":
        if not args.query:
            print("Error: --query is required for domain mode")
            return
        
        classifier = DomainClassifier(clients)
        result = classifier.classify(args.query)
        print(f"Domain classification: {result}")
        
        if result == "IN_DOMAIN":
            print("✓ Query is in domain - proceeding with RAC/NAICS classification recommended")
        else:
            print("✗ Query is out of domain - may not get good results from RAC/NAICS")
    
    elif args.mode == "rac":
        if not args.query:
            print("Error: --query is required for rac mode")
            return
        
        # Initialize components
        vector_store = VectorStore(db_path="data/processed/rac_chroma_db", collection_name="rac_collection")
        rac_classifier = RACClassifier(clients, vector_store, k=settings.retrieval_top_k)
        
        # Check if we need to train first
        if args.train_rac or vector_store.get_collection_count() == 0:
            print("Training RAC classifier...")
            sample_data = load_sample_data(args.data)
            texts = [item["text"] for item in sample_data]
            labels = [item["label"] for item in sample_data]
            
            rac_classifier.fit(texts, labels)
            print(f"✓ Trained RAC classifier with {len(texts)} examples")
        
        # Perform classification
        print(f"Classifying query: '{args.query}'")
        result = rac_classifier.classify_with_retrieval(args.query, use_llm=True)
        
        print(f"\nRAC Classification Results:")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Method: {result['method']}")
        print(f"\nRetrieved Examples:")
        for i, (text, label, score) in enumerate(result['retrieved_examples'], 1):
            print(f"  {i}. [{label}] {text} (score: {score:.3f})")
    
    elif args.mode == "naics":
        if not args.query:
            print("Error: --query is required for naics mode")
            return
        
        # Initialize components
        vector_store = VectorStore(db_path="data/processed/naics_chroma_db", collection_name="naics_collection")
        embeddings = NAICSEmbeddings(clients)
        naics_retriever = NAICSRetriever(vector_store, embeddings)
        
        # Check if vector store has data
        if vector_store.get_collection_count() == 0:
            print("Training NAICS retriever...")
            sample_data = load_sample_data(args.data)
            texts = [item["text"] for item in sample_data]
            labels = [item["label"] for item in sample_data]
            
            # Get embeddings and add to vector store
            text_embeddings = embeddings.get_embeddings(texts)
            vector_store.add_texts(texts, text_embeddings, labels)
            print(f"✓ Added {len(texts)} NAICS examples to vector store")
        
        # Perform NAICS classification
        print(f"Classifying query: '{args.query}'")
        
        # Step 1: Get shortlist
        shortlist = naics_retriever.shortlist_codes(args.query, k=settings.naics_shortlist_k)
        print(f"\nShortlisted {len(shortlist)} NAICS codes:")
        for item in shortlist:
            print(f"  {item['code']}: {item['description']} (score: {item['score']:.3f})")
        
        # Step 2: Final classification
        result = naics_retriever.classify(args.query, shortlist)
        print(f"\nFinal NAICS Classification:")
        print(f"Prediction: {result['prediction']}")
        print(f"Method: {result.get('from', 'unknown')}")
        if result.get('chosen'):
            chosen = result['chosen']
            print(f"Selected: {chosen['code']} - {chosen['description']}")
    
    elif args.mode == "train":
        print("Training all components...")
        
        # Load training data
        sample_data = load_sample_data(args.data)
        texts = [item["text"] for item in sample_data]
        labels = [item["label"] for item in sample_data]
        
        print(f"Loaded {len(texts)} training examples")
        
        # Train RAC classifier
        print("Training RAC classifier...")
        rac_vector_store = VectorStore(db_path="data/processed/rac_chroma_db", collection_name="rac_collection")
        rac_classifier = RACClassifier(clients, rac_vector_store, k=settings.retrieval_top_k)
        rac_classifier.fit(texts, labels)
        print("✓ RAC classifier trained")
        
        # Train NAICS retriever
        print("Training NAICS retriever...")
        naics_vector_store = VectorStore(db_path="data/processed/naics_chroma_db", collection_name="naics_collection")
        embeddings = NAICSEmbeddings(clients)
        text_embeddings = embeddings.get_embeddings(texts)
        naics_vector_store.add_texts(texts, text_embeddings, labels)
        print("✓ NAICS retriever trained")
        
        print(f"\n✓ Training completed! Both classifiers ready to use.")
        print(f"  - RAC database: {rac_vector_store.get_collection_count()} documents")
        print(f"  - NAICS database: {naics_vector_store.get_collection_count()} documents")

if __name__ == "__main__":
    main()
