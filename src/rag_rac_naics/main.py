# Main entry point
import argparse
from src.rag_rac_naics.classifiers.domain_classifier import DomainClassifier
from src.rag_rac_naics.classifiers.rac_classifier import RACClassifier
from src.rag_rac_naics.retrieval.naics_retriever import NAICSRetriever
from src.rag_rac_naics.clients import LLMClients
from src.rag_rac_naics.config import Settings

def main():
    parser = argparse.ArgumentParser(description="RAG with RAC and NAICS Classification")
    parser.add_argument("--mode", choices=["domain", "rac", "naics"], required=True)
    parser.add_argument("--query", required=True, help="Query to process")
    parser.add_argument("--config", default="config/hyperparameters.json")
    
    args = parser.parse_args()
    
    # Load configuration
    settings = Settings()
    clients = LLMClients.from_env()
    
    if args.mode == "domain":
        classifier = DomainClassifier(clients)
        result = classifier.classify(args.query)
        print(f"Domain classification: {result}")
    
    elif args.mode == "rac":
        # TODO: Implement RAC classification
        print("RAC classification not implemented yet")
    
    elif args.mode == "naics":
        # TODO: Implement NAICS classification
        print("NAICS classification not implemented yet")

if __name__ == "__main__":
    main()
