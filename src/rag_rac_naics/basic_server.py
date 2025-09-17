"""Basic HTTP server for classification without heavy dependencies."""

import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from .simple_main import SimpleClassifier, load_sample_data

# Global classifier
classifier = None

def initialize_classifier():
    """Initialize the classifier."""
    global classifier
    print("Initializing classifier...")
    
    sample_data = load_sample_data()
    texts = [item["text"] for item in sample_data]
    labels = [item["label"] for item in sample_data]
    
    classifier = SimpleClassifier()
    classifier.fit(texts, labels)
    
    print(f"✓ Classifier ready with {len(texts)} examples")

class ClassificationHandler(BaseHTTPRequestHandler):
    """HTTP request handler for classification."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "message": "Simple RAG RAC NAICS Classification Server",
                "version": "1.0.0",
                "endpoints": {
                    "GET /": "This help message",
                    "GET /health": "Health check",
                    "POST /classify": "Classify text (JSON: {\"query\": \"text\"})"
                }
            }
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        elif self.path == "/health":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "status": "healthy",
                "classifier_ready": classifier is not None,
                "training_examples": len(classifier.texts) if classifier else 0
            }
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"error": "Not found"}
            self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/classify":
            try:
                # Read request body
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Parse JSON
                data = json.loads(post_data.decode('utf-8'))
                query = data.get('query', '')
                
                if not query:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = {"error": "Missing 'query' field"}
                    self.wfile.write(json.dumps(response).encode())
                    return
                
                if not classifier:
                    self.send_response(503)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = {"error": "Classifier not initialized"}
                    self.wfile.write(json.dumps(response).encode())
                    return
                
                # Classify
                result = classifier.classify(query)
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    "query": query,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "examples": result["examples"]
                }
                
                self.wfile.write(json.dumps(response, indent=2).encode())
                
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {"error": "Invalid JSON"}
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {"error": f"Classification failed: {str(e)}"}
                self.wfile.write(json.dumps(response).encode())
        
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"error": "Not found"}
            self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server(port=8000):
    """Run the HTTP server."""
    # Initialize classifier
    initialize_classifier()
    
    # Start server
    server_address = ('', port)
    httpd = HTTPServer(server_address, ClassificationHandler)
    
    print(f"🚀 Server running on http://localhost:{port}")
    print("Endpoints:")
    print("  GET  /        - API info")
    print("  GET  /health  - Health check")
    print("  POST /classify - Classify text")
    print("\nExample usage:")
    print(f'curl -X POST http://localhost:{port}/classify -H "Content-Type: application/json" -d \'{{"query": "Software development services"}}\'')
    print("\nPress Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
