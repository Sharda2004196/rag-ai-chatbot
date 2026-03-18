"""
RAG AI Chatbot - Python Implementation
Uses: Groq (LLM), Cohere (Embeddings), Qdrant (Vector DB)
"""

import os
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import cohere
import requests
import json

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

COLLECTION_NAME = "rag_documents"
EMBEDDING_SIZE = 1024  # Cohere embed-english-v3.0

class RAGChatbot:
    def __init__(self):
        # Initialize clients
        self.cohere_client = cohere.Client(COHERE_API_KEY)
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections().collections
            if not any(c.name == COLLECTION_NAME for c in collections):
                self.qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=EMBEDDING_SIZE,
                        distance=Distance.COSINE
                    )
                )
                print(f"[OK] Created collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"Collection check: {e}")

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Cohere"""
        response = self.cohere_client.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        return response.embeddings

    def ingest_document(self, text: str, metadata: Dict = None):
        """Ingest a document into the vector database"""
        # Split text into chunks (simple chunking)
        chunks = self._chunk_text(text, chunk_size=1000, overlap=200)

        # Generate embeddings
        embeddings = self.embed_text(chunks)

        # Prepare points for Qdrant
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_metadata = metadata or {}
            point_metadata.update({
                "text": chunk,
                "chunk_index": idx
            })

            points.append(PointStruct(
                id=hash(chunk) % (10 ** 8),  # Simple ID generation
                vector=embedding,
                payload=point_metadata
            ))

        # Upload to Qdrant
        self.qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        print(f"[OK] Ingested {len(chunks)} chunks")
        return len(chunks)

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        # Generate query embedding
        query_embedding = self.embed_text([query])[0]

        # Search in Qdrant
        results = self.qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k
        )

        return [
            {
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            }
            for hit in results
        ]

    def chat(self, query: str, use_rag: bool = True) -> Dict:
        """Chat with RAG - retrieve context and generate response

        Args:
            query: User's question
            use_rag: If True, use RAG mode (search documents). If False, use general chat mode.
        """
        # General chat mode (no document retrieval)
        if not use_rag:
            prompt = f"""You are a helpful AI assistant. Answer the user's question accurately and concisely.

USER QUESTION: {query}

Provide a clear and helpful answer."""

            response = self._call_groq(prompt)
            return {
                "answer": response,
                "sources": [],
                "mode": "general"
            }

        # RAG mode (with document retrieval)
        # Retrieve relevant documents
        search_results = self.search(query, top_k=5)

        if not search_results:
            return {
                "answer": "I don't have enough information to answer that question. Try uploading some documents first, or switch to general chat mode.",
                "sources": [],
                "mode": "rag"
            }

        # Build context from search results
        context = "\n\n".join([
            f"[Source {i+1}]\n{result['text']}"
            for i, result in enumerate(search_results)
        ])

        # Build prompt
        prompt = f"""You are a helpful AI assistant with access to a knowledge base. Use the following context to answer the user's question accurately and concisely.

If you cannot find the answer in the context, say so clearly. Always cite which sources you used.

CONTEXT:
{context}

---

USER QUESTION: {query}

Provide a detailed and accurate answer based on the context provided. Mention which sources you referenced."""

        # Call Groq API
        response = self._call_groq(prompt)

        return {
            "answer": response,
            "sources": [
                {
                    "text": r["text"][:200] + "...",
                    "score": r["score"],
                    "metadata": r["metadata"]
                }
                for r in search_results
            ],
            "mode": "rag"
        }

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API for LLM response"""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)

            # Log error details if request fails
            if response.status_code != 200:
                print(f"[ERROR] Groq API returned {response.status_code}")
                print(f"Response: {response.text}")

                # Try to parse error message
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', response.text)
                    raise Exception(f"Groq API Error: {error_msg}")
                except:
                    raise Exception(f"Groq API Error {response.status_code}: {response.text}")

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            raise Exception("Groq API request timed out after 30 seconds")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Groq API request failed: {str(e)}")


def main():
    """Interactive CLI for RAG Chatbot"""
    print("=" * 60)
    print("RAG AI Chatbot - Python Edition")
    print("=" * 60)
    print()

    # Check API keys
    if not GROQ_API_KEY:
        print("[ERROR] GROQ_API_KEY not set!")
        return
    if not COHERE_API_KEY:
        print("[ERROR] COHERE_API_KEY not set!")
        return
    if not QDRANT_URL:
        print("[ERROR] QDRANT_URL not set!")
        return

    print("[OK] API keys configured")
    print()

    # Initialize chatbot
    chatbot = RAGChatbot()
    print("[OK] RAG Chatbot initialized")
    print()

    # Interactive loop
    print("Commands:")
    print("  /ingest <text>  - Add text to knowledge base")
    print("  /quit          - Exit")
    print("  <question>     - Ask a question")
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                print("Goodbye!")
                break

            if user_input.startswith("/ingest "):
                text = user_input[8:]
                chunks = chatbot.ingest_document(text, {"source": "manual"})
                print(f"[OK] Ingested {chunks} chunks\n")

            else:
                # Chat
                print("Thinking...")
                result = chatbot.chat(user_input)
                print(f"\nBot: {result['answer']}\n")

                if result['sources']:
                    print(f"[Sources] Used: {len(result['sources'])}")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"  {i}. Score: {source['score']:.3f}")
                print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
