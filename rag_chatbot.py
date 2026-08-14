"""
RAG AI Chatbot - Python Implementation
Uses: Groq (LLM), Cohere (Embeddings), Pinecone (Vector DB), Upstash Redis (Memory), faster-whisper (Voice STT)
"""

import os
import re
import io
import wave
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()
from pinecone import Pinecone, ServerlessSpec
import cohere
import requests
import base64
from PIL import Image

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL", "")
UPSTASH_REDIS_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN", "")

INDEX_NAME = "rag-documents"
EMBEDDING_SIZE = 1536  # Cohere embed-v4.0 (supports images and text)
REDIS_SESSION_TTL = 86400  # 24 hours in seconds

class RAGChatbot:
    def __init__(self):
        # Initialize clients lazily (only when needed)
        self._cohere_client = None
        self._pinecone_client = None
        self._pinecone_index = None
        self._redis_client = None
        self._stt_model = None

    @property
    def redis_client(self):
        """Initialize Redis client lazily"""
        if self._redis_client is None and UPSTASH_REDIS_URL and UPSTASH_REDIS_TOKEN:
            try:
                import redis
                # Convert https:// to rediss:// for Upstash
                redis_url = UPSTASH_REDIS_URL.replace('https://', 'rediss://', 1)
                self._redis_client = redis.from_url(
                    redis_url,
                    password=UPSTASH_REDIS_TOKEN,
                    decode_responses=True
                )
                self._redis_client.ping()
                print("[OK] Redis connected")
            except Exception as e:
                print(f"[WARNING] Redis not available: {e}")
                self._redis_client = None
        return self._redis_client

    @property
    def cohere_client(self):
        if self._cohere_client is None:
            self._cohere_client = cohere.Client(COHERE_API_KEY)
        return self._cohere_client

    @property
    def pinecone_client(self):
        if self._pinecone_client is None:
            print(f"[DEBUG] PINECONE_API_KEY length: {len(PINECONE_API_KEY)}")
            print(f"[DEBUG] PINECONE_API_KEY prefix: {PINECONE_API_KEY[:10]}...")
            self._pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
            print("[DEBUG] Pinecone client created")
            self._ensure_index()
        return self._pinecone_client

    @property
    def pinecone_index(self):
        if self._pinecone_index is None:
            self._pinecone_index = self.pinecone_client.Index(INDEX_NAME)
        return self._pinecone_index

    def _ensure_index(self):
        """Create Pinecone index if it doesn't exist"""
        try:
            print("[DEBUG] Listing indexes...")
            existing_indexes = self.pinecone_client.list_indexes()
            print(f"[DEBUG] Existing indexes: {existing_indexes}")
            index_names = [idx.name for idx in existing_indexes]
            print(f"[DEBUG] Index names: {index_names}")
            
            if INDEX_NAME not in index_names:
                print(f"[DEBUG] Creating index {INDEX_NAME} with dimension {EMBEDDING_SIZE}")
                self.pinecone_client.create_index(
                    name=INDEX_NAME,
                    dimension=EMBEDDING_SIZE,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"[OK] Created index: {INDEX_NAME} (dim={EMBEDDING_SIZE})")
            else:
                # Check if index is ready (Pinecone 9.x returns dict)
                print(f"[DEBUG] Index {INDEX_NAME} exists, checking status...")
                index_desc = self.pinecone_client.describe_index(INDEX_NAME)
                print(f"[DEBUG] Index desc: {index_desc}")
                if isinstance(index_desc, dict):
                    status = index_desc.get('status', {})
                    ready = status.get('ready', False) if isinstance(status, dict) else False
                else:
                    ready = getattr(index_desc, 'status', {}).get('ready', False) if hasattr(index_desc, 'status') else False
                    
                if ready:
                    print(f"[OK] Index {INDEX_NAME} is ready")
                else:
                    print(f"[INFO] Index {INDEX_NAME} is not ready yet, waiting...")
                    import time
                    for _ in range(30):
                        time.sleep(1)
                        index_desc = self.pinecone_client.describe_index(INDEX_NAME)
                        if isinstance(index_desc, dict):
                            status = index_desc.get('status', {})
                            ready = status.get('ready', False) if isinstance(status, dict) else False
                        else:
                            ready = getattr(index_desc, 'status', {}).get('ready', False) if hasattr(index_desc, 'status') else False
                        if ready:
                            break
                    print(f"[OK] Index {INDEX_NAME} is now ready")
        except Exception as e:
            import traceback
            print(f"[ERROR] Index check/creation failed: {e}")
            traceback.print_exc()

    def clear_documents(self):
        """Clear all documents from the index"""
        try:
            # Get index stats first to check if any vectors exist
            try:
                stats = self.pinecone_index.describe_index_stats()
                total_vectors = stats.get('total_vector_count', 0)
                if total_vectors > 0:
                    # Delete all vectors from the index (v9.x API)
                    self.pinecone_index.delete(delete_all=True)
                    print(f"[OK] Cleared all documents from index: {INDEX_NAME}")
                else:
                    print("[INFO] No documents to clear")
            except Exception as stats_err:
                print(f"[DEBUG] Could not get stats: {stats_err}")
                # Try to delete anyway
                self.pinecone_index.delete(delete_all=True)
                print(f"[OK] Cleared all documents from index: {INDEX_NAME}")
        except Exception as e:
            print(f"[WARNING] Failed to clear documents: {e}")

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for DOCUMENTS using Cohere embed-v4.0.
        Uses input_type='search_document' (asymmetric embeddings require
        documents and queries to be embedded differently)."""
        response = self.cohere_client.embed(
            texts=texts,
            model="embed-v4.0",
            input_type="search_document"
        )
        return response.embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate an embedding for a USER QUERY.
        Must use input_type='search_query' — Cohere embed-v4.0 is
        asymmetric, so a query embedded as a document retrieves poorly."""
        response = self.cohere_client.embed(
            texts=[query],
            model="embed-v4.0",
            input_type="search_query"
        )
        return response.embeddings[0]

    def analyze_image(self, image_path: str) -> str:
        """Analyze image content using Groq multimodal (Llama-4)"""
        try:
            import base64
            with open(image_path, 'rb') as img_file:
                img_bytes = img_file.read()
            
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            image_type = 'image/jpeg' if image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg') else 'image/png'
            
            # Use Groq multimodal API
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in 2-3 sentences. What is in the image? What objects, text, or scenes can you see?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_type};base64,{base64_image}"
                            }
                        }
                    ]
                }],
                "max_tokens": 300
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            if response.status_code != 200:
                raise Exception(f"Groq API Error {response.status_code}: {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[WARNING] Image analysis failed: {e}")
            return f"Image: {image_path}"

    def embed_image(self, image_path: str) -> List[float]:
        """Generate embedding for an image using Cohere visual model"""
        try:
            import requests as req_lib
            print(f"[DEBUG] embed_image called with path: {repr(image_path)}")
            with open(image_path, 'rb') as img_file:
                img_bytes = img_file.read()
            
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            image_type = 'image/jpeg' if image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg') else 'image/png'
            image_data_uri = f"data:{image_type};base64,{base64_image}"
            
            # Use Cohere v2 API directly
            url = "https://api.cohere.com/v2/embed"
            headers = {
                "Authorization": f"Bearer {COHERE_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "embed-v4.0",
                "input_type": "image",
                "images": [image_data_uri]
            }
            
            response = req_lib.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code != 200:
                raise Exception(f"Cohere API Error {response.status_code}: {response.text}")
            
            result = response.json()
            # Cohere v2 returns embeddings directly
            if "embeddings" in result:
                emb = result["embeddings"]
                if isinstance(emb, dict):
                    # Try float first, then take first available
                    if "float" in emb:
                        return emb["float"][0]
                    elif emb:
                        return list(emb.values())[0][0]
                elif isinstance(emb, list):
                    return emb[0]
            raise Exception("No embeddings returned from Cohere")
        except Exception as e:
            raise Exception(f"Failed to embed image: {str(e)}")

    def ingest_document(self, text: str, metadata: Dict = None, content_type: str = "text", image_path: str = None, clear_existing: bool = False) -> int:
        """Ingest a document into the vector database

        Args:
            text: Text content to ingest
            metadata: Optional metadata
            content_type: Type of content ("text", "url", "image")
            image_path: Optional path to image file (for image content type)
            clear_existing: If True, wipe the index before ingesting.
                            Defaults to False so multiple documents can
                            coexist in the knowledge base.
        """
        print(f"[INFO] ingest_document: text_len={len(text)}, content_type={repr(content_type)}, image_path={repr(image_path)}, clear_existing={clear_existing}")

        if clear_existing:
            print("[INFO] Clearing old documents...")
            self.clear_documents()

        if content_type == "image" and image_path:
            # For images, embed the image directly without chunking
            print("[INFO] Analyzing and embedding image...")
            
            # Analyze image content
            image_description = self.analyze_image(image_path)
            print(f"[INFO] Image analysis: {image_description[:100]}...")
            
            # Embed the image
            embedding = self.embed_image(image_path)
            
            # Use the image description as the text content
            text_content = image_description
            
            # Prepare metadata for Pinecone
            record_metadata = metadata or {}
            record_metadata.update({
                "text": text_content,
                "chunk_index": "0",
                "content_type": "image"
            })

            # Generate unique ID
            record_id = f"doc_{abs(hash(text_content)) % (10 ** 8)}"
            
            # Upsert to Pinecone (legacy API for custom vectors)
            self.pinecone_index.upsert(
                vectors=[{
                    "id": record_id,
                    "values": embedding,
                    "metadata": record_metadata
                }]
            )
            
            print(f"[OK] Ingested 1 image chunk with analysis")
            return 1
        else:
            # For text/URL, use chunking
            chunks = self._chunk_text(text, chunk_size=1000, overlap=200)

            # Generate embeddings based on content type
            if content_type == "image":
                embeddings = [self.embed_image(chunk) for chunk in chunks]
            else:
                embeddings = self.embed_text(chunks)

            # Prepare records for Pinecone
            records = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                record_metadata = metadata or {}
                record_metadata.update({
                    "text": chunk,
                    "chunk_index": str(idx),
                    "content_type": content_type
                })

                record_id = f"doc_{abs(hash(chunk)) % (10 ** 8)}"
                
                records.append({
                    "id": record_id,
                    "values": embedding,
                    "metadata": record_metadata
                })

            # Upload to Pinecone (legacy API for custom vectors)
            self.pinecone_index.upsert(
                vectors=records
            )

            print(f"[OK] Ingested {len(chunks)} chunks")
            return len(chunks)

    def fetch_url_content(self, url: str) -> str:
        """Fetch content from a URL using BeautifulSoup (with trafilatura fallback)"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Fetch the URL
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            import re
            text = re.sub(r'\s+', ' ', text)
            
            if not text or len(text) < 100:
                raise Exception("Extracted content too short, trying trafilatura fallback...")
            
            return text.strip()
            
        except ImportError:
            raise Exception("BeautifulSoup not installed. Run: pip install beautifulsoup4 requests")
        except Exception as e:
            # Fallback to trafilatura if BeautifulSoup fails
            try:
                import trafilatura
                downloaded = trafilatura.fetch_url(url)
                if not downloaded:
                    raise Exception("Failed to fetch URL content")
                
                text = trafilatura.extract(downloaded, include_tables=True, include_links=True)
                if not text:
                    raise Exception("Failed to extract text from URL")
                
                return text.strip()
            except ImportError:
                raise Exception(f"Failed to fetch URL: {str(e)}. Install trafilatura as backup: pip install trafilatura")
            except Exception as e2:
                raise Exception(f"Failed to fetch URL: {str(e2)}")

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into semantic chunks.

        Unlike naive character-slicing, this respects paragraph and line
        boundaries so section headers (e.g. "Experience") stay with their
        content. Falls back to character-slicing only for very long
        unbroken blocks.
        """
        # Normalize line endings and split into logical units (paragraphs).
        # Blank-line-separated paragraphs are the primary boundary; single
        # newlines (bullet lists, resume lines) are kept together.
        normalized = re.sub(r'\r\n|\r', '\n', text)
        paragraphs = [p.strip() for p in normalized.split('\n\n') if p.strip()]

        # If there are no blank-line paragraphs, fall back to line groups.
        if not paragraphs:
            lines = [l.strip() for l in normalized.split('\n') if l.strip()]
            groups = []
            current = []
            current_len = 0
            for line in lines:
                if current_len + len(line) > chunk_size and current:
                    groups.append("\n".join(current))
                    current = []
                    current_len = 0
                current.append(line)
                current_len += len(line) + 1
            if current:
                groups.append("\n".join(current))
            paragraphs = groups if groups else [normalized]

        # Merge paragraphs into chunks bounded by chunk_size.
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            # A single paragraph larger than chunk_size: hard-slice it.
            if len(para) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunks.append(para[start:end].strip())
                    start = end - overlap
            else:
                if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Drop empty chunks
        return [c for c in chunks if c.strip()]

    def search(self, query: str, top_k: int = 5, min_score: float = 0.3) -> List[Dict]:
        """Search for relevant documents.

        Uses embed_query (asymmetric query embedding) and filters out
        results below min_score so irrelevant chunks are never passed to
        the LLM. Cosine similarity for Cohere embed-v4.0 typically sits in
        the 0.3-0.9 range for relevant matches.
        """
        # Generate query embedding with the correct input type
        query_embedding = self.embed_query(query)

        # Search in Pinecone (legacy API for custom vectors)
        results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=top_k * 2,  # fetch extra, then filter by score
            include_metadata=True,
            include_values=False
        ).matches

        # Filter by relevance threshold
        filtered = [
            hit for hit in results
            if hit.score is not None and hit.score >= min_score
        ][:top_k]

        # Debug: Show what's in the database
        print(f"[DEBUG] Search returned {len(results)} raw, {len(filtered)} above threshold {min_score}")
        for i, r in enumerate(filtered[:3]):
            text_preview = r.metadata.get('text', 'N/A')[:80] if r.metadata else 'N/A'
            print(f"  Result {i+1}: (score={r.score:.3f}) {text_preview}...")

        return [
            {
                "text": hit.metadata.get("text", "") if hit.metadata else "",
                "score": hit.score,
                "metadata": {k: v for k, v in (hit.metadata or {}).items() if k != "text"}
            }
            for hit in filtered
        ]

    def _get_session_history(self, session_id: str) -> List[Dict]:
        """Get conversation history from Redis"""
        if not self.redis_client:
            return []
        
        try:
            history_json = self.redis_client.get(f"chat_history:{session_id}")
            if history_json:
                return json.loads(history_json)
            return []
        except Exception as e:
            print(f"[WARNING] Failed to get session history: {e}")
            return []

    def _save_session_history(self, session_id: str, history: List[Dict]):
        """Save conversation history to Redis"""
        print(f"[DEBUG] _save_session_history: redis_client={self.redis_client is not None}, session_id={session_id}")
        if not self.redis_client:
            print("[WARNING] Redis client not available, skipping history save")
            return
        
        try:
            self.redis_client.setex(
                f"chat_history:{session_id}",
                REDIS_SESSION_TTL,
                json.dumps(history)
            )
            print(f"[OK] Saved history for session {session_id}")
        except Exception as e:
            print(f"[WARNING] Failed to save session history: {e}")

    def chat(self, query: str, use_rag: bool = True, session_id: str = "default", include_history: bool = True) -> Dict:
        """Chat with RAG - retrieve context and generate response

        Args:
            query: User's question
            use_rag: If True, use RAG mode (search documents). If False, use general chat mode.
            session_id: Unique identifier for chat session
            include_history: Whether to include conversation history in context
        """
        # Simple, flexible prompt that follows user's preferred format
        
        # General chat mode (no document retrieval)
        if not use_rag:
            history = []
            if include_history:
                history = self._get_session_history(session_id)
            
            history_text = ""
            if history:
                history_text = "\n\nCONVERSATION HISTORY:\n" + "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in history[-6:]
                ])
            
            # Clean, structured prompt
            prompt = f"""You are a helpful AI assistant. Provide clear, well-organized answers.
Use bullet points or numbered lists. Be concise but complete.

USER QUESTION: {query}{history_text}

Answer:"""
            
            response = self._call_groq(prompt)
            
            # Save to history
            if include_history and self.redis_client:
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": response})
                self._save_session_history(session_id, history)
            
            return {
                "answer": response,
                "sources": [],
                "mode": "general",
                "session_id": session_id
            }

        # RAG mode (with document retrieval)
        history = []
        if include_history:
            history = self._get_session_history(session_id)
        
        # Retrieve relevant documents
        search_results = self.search(query, top_k=5)

        # If no docs found or very low relevance, fallback to General mode
        if not search_results:
            print("[INFO] No RAG results, falling back to GENERAL mode")
            history_text = ""
            if history:
                history_text = "\n\nCONVERSATION HISTORY:\n" + "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in history[-6:]
                ])
                
            prompt = f"""You are a helpful AI assistant. Provide clear, well-organized answers.
Use bullet points or numbered lists. Be concise but complete.

{history_text}

Question: {query}

Answer:"""
            
            response = self._call_groq(prompt)
                
            # Save to history
            if include_history and self.redis_client:
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": response})
                self._save_session_history(session_id, history)
                
            return {
                "answer": response,
                "sources": [],
                "mode": "general",
                "session_id": session_id
            }
        
        # Build context from search results
        context = "\n\n".join([
            f"[Source {i+1}]\n{result['text']}"
            for i, result in enumerate(search_results)
        ])

        # Build history context (kept separate from retrieved docs)
        history_text = ""
        if history:
            history_text = "\n\nCONVERSATION HISTORY:\n" + "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in history[-6:]
            ])

        # Strict RAG prompt: answer ONLY from retrieved sources, no citations.
        prompt = f"""You are an AI assistant that answers questions using ONLY the provided retrieved document excerpts. The conversation history is for context only and must NOT be used as a source of facts.

Rules:
- Answer ONLY from the [Source N] excerpts below.
- Do NOT mention sources, citations, "Source 1", or brackets in your answer.
- If the excerpts do not contain the answer, say "The uploaded documents do not contain this information." Do not guess or use general knowledge.
- Be concise, use bullet points where helpful.

RETRIEVED DOCUMENTS:
{context}
{history_text}

Question: {query}

Answer:"""

        # Call Groq API
        response = self._call_groq(prompt)
        
        # Save to history
        if include_history and self.redis_client:
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})
            self._save_session_history(session_id, history)
        
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
            "mode": "rag",
            "session_id": session_id
        }

    def analyze_question_complexity(self, query: str) -> str:
        """Analyze question complexity to determine response format
        
        Returns: 'simple' or 'complex'
        """
        # Simple question indicators
        simple_indicators = [
            r'\b(what is|what are|who is|who are|when is|when are|where is|where are|how is|how are)\b',
            r'\b(is|are|can|does|do|did|has|have|had|will|would|could|should|may|might)\b',
            r'\b(yes|no)\b',
            r'^[A-Z][^.!?]{0,30}[\?\.]$',
        ]
        
        # Complex question indicators
        complex_indicators = [
            r'\b(explain|describe|analyze|compare|contrast|discuss|evaluate|assess)\b',
            r'\b(why|how does|how do|what causes|what leads to)\b',
            r'\b(steps|process|method|approach|strategy|technique)\b',
            r'\b(provide|detailed|comprehensive|thorough|in-depth)\b',
        ]
        
        query_lower = query.lower()
        
        # Check for complex indicators first
        for pattern in complex_indicators:
            if re.search(pattern, query_lower):
                return 'complex'
        
        # Check for simple indicators
        for pattern in simple_indicators:
            if re.search(pattern, query_lower):
                # Exception: if question is very short (under 10 words), it's simple
                word_count = len(query.split())
                if word_count <= 15:
                    return 'simple'
        
        # Default: if question is short, treat as simple
        word_count = len(query.split())
        if word_count <= 10:
            return 'simple'
        
        return 'complex'

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API for LLM response with JSON structured output"""
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

            if response.status_code != 200:
                print(f"[ERROR] Groq API returned {response.status_code}")
                print(f"Response: {response.text}")
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

    def _call_groq_stream(self, prompt: str):
        """Call Groq API in streaming mode. Yields text deltas as they arrive."""
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
            "max_tokens": 2000,
            "stream": True
        }

        try:
            response = requests.post(url, headers=headers, json=data, stream=True, timeout=60)

            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', response.text)
                    raise Exception(f"Groq API Error: {error_msg}")
                except:
                    raise Exception(f"Groq API Error {response.status_code}: {response.text}")

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = line[len("data: "):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

        except requests.exceptions.Timeout:
            raise Exception("Groq API request timed out after 60 seconds")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Groq API request failed: {str(e)}")

    def stream_chat(self, query: str, use_rag: bool = True, session_id: str = "default", include_history: bool = True):
        """Stream a chat response. Yields event dicts:
        {"type": "text", "content": "..."} for each token delta, then
        {"type": "done", "answer": "...", "mode": "..."} at the end.
        Supports RAG, general, and rag_no_docs modes like chat()."""
        # ---- General chat mode ----
        if not use_rag:
            history = []
            if include_history:
                history = self._get_session_history(session_id)

            history_text = ""
            if history:
                history_text = "\n\nCONVERSATION HISTORY:\n" + "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in history[-6:]
                ])

            prompt = f"""You are a helpful AI assistant. Provide clear, well-organized answers.
Use bullet points or numbered lists. Be concise but complete.

USER QUESTION: {query}{history_text}

Answer:"""

            full = ""
            for delta in self._call_groq_stream(prompt):
                full += delta
                yield {"type": "text", "content": delta}

            if include_history and self.redis_client:
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": full})
                self._save_session_history(session_id, history)

            yield {"type": "done", "answer": full, "mode": "general"}
            return

        # ---- RAG mode ----
        history = []
        if include_history:
            history = self._get_session_history(session_id)

        search_results = self.search(query, top_k=5)

        if not search_results:
            if history:
                history_text = "\n\nCONVERSATION HISTORY:\n" + "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in history[-6:]
                ])

                prompt = f"""You are a helpful AI assistant. Provide clear, well-organized answers.
Use bullet points or numbered lists. Be concise but complete.

{history_text}

Question: {query}

Answer:"""

                full = ""
                for delta in self._call_groq_stream(prompt):
                    full += delta
                    yield {"type": "text", "content": delta}

                if self.redis_client:
                    history.append({"role": "user", "content": query})
                    history.append({"role": "assistant", "content": full})
                    self._save_session_history(session_id, history)

                yield {"type": "done", "answer": full, "mode": "rag_no_docs"}
                return

            msg = "I don't have enough information to answer that question. Try uploading some documents first, or switch to general chat mode."
            yield {"type": "text", "content": msg}
            yield {"type": "done", "answer": msg, "mode": "rag"}
            return

        # Build context from search results
        context = "\n\n".join([
            f"[Source {i+1}]\n{result['text']}"
            for i, result in enumerate(search_results)
        ])

        history_text = ""
        if history:
            history_text = "\n\nCONVERSATION HISTORY:\n" + "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in history[-6:]
            ])

        prompt = f"""You are an AI assistant that answers questions using ONLY the provided retrieved document excerpts. The conversation history is for context only and must NOT be used as a source of facts.

Rules:
- Answer ONLY from the [Source N] excerpts below.
- Do NOT mention sources, citations, "Source 1", or brackets in your answer.
- If the excerpts do not contain the answer, say "The uploaded documents do not contain this information." Do not guess or use general knowledge.
- Be concise, use bullet points where helpful.

RETRIEVED DOCUMENTS:
{context}
{history_text}

Question: {query}

Answer:"""

        full = ""
        for delta in self._call_groq_stream(prompt):
            full += delta
            yield {"type": "text", "content": delta}

        if include_history and self.redis_client:
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": full})
            self._save_session_history(session_id, history)

        yield {"type": "done", "answer": full, "mode": "rag"}

    def _get_stt_model(self):
        """Lazy-load the faster-whisper model (base.en = fast, accurate, CPU-friendly)"""
        if self._stt_model is None:
            try:
                from faster_whisper import WhisperModel
                print("[OK] Loading faster-whisper base.en model...")
                self._stt_model = WhisperModel("base.en", device="cpu", compute_type="int8")
                print("[OK] Whisper model loaded")
            except Exception as e:
                print(f"[ERROR] Failed to load Whisper model: {e}")
                raise
        return self._stt_model

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes (WAV/MP3/OGG/WebM) to text using faster-whisper."""
        import tempfile
        import os as _os

        model = self._get_stt_model()

        suffix = ".wav"
        if isinstance(audio_bytes, bytes):
            pass

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            segments, info = model.transcribe(tmp_path, beam_size=5, vad_filter=True)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text
        finally:
            try:
                _os.unlink(tmp_path)
            except OSError:
                pass


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
    if not PINECONE_API_KEY:
        print("[ERROR] PINECONE_API_KEY not set!")
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
