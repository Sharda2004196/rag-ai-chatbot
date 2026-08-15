"""
RAG Chatbot - Web Interface using Flask
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import os
import json
from dotenv import load_dotenv
from document_parser import parse_file_content
import uuid

# Load environment variables
load_dotenv()

# Import the RAG chatbot
from rag_chatbot import RAGChatbot

app = Flask(__name__, template_folder='templates')
CORS(app)

# Disable caching
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'md'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize chatbot
chatbot = RAGChatbot()

def allowed_file(filename, image_allowed=False):
    ext = filename.rsplit('.', 1)[1].lower()
    if image_allowed:
        return ext in ALLOWED_IMAGE_EXTENSIONS
    return '.' in filename and ext in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['POST'])
def send_message():
    """Handle form-based chat submission"""
    try:
        message = request.form.get('message', '')
        mode = request.form.get('mode', 'auto')
        
        if not message:
            return render_template('index.html', error="No message provided")
        
        # Get or create session
        session_id = request.cookies.get('session_id', 'default')
        
        # Determine mode
        use_rag = True
        if mode == 'general':
            use_rag = False
        
        # Get chat history from Redis
        history = []
        if chatbot.redis_client:
            import json
            history_json = chatbot.redis_client.get(f"chat_history:{session_id}")
            if history_json:
                history = json.loads(history_json)
        
        # Add user message to history
        from datetime import datetime
        history.append({'role': 'user', 'content': message, 'timestamp': datetime.now().isoformat()})
        
        # Get bot response
        result = chatbot.chat(message, use_rag=use_rag, session_id=session_id, include_history=False)
        
        # Add bot response to history
        history.append({'role': 'assistant', 'content': result.get('response', ''), 'timestamp': datetime.now().isoformat()})
        
        # Save history to Redis
        if chatbot.redis_client:
            import json
            chatbot.redis_client.set(f"chat_history:{session_id}", json.dumps(history[-20:]))  # Keep last 20 messages
        
        # Render page with messages
        return render_template('index.html', messages=history[-10:], mode=mode)
        
    except Exception as e:
        print(f"[ERROR] Form send failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=str(e))

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint - supports both RAG and general chat modes"""
    try:
        data = request.json
        query = data.get('query', '')
        mode = data.get('mode', 'auto')  # 'rag', 'general', or 'auto'
        session_id = data.get('session_id', 'default')

        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # Determine which mode to use
        use_rag = True
        if mode == 'general':
            use_rag = False
            print("[INFO] Mode: GENERAL")
        else:
            # RAG mode (default)
            use_rag = True
            print("[INFO] Mode: RAG")

        print(f"[INFO] Processing query in {'RAG' if use_rag else 'GENERAL'} mode: {query[:50]}...")
        result = chatbot.chat(query, use_rag=use_rag, session_id=session_id, include_history=True)
        print(f"[INFO] Response generated successfully")
        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] Chat failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint (Server-Sent Events).
    Yields events: data: {"type":"text","content":"..."} ... then
    data: {"type":"done","answer":"...","mode":"..."}"""
    try:
        data = request.json or {}
        query = data.get('query', '')
        mode = data.get('mode', 'auto')
        session_id = data.get('session_id', 'default')

        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # Determine mode
        if mode == 'general':
            use_rag = False
        else:
            use_rag = True

        def generate():
            try:
                for event in chatbot.stream_chat(query, use_rag=use_rag, session_id=session_id, include_history=True):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                print(f"[ERROR] Streaming chat failed: {str(e)}")
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        print(f"[ERROR] Streaming chat setup failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """Transcribe audio file to text using faster-whisper (local, offline STT)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        f = request.files['file']
        if not f or not f.filename:
            return jsonify({'error': 'No file provided'}), 400

        audio_bytes = f.read()
        if not audio_bytes:
            return jsonify({'error': 'Empty audio file'}), 400

        text = chatbot.transcribe_audio(audio_bytes)
        return jsonify({'text': text})

    except Exception as e:
        print(f"[ERROR] Transcription failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/fetch-url', methods=['POST'])
def fetch_url():
    """Fetch content from a URL and ingest it"""
    try:
        data = request.json
        url = data.get('url', '')

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        print(f"[INFO] Fetching URL: {url}")
        content = chatbot.fetch_url_content(url)
        
        metadata = {
            'source': 'url',
            'url': url,
            'content_type': 'url'
        }
        
        chunks = chatbot.ingest_document(content, metadata, content_type="url")
        
        return jsonify({
            'success': True,
            'chunks': chunks,
            'message': f'URL fetched and ingested ({chunks} chunks)',
            'content_type': 'url'
        })

    except Exception as e:
        print(f"[ERROR] URL fetch failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """Upload and ingest an image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if not allowed_file(image_file.filename, image_allowed=True):
            return jsonify({'error': f'Unsupported image type. Allowed: {", ".join(ALLOWED_IMAGE_EXTENSIONS)}'}), 400
        
        # Save the image temporarily
        filename = f"img_{uuid.uuid4().hex[:8]}.{image_file.filename.rsplit('.', 1)[1].lower()}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(filepath)
        
        print(f"[INFO] Processing image: {filename}")
        print(f"[DEBUG] filepath = {repr(filepath)}")
        
        # Generate embedding for the image
        try:
            image_embedding = chatbot.embed_image(filepath)
            
            # Create a simple text description for the image
            image_metadata = {
                'source': 'image_upload',
                'filename': filename,
                'content_type': 'image'
            }
            
            # Ingest with minimal text (just use the filename as chunk)
            print(f"[DEBUG] Calling ingest_document")
            print(f"[DEBUG]   text = {repr(f'Image: {filename}')}")
            print(f"[DEBUG]   metadata = {image_metadata}")
            print(f"[DEBUG]   content_type = {repr('image')}")
            print(f"[DEBUG]   image_path = {repr(filepath)}")
            chunks = chatbot.ingest_document(
                f"Image: {filename}",
                image_metadata,
                content_type="image",
                image_path=filepath
            )
            
            # Clean up temp file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'success': True,
                'chunks': chunks,
                'message': f'Image analyzed and ingested ({chunks} chunks)',
                'content_type': 'image',
                'filename': filename
            })
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e

    except Exception as e:
        print(f"[ERROR] Image upload failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history for a session"""
    try:
        session_id = request.args.get('session_id', 'default')
        history = chatbot._get_session_history(session_id)
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear conversation history for a session"""
    try:
        data = request.json or {}
        session_id = data.get('session_id', request.args.get('session_id', 'default'))
        
        if chatbot.redis_client:
            chatbot.redis_client.delete(f"chat_history:{session_id}")
        
        return jsonify({'success': True, 'message': 'History cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-documents', methods=['POST'])
def clear_documents():
    """Wipe all vectors from the Pinecone index"""
    try:
        chatbot.clear_documents()
        return jsonify({'success': True, 'message': 'All documents cleared from index'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ingest', methods=['POST'])
def ingest():
    """Document ingestion endpoint - supports both file upload and text"""
    try:
        # Check if it's a file upload or text
        if 'file' in request.files:
            file = request.files['file']

            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            if not allowed_file(file.filename):
                return jsonify({'error': f'Unsupported file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

            # Parse the file
            print(f"[INFO] Parsing file: {file.filename}")
            file_content = file.read()
            text = parse_file_content(file_content, file.filename)

            metadata = {
                'source': 'file_upload',
                'filename': file.filename
            }
        else:
            # Text-based ingestion
            data = request.json
            text = data.get('text', '')
            metadata = data.get('metadata', {})

            if not text:
                return jsonify({'error': 'No text provided'}), 400

        # Ingest the document
        print(f"[INFO] Ingesting document ({len(text)} characters)")
        chunks = chatbot.ingest_document(text, metadata)

        return jsonify({
            'success': True,
            'chunks': chunks,
            'message': f'Ingested {chunks} chunks'
        })

    except Exception as e:
        print(f"[ERROR] Ingestion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("RAG AI Chatbot - Web Interface")
    print("=" * 60)

    # Get port from environment variable (for Render) or use 5000 for local
    port = int(os.environ.get('PORT', 5000))
    print(f"\nStarting server on port {port}")
    print(f"Open your browser and go to: http://localhost:{port}\n")

    app.run(debug=False, host='0.0.0.0', port=port)
