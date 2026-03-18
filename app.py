"""
RAG Chatbot - Web Interface using Flask
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from document_parser import parse_file_content

# Load environment variables
load_dotenv()

# Import the RAG chatbot
from rag_chatbot import RAGChatbot

app = Flask(__name__)
CORS(app)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'md'}

# Initialize chatbot
chatbot = RAGChatbot()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint - supports both RAG and general chat modes"""
    try:
        data = request.json
        query = data.get('query', '')
        mode = data.get('mode', 'auto')  # 'rag', 'general', or 'auto'

        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # Determine which mode to use
        use_rag = True
        if mode == 'general':
            use_rag = False
        elif mode == 'auto':
            # Auto mode: try RAG first, fallback to general if no documents
            use_rag = True

        print(f"[INFO] Processing query in {'RAG' if use_rag else 'GENERAL'} mode: {query[:50]}...")
        result = chatbot.chat(query, use_rag=use_rag)
        print(f"[INFO] Response generated successfully")
        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] Chat failed: {str(e)}")
        import traceback
        traceback.print_exc()
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
    print("\nStarting server at http://localhost:5000")
    print("Open your browser and go to: http://localhost:5000\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
