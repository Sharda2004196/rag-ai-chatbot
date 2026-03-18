# RAG AI Chatbot - Session History

## Session 3: 2026-03-18
**Duration**: ~1 hour
**Status**: FULLY FUNCTIONAL - PDF/DOCX Support Added

### Major Updates

**Fixed Critical Issues:**
1. ✅ Fixed Unicode encoding error (replaced ✓ ❌ 📚 with ASCII-safe [OK], [ERROR], [Sources])
2. ✅ Fixed Groq API 400 error - Updated model from `llama-3.1-70b-versatile` (decommissioned) to `llama-3.3-70b-versatile`
3. ✅ Added PDF and DOCX parsing support (PyPDF2 + python-docx)
4. ✅ Added General Chat Mode (can chat without uploading documents)

### New Features Added

**1. Document Parser (`document_parser.py`)**
- Supports PDF, DOCX, TXT, MD files
- Automatic file type detection
- Proper error handling for different encodings
- Page-by-page PDF extraction
- Paragraph-based DOCX extraction

**2. Dual Chat Modes**
- **RAG Mode**: Search uploaded documents for answers
- **General Chat Mode**: Direct LLM conversation without documents
- **Auto Mode**: Tries RAG first, falls back to general if no documents
- Mode selector dropdown in UI

**3. File Upload Enhancement**
- Direct file upload (no need to paste text)
- 16MB max file size limit
- File type validation
- Real-time upload progress

**4. Updated Dependencies**
```
PyPDF2==3.0.1
python-docx==1.1.0
```

### Files Modified
- `rag_chatbot.py` - Fixed Unicode, updated Groq model, added dual-mode chat
- `app.py` - Added file upload handling, document parser integration
- `templates/index.html` - Added mode selector, improved file upload
- `requirements.txt` - Added PDF/DOCX libraries
- `document_parser.py` - NEW FILE for parsing documents

### Current Status: READY TO USE

**To Start:**
```bash
cd C:\Users\hp
C:\Users\hp\AppData\Local\Programs\Python\Python310\python.exe app.py
```

**Features Working:**
- ✅ Upload PDF, DOCX, TXT, MD files
- ✅ Chat about uploaded documents (RAG mode)
- ✅ General chat without documents
- ✅ Source citations
- ✅ Beautiful web interface at http://localhost:5000

---

## Session 2: 2026-03-17
**Duration**: ~2 hours
**Status**: Python Implementation Complete - Pending API Key Configuration

### Major Pivot: Switched from n8n to Python

**Decision**: Abandoned n8n approach due to workflow activation issues
**Reason**:
- n8n workflow couldn't be activated (no visible toggle in UI)
- Webhooks returning 404 errors
- User preferred simpler, more direct approach
- Python gives full control without n8n complexity

### What We Built (Python Edition)

Created a complete Python-based RAG chatbot with web interface:

**Files Created:**
1. **`rag_chatbot.py`** - Core RAG chatbot logic
   - Document ingestion with chunking (1000 chars, 200 overlap)
   - Cohere embeddings integration
   - Qdrant vector database operations
   - Groq LLM integration
   - Search and retrieval functionality

2. **`app.py`** - Flask web server
   - `/api/chat` endpoint for questions
   - `/api/ingest` endpoint for document upload
   - CORS enabled for browser access

3. **`templates/index.html`** - Beautiful web interface
   - Two-panel design (document upload + chat)
   - Drag-and-drop file upload
   - Real-time chat interface
   - Source citation display
   - Modern gradient design

4. **`requirements.txt`** - Python dependencies
   - qdrant-client==1.7.0
   - cohere==4.37
   - requests==2.31.0
   - flask==3.0.0
   - flask-cors==4.0.0
   - python-dotenv==1.0.0

5. **`.env`** - Configuration file (template created)
6. **`.env.example`** - Example configuration
7. **`README.md`** - Complete documentation
8. **`setup.bat`** - Windows setup script

### Technical Setup Completed

✅ **Python 3.10 Identified** - Found at `C:\Users\hp\AppData\Local\Programs\Python\Python310\`
✅ **Dependencies Installed** - All packages installed successfully
✅ **Project Structure Created** - All files in `C:\Users\hp\`
✅ **Documentation Written** - Comprehensive README with instructions

### Python Version Issue Resolved

**Problem**: User had Python 3.14 installed (too new)
**Solution**: Used Python 3.10 which was also installed
**Reason**: qdrant-client doesn't support Python 3.14 yet (max 3.12)

### Architecture

**Technology Stack:**
- **Backend**: Flask (Python web framework)
- **LLM**: Groq API (Llama 3.1 70B)
- **Embeddings**: Cohere (embed-english-v3.0, 1024 dimensions)
- **Vector DB**: Qdrant Cloud (Europe region)
- **Frontend**: Vanilla HTML/CSS/JavaScript

**API Endpoints:**
- `GET /` - Web interface
- `POST /api/ingest` - Upload documents
- `POST /api/chat` - Ask questions

**Features:**
- Document chunking with overlap
- Vector similarity search (top 5 results)
- Context-aware responses with source citations
- Beautiful gradient UI
- Real-time chat interface

---

## Session 1: 2026-03-16
**Duration**: ~2 hours
**Status**: ABANDONED - n8n workflow activation issues

---

## What We Built

### 1. n8n RAG Chatbot Workflow
Created a complete Retrieval-Augmented Generation (RAG) chatbot automation that:
- Ingests documents in multiple formats (PDF, DOCX, TXT, MD, Audio, Video)
- Processes and chunks documents for optimal retrieval
- Stores embeddings in a vector database
- Answers questions using retrieved context from documents
- Cites sources in responses

### 2. Technology Stack (100% Free APIs)
- **LLM**: Groq API (Llama 3.1 70B) - 14,400 requests/day free
- **Embeddings**: Cohere API (embed-english-v3.0) - 100 calls/min free
- **Vector Database**: Qdrant Cloud (Europe region, AWS) - 1GB free tier
- **Orchestration**: n8n workflow automation
- **Transcription**: Groq Whisper API for audio/video files

---

## Files Created

### Workflow Files
1. **`n8n-rag-chatbot-workflow.json`** (C:\Users\hp\)
   - Original workflow with OpenAI + Pinecone (paid services)
   - Not used in final implementation

2. **`n8n-rag-chatbot-free.json`** (C:\Users\hp\)
   - Modified workflow using 100% free APIs
   - Uses Groq, Cohere, and Qdrant
   - **This is the active workflow file**

### Documentation
3. **`RAG-CHATBOT-SETUP.md`** (C:\Users\hp\)
   - Comprehensive setup guide for paid version
   - Includes troubleshooting and customization options

4. **`FREE-RAG-SETUP-GUIDE.md`** (C:\Users\hp\)
   - Setup guide for free API version
   - Step-by-step instructions for Groq, Cohere, Qdrant
   - Usage limits and best practices

### Test Interface
5. **`rag-chatbot-test.html`** (C:\Users\hp\)
   - Beautiful web interface for testing the chatbot
   - Two-panel design: document upload + chat interface
   - Drag-and-drop file upload
   - Real-time status updates
   - Source citation display

---

## Key Decisions Made

### 1. Free vs Paid Services
**Decision**: Use 100% free alternatives instead of paid APIs
**Reason**: User wanted to avoid API costs
**Result**:
- Groq (free) instead of OpenAI ($$$)
- Cohere (free) instead of OpenAI embeddings ($$$)
- Qdrant Cloud (free) instead of Pinecone ($70/month)

### 2. Vector Database Choice
**Decision**: Qdrant Cloud (Europe region)
**Reason**:
- No India region available
- Europe has better latency to India than Americas (~100-150ms)
- Free tier: 1GB storage, sufficient for testing

### 3. Embedding Model
**Decision**: Cohere embed-english-v3.0 (1024 dimensions)
**Reason**: Free tier, good quality, well-supported in n8n
**Configuration**: Qdrant collection created with vector size 1024

### 4. LLM Model
**Decision**: Groq Llama 3.1 70B Versatile
**Reason**:
- Very fast inference
- High quality responses
- Generous free tier (14,400 requests/day)
- Better than running local models

### 5. Workflow Architecture
**Decision**: Two separate webhooks (ingest + chat)
**Reason**:
- Clean separation of concerns
- Easier to debug
- Can scale independently

---

## Configuration Completed

### API Keys Obtained
✅ **Groq API Key** - Configured in n8n
✅ **Cohere API Key** - Configured in n8n
✅ **Qdrant Cloud** - Cluster created, credentials configured

### Qdrant Setup
- **Cloud Provider**: AWS
- **Region**: Europe (eu-central-1)
- **Cluster**: Created and running
- **Collection**: `rag_documents` created
  - Vector size: 1024
  - Distance metric: Cosine
  - Embedding type: Simple Single Embedding
  - Use case: Global Search

### n8n Workflow
- **File**: `n8n-rag-chatbot-free.json` imported
- **Credentials**: All three services configured
- **Nodes**: 16 nodes total
  - 2 webhook endpoints
  - Document processing pipeline
  - Vector storage and retrieval
  - LLM integration

---

## Current Issue: Webhook 404 Error

### Problem
Webhooks returning 404 Not Found when called:
```
curl http://localhost:5678/webhook/ingest → 404
curl http://localhost:5678/webhook/chat → 404
```

### Symptoms
- n8n is running (accessible at localhost:5678)
- Workflow imported successfully
- All credentials configured
- User reports no "Active" toggle or "Save" button visible
- Production URLs copied correctly from webhook nodes

### Attempted Fixes
1. ✅ Added CORS headers to webhook nodes
2. ✅ Started local HTTP server (http-server on port 8000) to avoid file:// CORS issues
3. ✅ Verified n8n is running
4. ❌ Cannot verify workflow activation status (no visible toggle)

### Likely Causes
1. Workflow not activated/saved in n8n
2. n8n version differences (UI may vary)
3. Webhook paths not registered
4. Workflow execution mode issue

---

## What's Pending

### Immediate Next Steps (User Action Required)

1. **Add API Keys to `.env` file**
   - Open `C:\Users\hp\.env` in notepad (already opened)
   - Replace placeholder values with actual API keys:
     - GROQ_API_KEY
     - COHERE_API_KEY
     - QDRANT_URL
     - QDRANT_API_KEY
   - Save the file

2. **Run the Chatbot**
   ```bash
   cd C:\Users\hp
   C:\Users\hp\AppData\Local\Programs\Python\Python310\python.exe app.py
   ```

3. **Access Web Interface**
   - Open browser to: http://localhost:5000
   - Upload documents via the interface
   - Start chatting with your RAG AI!

### Testing Checklist

- [ ] Add API keys to `.env`
- [ ] Start Flask server
- [ ] Access web interface
- [ ] Upload a test document
- [ ] Ask questions about the document
- [ ] Verify source citations work
- [ ] Test with different document types

### Future Enhancements

- Add PDF/DOCX file parsing (currently text only)
- Implement conversation history/memory
- Add user authentication
- Deploy to cloud (Heroku, Railway, Render)
- Add document management (list, delete documents)
- Implement rate limiting
- Add analytics and usage tracking

---

## Key Decisions & Lessons Learned

### Session 2 Decisions

1. **Python over n8n**
   - Simpler to debug and maintain
   - Full control over code
   - No UI activation issues
   - Easier to extend and customize

2. **Flask for Web Server**
   - Lightweight and simple
   - Easy to understand
   - Good for prototyping
   - Can scale to production

3. **Vanilla JavaScript Frontend**
   - No build tools needed
   - Fast development
   - Easy to modify
   - Single HTML file

### Lessons Learned

1. **Python version compatibility matters** - Always check package requirements
2. **n8n can be tricky** - UI varies by version, activation not always obvious
3. **Direct code is often simpler** - Sometimes frameworks add unnecessary complexity
4. **Free APIs are powerful** - Groq, Cohere, Qdrant provide excellent free tiers
5. **Documentation is crucial** - Good README saves time later

---

## Resources & Links

### API Dashboards
- Groq Console: https://console.groq.com/
- Cohere Dashboard: https://dashboard.cohere.com/
- Qdrant Cloud: https://cloud.qdrant.io/

### Documentation
- Flask Docs: https://flask.palletsprojects.com/
- Groq API Docs: https://console.groq.com/docs
- Cohere Docs: https://docs.cohere.com/
- Qdrant Docs: https://qdrant.tech/documentation/

### Local Files
- Project Directory: `C:\Users\hp\`
- Python Path: `C:\Users\hp\AppData\Local\Programs\Python\Python310\python.exe`
- Main Script: `app.py`
- Config File: `.env`

---

## Session Status: READY TO RUN

**Status**: All code complete, dependencies installed
**Blocker**: Waiting for user to add API keys to `.env` file
**Next Action**: User adds API keys, then run `python app.py`
**Expected Result**: Web interface at http://localhost:5000

---

**Last Updated**: 2026-03-17 22:58 IST
