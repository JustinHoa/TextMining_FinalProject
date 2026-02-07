# ViFactCheck Server

Backend API service for the ViFactCheck fact-checking application. This server uses FastAPI to provide fact-checking capabilities using vector search and LLM-based verification.

## 🚀 Features

- **Fact-Checking API**: Verify claims using semantic search and LLM analysis
- **Vector Database**: Qdrant-based vector storage for efficient similarity search
- **LLM Integration**: OpenAI-powered claim verification
- **CORS Support**: Configured for frontend integration

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **API Keys**: 
  - OpenAI API key (for LLM verification)
  - Serper API key (for web search, if used)

## 🛠️ Installation & Setup

### Option 1: Automated Setup (Recommended for Windows)

Run the PowerShell setup script which will handle everything automatically:

```powershell
# From the server directory
.\setup.ps1
```

This script will:
1. Create `.env` file from `.env.example`
2. Set up a Python virtual environment
3. Install all dependencies
4. Download and seed the dataset
5. Generate embeddings and populate the vector database

### Option 2: Manual Setup

#### 1. Create Environment File

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

#### 2. Create Virtual Environment

```bash
python -m venv venv
```

#### 3. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 5. Seed the Database

Download and prepare the dataset:

```bash
python scripts/seeding.py
```

#### 6. Generate Embeddings

Create vector embeddings for the dataset (this may take a while):

```bash
python scripts/embedding.py
```

## 🏃 Running the Server

### Development Mode

With auto-reload enabled (recommended for development):

```bash
uvicorn main:app --reload
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Alternative: Direct Python Execution

```bash
python main.py
```

The server will start at: `http://localhost:8000`

## 📡 API Endpoints

### GET `/`
Health check endpoint.

**Response:**
```json
{
  "message": "Welcome to ViFactCheck API. Use /check to verify news."
}
```

### POST `/check`
Verify a claim against the knowledge base.

**Request Body:**
```json
{
  "claim": "Your claim to verify here"
}
```

**Response:**
```json
{
  "claim": "Your claim to verify here",
  "status": "ĐÚNG | SAI | KHÔNG RÕ",
  "explanation": "Detailed explanation of the verification",
  "confidence": 0.85,
  "evidence": [
    {
      "title": "Evidence title",
      "content": "Evidence content",
      "score": 0.92,
      "url": "source_url"
    }
  ]
}
```

## 🧪 Testing the API

Use the provided test script:

```bash
python test_api.py
```

Or use curl:

```bash
curl -X POST "http://localhost:8000/check" \
  -H "Content-Type: application/json" \
  -d "{\"claim\": \"Your claim here\"}"
```

## 📁 Project Structure

```
server/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── setup.ps1              # Automated setup script
├── .env.example           # Environment variables template
├── test_api.py            # API testing script
├── services/              # Core services
│   ├── retrieving.py      # Vector search service
│   ├── llm_service.py     # LLM verification service
│   └── scraper.py         # Web scraping service
├── scripts/               # Utility scripts
│   ├── seeding.py         # Database seeding
│   └── embedding.py       # Vector embedding generation
├── data/                  # Dataset storage
├── vectordb/              # Qdrant vector database
└── docs/                  # Additional documentation
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for LLM verification | Yes |
| `SERPER_API_KEY` | Serper API key for web search | Optional |

### Search Parameters

You can adjust search parameters in `main.py`:

```python
evidence_list = searcher.search(
    claim, 
    k=3,           # Number of results to retrieve
    threshold=0.65 # Similarity threshold (0.0 - 1.0)
)
```

## 🔧 Troubleshooting

### Virtual Environment Issues

If the virtual environment doesn't activate:
- **Windows**: Run PowerShell as Administrator and execute: `Set-ExecutionPolicy RemoteSigned`
- Alternatively, use Command Prompt instead of PowerShell

### Module Import Errors

Make sure you're in the virtual environment and all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

### Vector Database Errors

If Qdrant throws errors, try regenerating the embeddings:
```bash
# Delete the vectordb folder
rm -rf vectordb

# Regenerate embeddings
python scripts/embedding.py
```

### API Key Errors

Verify your `.env` file:
- Check that API keys are set correctly
- Ensure no extra spaces or quotes around keys
- Restart the server after modifying `.env`

## 📝 Development Notes

- The server uses **BGE-M3** model for embeddings
- Default port is **8000**
- CORS is configured for `http://localhost:3000` (frontend)
- Search threshold of **0.65** is used for evidence retrieval

## 🤝 Contributing

When making changes:
1. Update dependencies in `requirements.txt` if needed
2. Test endpoints with `test_api.py`
3. Update this README if adding new features or endpoints

## 📄 License

This project is part of a Text Mining Application assignment.
