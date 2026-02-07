from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add relevant paths so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "services")) 

from services.retrieving import FactCheckSearcher
from services.llm_service import LLMFactChecker

app = FastAPI(title="ViFactCheck API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # hoặc ["*"] khi dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL SERVICES ---
searcher = None
llm_checker = None

@app.on_event("startup")
async def startup_event():
    global searcher, llm_checker
    try:
        # Initialize Searcher (Qdrant + Embedding Model)
        print("🚀 Starting Search Service...")
        # Running from server/ -> db_path should be "vectordb"
        searcher = FactCheckSearcher(db_path="vectordb")
        
        # Initialize LLM Service (OpenAI)
        print("🧠 Starting LLM Service...")
        llm_checker = LLMFactChecker()
        
        print("✅ System Ready!")
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        # In production, you might want to exit here
        pass

@app.on_event("shutdown")
async def shutdown_event():
    global searcher
    if searcher:
        if hasattr(searcher, 'close'):
             searcher.close()
        print("🛑 Search Service Closed.")

# --- MODELS ---
class CheckRequest(BaseModel):
    claim: str

class CheckResponse(BaseModel):
    claim: str
    status: str
    explanation: str
    confidence: float
    evidence: list

# --- ENDPOINTS ---
@app.get("/")
def read_root():
    return {"message": "Welcome to ViFactCheck API. Use /check to verify news."}

@app.post("/check", response_model=CheckResponse)
async def check_claim(request: CheckRequest):
    global searcher, llm_checker
    
    if not searcher or not llm_checker:
        raise HTTPException(status_code=503, detail="Services not initialized")

    claim = request.claim.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Claim cannot be empty")

    print(f"\n📨 Received Claim: {claim}")

    try:
        # 1. Retrieve Evidence
        # Threshold 0.65 as discussed
        evidence_list = searcher.search(claim, k=3, threshold=0.65)
        
        # 2. Check with LLM
        llm_result = llm_checker.verify_claim(claim, evidence_list)
        
        return {
            "claim": claim,
            "status": llm_result.get("status", "LỖI"),
            "explanation": llm_result.get("explanation", "Không thể phân tích"),
            "confidence": llm_result.get("confidence", 0.0),
            "evidence": evidence_list
        }
    except Exception as e:
        print(f"Error during processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
