from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import uuid
import hashlib
import re
import sys
import os

# Add parent directory to path to allow importing modules from sibling directories if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Internet search class
try:
    # Try importing from local directory first, then services/scripts if needed
    from internet_search import InternetSearcher
except ImportError:
    try:
        from services.internet_search import InternetSearcher
    except ImportError:
        print("⚠️ Warning: Could not import InternetSearcher. Internet search capabilities will be disabled.")
        InternetSearcher = None

# --- MAIN CLASS ---
class FactCheckSearcher:
    def __init__(self, db_path="../vectordb", collection_name="vifactcheck_chunks"):
        print(f"-> Connecting to Local DB at: {db_path}")
        self.client = QdrantClient(path=db_path)
        self.collection_name = collection_name
        
        print("-> Loading model BGE-M3...")
        self.model = SentenceTransformer('BAAI/bge-m3')
        
        # Initialize Internet Searcher
        if InternetSearcher:
            self.internet_searcher = InternetSearcher()
        else:
            self.internet_searcher = None

    def _search_local(self, query_vec, k=3):
        """Search in DB with Deduplication logic (Filter duplicate articles)"""
        try:
            # Get extra results to filter duplicates (3x k)
            search_limit = k * 3
            # Method for qdrant-client v1.7+
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec.tolist(),
                limit=search_limit
            ).points

            results = []
            seen_article_ids = set()
            
            for hit in search_result:
                art_id = hit.payload.get('article_id')
                
                # Deduplication logic: Only take the chunk with the highest score of each article
                if art_id and art_id in seen_article_ids:
                    continue
                if art_id:
                    seen_article_ids.add(art_id)
                
                results.append({
                    "source": "Local-DB",
                    "score": hit.score,
                    "evidence_chunk": hit.payload.get('text', ''),         # Chunk text (Specific)
                    "context_summary": hit.payload.get('summarize', ''),   # Full Summary (Context)
                    "statement": hit.payload.get('statement', ''),
                    "url": hit.payload.get('url', ''),
                    "trust_level": hit.payload.get('label', 'Unknown')
                })
                
                if len(results) >= k:
                    break
            return results
        except Exception as e:
            print(f"⚠️ Lỗi Local Search: {e}")
            return []

    def search(self, query, k=3, threshold=0.65):
        print(f"\n🔎 Searching for: '{query}'")
        try:
            # B1: Embed query
            query_vec = self.model.encode(query, convert_to_tensor=False)
            
            # B2: Local search
            local_results = self._search_local(query_vec, k)
            
            # Get the highest score to decide
            best_score = local_results[0]['score'] if local_results else 0.0

            if best_score >= threshold:
                print(f"✅ Found in Database (Score: {best_score:.4f})")
                return local_results
            else:
                print(f"⚠️ Điểm Local thấp ({best_score:.4f} < {threshold}).")
                print("🔄 Switching to Internet search...")
                
                if not self.internet_searcher:
                    print("⚠️ InternetSearcher module not available.")
                    return local_results

                internet_results = self.internet_searcher.search(query, k=k)
                
                if internet_results:
                    # Return results immediately to user
                    formatted_results = []
                    for res in internet_results:
                        # With internet results, evidence is usually full text or long snippet
                        full_text = res.get('evidence', '')
                        
                        formatted_results.append({
                            "source": "Internet",
                            "score": res.get('score', 0.9), 
                            "evidence_chunk": full_text[:500] + "...", # Truncate for display
                            "context_summary": full_text,              # Keep full text as context for LLM
                            "url": res.get('url', ''),
                            "trust_level": res.get("trust_level", "Internet"),
                            "statement": res.get('statement', '')
                        })
                    return formatted_results
                else:
                    print("⚠️ Internet also found nothing.")
                    return local_results 

        except Exception as e:
            print(f"❌ System search error: {e}")
            return []
    
    def close(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()

# --- RUN TEST ---
if __name__ == "__main__":
    searcher = FactCheckSearcher()
    try:
        # Test query
        # query = "Vụ cháy chung cư mini Khương Hạ nguyên nhân do đâu?"
        query = "Phó Thủ tướng Trần Hồng Hà chúc mừng Đài Truyền hình Việt Nam."
        results = searcher.search(query, k=3, threshold=0.65) # Increase threshold to force it to go to Internet test
        
        if results:
            print("\n=== RETURNED RESULTS ===")
            for i, res in enumerate(results):
                print(f"\n#{i+1} [{res['source']}] (Score: {res['score']:.2f})")
                print(f"URL: {res.get('url', 'N/A')}")
                print(f"Evidence Chunk: {res.get('evidence_chunk', '')[:500]}...")
                print(f"Context Summary: {res.get('context_summary', '')[:500]}...")
                
                # Check context length to ensure full article is retrieved
                ctx_len = len(res.get('context_summary', ''))
                print(f"Full Context Length: {ctx_len} characters")
    finally:
        searcher.close()