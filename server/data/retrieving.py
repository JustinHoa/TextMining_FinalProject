from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from FlagEmbedding import BGEM3FlagModel
import uuid
import sys

from internet_search import InternetSearcher 

class FactCheckSearcher:
    def __init__(self, db_path="vectordb", collection_name="vifactcheck"):
        # creating Qdrant client
        print(f"Kết nối Local DB tại: {db_path}")
        self.client = QdrantClient(path=db_path)
        self.collection_name = collection_name
        
        # 2. creating embedding model
        print("Load model BGE-M3...")
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        
        # 3. online searcher
        self.internet_searcher = InternetSearcher() 

    def _search_local(self, query_vec, k):
        """Hàm nội bộ tìm trong Qdrant"""
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vec.tolist(),
                limit=k
            )
            results = []
            for hit in search_result:
                results.append({
                    "source": "Local-DB",
                    "score": hit.score,
                    "evidence": hit.payload.get('evidence'),
                    "statement": hit.payload.get('statement'),
                    "url": hit.payload.get('url'),
                    "trust_level": hit.payload.get('label', 'Verified Dataset')
                })
            return results
        except Exception as e:
            print(f"⚠️ Lỗi Local Search: {e}")
            return []

    def _save_to_local(self, internet_results):
        """
        Hàm lưu dữ liệu mới từ Internet vào Qdrant (Cơ chế tự học)
        """
        if not internet_results:
            return

        print(f"💾 Đang lưu {len(internet_results)} kiến thức mới vào Database...")
        
        try:
            # embedding evidence from internet results
            texts = [item['evidence'] for item in internet_results]
            embeddings = self.model.encode(texts, return_dense=True)['dense_vecs']
            
            # creating points to upsert
            points = []
            for i, item in enumerate(internet_results):
                new_id = str(uuid.uuid4())  # random uuid
                
                points.append(PointStruct(
                    id=new_id,
                    vector=embeddings[i].tolist(),
                    payload={
                        "statement": item.get('statement', ''), # Title bài báo
                        "evidence": item.get('evidence', ''),   # Nội dung snippet
                        "url": item.get('url', ''),
                        "topic": "Internet News",               # Đánh dấu nguồn
                        "label": item.get('trust_level', 'Unverified') # Lưu độ tin cậy
                    }
                ))
            
            # upserting into Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print("Đã lưu thành công vào Database.")
            
        except Exception as e:
            print(f"Lỗi khi lưu vào DB: {e}")

    def search(self, query, k=3, threshold=0.55):
        try:
            # B1: Embed query
            query_vec = self.model.encode([query], return_dense=True)['dense_vecs'][0]
            
            # B2: Tìm Local trước
            local_results = self._search_local(query_vec, k)
            
            # Lấy điểm cao nhất
            best_local_score = local_results[0]['score'] if local_results else 0.0
            
            # B3: Quyết định
            if best_local_score >= threshold:
                print(f"✅ Tìm thấy trong Database (Score: {best_local_score:.4f})")
                return local_results
            else:
                print(f"⚠️ Kết quả Local thấp ({best_local_score:.4f} < {threshold}).")
                print("🔄 Chuyển sang tìm kiếm Internet...")
                
                # Gọi module Internet Search
                internet_results = self.internet_searcher.search(query, k=k)
                
                if internet_results:
                    # --- BƯỚC MỚI: TỰ ĐỘNG LƯU VÀO DB ---
                    self._save_to_local(internet_results)
                    return internet_results
                else:
                    print("⚠️ Internet cũng không tìm thấy gì.")
                    return local_results 

        except Exception as e:
            print(f"❌ Lỗi hệ thống search: {e}")
            return []
    
    def close(self):
        if self.client:
            self.client.close()

# --- CHẠY THỬ ---
if __name__ == "__main__":
    searcher = None
    try:
        searcher = FactCheckSearcher()
        
        # Test 1 câu hỏi mới toanh chưa có trong DB
        query = "Nguyên nhân vụ cháy chung cư mini Khương Hạ"
        
        print(f"\n🔎 Query: '{query}'")
        output = searcher.search(query, k=3, threshold=0.6)
        
        if output:
            print(f"\n✅ KẾT QUẢ TRẢ VỀ:")
            for res in output:
                print("-" * 50)
                print(f"[{res['source']}] Trust: {res['trust_level']}")
                print(f"Content: {res['evidence'][:150]}...")
                
    finally:
        if searcher:
            searcher.close()
            print("\nĐã đóng kết nối.")