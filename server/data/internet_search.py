import requests
import tldextract
import os
from dotenv import load_dotenv

load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
print(f"SERPER_API_KEY: {SERPER_API_KEY is not None}")

TIER_1_DOMAINS = {
    "chinhphu.vn", "baochinhphu.vn", "moet.gov.vn", "moh.gov.vn", 
    "nhandan.vn", "vtv.vn", "vov.vn", "thanhnien.vn", "tuoitre.vn", 
    "vnexpress.net", "laodong.vn", "tienphong.vn", "cand.com.vn"
}

TIER_2_DOMAINS = {
    "dantri.com.vn", "vietnamnet.vn", "soha.vn", "cafef.vn", "zingnews.vn",
    "kenh14.vn", "vtc.vn", "baomoi.com", "sggp.org.vn", "nld.com.vn"
}

class InternetSearcher:
    def __init__(self, api_key=SERPER_API_KEY):
        self.api_key = api_key
        self.url = "https://google.serper.dev/search"

    def _calculate_authority_score(self, url):
        """Hàm chấm điểm uy tín dựa trên tên miền"""
        try:
            extracted = tldextract.extract(url)
            domain = f"{extracted.domain}.{extracted.suffix}"
            
            if "gov.vn" in domain: 
                return 1.0
            if domain in TIER_1_DOMAINS: 
                return 0.9
            if domain in TIER_2_DOMAINS: 
                return 0.75
            if "edu.vn" in domain: 
                return 0.6
            
            return 0.3
        except:
            return 0.1

    def search(self, query, k=5):
        """Tìm kiếm, chấm điểm và trả về Top K kết quả uy tín"""
        print(f"Đang tìm trên Internet: '{query}'...")
        
        payload = {"q": query, "gl": "vn", "hl": "vi", "num": 15}
        headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}

        try:
            response = requests.post(self.url, headers=headers, json=payload)
            data = response.json()
            organic_results = data.get("organic", [])
            
            processed_results = []
            for item in organic_results:
                link = item.get("link", "")
                
                # Chấm điểm uy tín
                auth_score = self._calculate_authority_score(link)
                
                processed_results.append({
                    "source": "Internet",
                    "score": auth_score,
                    "evidence": item.get("snippet", ""),
                    "statement": item.get("title", ""),
                    "url": link,
                    "trust_level": "High" if auth_score >= 0.75 else "Low"
                })

            processed_results.sort(key=lambda x: x['score'], reverse=True)
            
            return processed_results[:k]
            
        except Exception as e:
            print(f"Lỗi API Internet: {e}")
            return []