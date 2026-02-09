import requests
import tldextract
import os
import json
import trafilatura
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load API Key
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Trused domains
TIER_1_DOMAINS = {
    "chinhphu.vn", "baochinhphu.vn", "moet.gov.vn", "moh.gov.vn", 
    "nhandan.vn", "vtv.vn", "vov.vn", "thanhnien.vn", "tuoitre.vn", 
    "vnexpress.net", "laodong.vn", "tienphong.vn", "cand.com.vn", "qdnd.vn"
}

TIER_2_DOMAINS = {
    "dantri.com.vn", "vietnamnet.vn", "soha.vn", "cafef.vn", "zingnews.vn",
    "kenh14.vn", "vtc.vn", "baomoi.com", "sggp.org.vn", "nld.com.vn", "plo.vn"
}

class InternetSearcher:
    def __init__(self, api_key=SERPER_API_KEY):
        self.api_key = api_key
        self.url = "https://google.serper.dev/search"
        
        # Config browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://www.google.com/'
        }

    def _scrape_with_trafilatura(self, url):
        """
        Use requests to simulate browser to download HTML -> Trafilatura extract text
        """
        try:
            # 1. Download HTML using requests with good Header
            # timeout=10s to avoid hanging, verify=False to ignore SSL errors of some old sites
            response = requests.get(url, headers=self.headers, timeout=10, verify=False)
            
            # If request success (200)
            if response.status_code == 200:
                # Set encoding manually to avoid Vietnamese font errors
                response.encoding = response.apparent_encoding 
                html_content = response.text

                # 2. Give HTML to trafilatura to process
                text = trafilatura.extract(
                    html_content, 
                    include_comments=False, 
                    include_tables=True,
                    no_fallback=True # Try to get main content
                )
                
                if text and len(text) > 200: # Only take if content is long enough
                    return text
                    
        except Exception as e:
            print(f"Scrape error for {url}: {e}") # Debug if needed
            return None
        return None
    
    def calculate_authority_score(self, url):
        try:
            extracted = tldextract.extract(url)
            domain = f"{extracted.domain}.{extracted.suffix}"
            if "gov.vn" in domain: return 1.0
            if domain in TIER_1_DOMAINS: return 0.9
            if domain in TIER_2_DOMAINS: return 0.75
            if "edu.vn" in domain: return 0.6
            return 0.3
        except:
            return 0.1

    def search(self, query, k=5):
        print(f"Searching on the Internet: '{query}'...")
        
        payload = json.dumps({"q": query, "gl": "vn", "hl": "vi", "num": 10}) 
        headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}

        try:
            # Turn off InsecureRequestWarning when using verify=False
            requests.packages.urllib3.disable_warnings()
            
            response = requests.request("POST", self.url, headers=headers, data=payload)
            data = response.json()
            organic_results = data.get("organic", [])
            
            if not organic_results:
                return []

            candidates = []
            for item in organic_results:
                link = item.get("link", "")
                score = self.calculate_authority_score(link)
                candidates.append({"item": item, "score": score})
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            top_candidates = candidates[:k] 

            final_results = []
            print(f"-> Scraping content from {len(top_candidates)} URLs...")
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_obj = {
                    executor.submit(self._scrape_with_trafilatura, obj['item']['link']): obj
                    for obj in top_candidates
                }

                for future in as_completed(future_to_obj):
                    obj = future_to_obj[future]
                    item = obj['item']
                    auth_score = obj['score']
                    
                    try:
                        full_text = future.result()
                        
                        # Logic Fallback
                        if full_text:
                            evidence = full_text
                            is_full_content = True
                        else:
                            evidence = item.get("snippet", "")
                            is_full_content = False

                    except Exception:
                        evidence = item.get("snippet", "")
                        is_full_content = False

                    if len(evidence) > 4000:
                        evidence = evidence[:4000] + "..."

                    # Mark in console to know which article got full content
                    status_icon = "✅ FULL" if is_full_content else "⚠️ SNIPPET"
                    
                    final_results.append({
                        "source": "Internet",
                        "score": auth_score,
                        "evidence": evidence,
                        "statement": item.get("title", ""),
                        "url": item.get("link", ""),
                        "trust_level": "High" if auth_score >= 0.75 else "Low",
                        "status": status_icon # debugging
                    })

            final_results.sort(key=lambda x: x['score'], reverse=True)
            return final_results

        except Exception as e:
            print(f"Internet API Error: {e}")
            return []

# --- TEST BLOCK ---
def main():
    if not SERPER_API_KEY:
        print("Error: SERPER_API_KEY is not configured in .env file")
        return

    searcher = InternetSearcher()
    query = "Vụ cháy chung cư mini Khương Hạ nguyên nhân do đâu?"
    results = searcher.search(query, k=5)
    
    print(f"\n=== FOUND {len(results)} RESULTS ===")
    for i, result in enumerate(results):
        print(f"\n#{i+1} {result['status']} [{result['trust_level']}] Score: {result['score']}")
        print(f"Title: {result['statement']}")
        print(f"URL: {result['url']}")
        # Print first 1000 characters, remove newlines for easier reading
        content_preview = result['evidence'][:1000].replace('\n', ' ')
        print(f"Content Preview: {content_preview}...") 

if __name__ == "__main__":
    main()