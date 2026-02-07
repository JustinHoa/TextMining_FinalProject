
import requests
import json

BASE_URL = "http://localhost:8000"

def test_api_check():
    url = f"{BASE_URL}/check"
    
    # Payload testing
    payload = {
        # "claim": "Vụ cháy chung cư mini Khương Hạ nguyên nhân do chập điện xe máy."
        "claim": "Phó Thủ tướng Trần Hồng Hà chúc mừng Đài Truyền hình Việt Nam."
    }
    
    print(f"\nTesting POST {url} with claim: '{payload['claim']}'")
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success!")
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is it running?")
        print("Run: uvicorn main:app --reload")

if __name__ == "__main__":
    test_api_check()
