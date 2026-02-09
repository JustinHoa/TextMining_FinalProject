# 📊 FLOW HỆ THỐNG FACT-CHECKING - ViFactCheck

## 🎯 Tổng Quan
Hệ thống fact-checking của bạn sử dụng **hybrid approach** kết hợp **Local Vector Database** và **Internet Search** để xác minh tính chính xác của tin tức.

---

## 🔄 FLOW TỔNG QUÁT

```
┌─────────────┐
│   CLIENT    │
│   (React)   │
└──────┬──────┘
       │
       │ POST /check
       │ {"claim": "..."}
       ▼
┌───────────────────────────────────────────────────┐
│             FASTAPI SERVER (main.py)              │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │  1. Nhận Claim từ User                      │ │
│  └─────────────────┬───────────────────────────┘ │
│                    │                             │
│                    ▼                             │
│  ┌─────────────────────────────────────────────┐ │
│  │  2. RETRIEVAL PHASE                         │ │
│  │     FactCheckSearcher.search()              │ │
│  │     (services/retrieving.py)                │ │
│  └─────────────────┬───────────────────────────┘ │
│                    │                             │
│                    ▼                             │
│  ┌─────────────────────────────────────────────┐ │
│  │  3. VERIFICATION PHASE                      │ │
│  │     LLMFactChecker.verify_claim()           │ │
│  │     (services/llm_service.py)               │ │
│  └─────────────────┬───────────────────────────┘ │
│                    │                             │
└────────────────────┼─────────────────────────────┘
                     │
                     ▼
              ┌─────────────┐
              │   RESPONSE  │
              │   (JSON)    │
              └─────────────┘
```

---

## 📝 CHI TIẾT TỪNG BƯỚC

### **BƯỚC 1: Nhận Request từ Client**

**File:** `main.py` (Line 70-101)

```python
@app.post("/check", response_model=CheckResponse)
async def check_claim(request: CheckRequest):
    claim = request.claim.strip()
    # Validate claim không rỗng
    # Gọi searcher và llm_checker
```

**Input:**
```json
{
  "claim": "Phó Thủ tướng Trần Hồng Hà chúc mừng Đài Truyền hình Việt Nam"
}
```

---

### **BƯỚC 2: RETRIEVAL PHASE - Tìm Kiếm Bằng Chứng**

**File:** `services/retrieving.py` - Class `FactCheckSearcher`

#### 2.1. Embedding Query
```python
query_vec = self.model.encode(query, convert_to_tensor=False)
```
- **Model:** `BAAI/bge-m3` (BGE-M3 Embedding Model)
- **Output:** Vector 1024 chiều đại diện cho semantic của claim

#### 2.2. Local Search - Tìm trong Vector Database
```python
local_results = self._search_local(query_vec, k=5)
```

**Logic:**
1. Query Qdrant vector database collection `vifactcheck_chunks`
2. Lấy `k * 3 = 15` kết quả để lọc duplicate
3. **Deduplication:** Chỉ lấy 1 chunk có score cao nhất từ mỗi article
   - Sử dụng `article_id` để identify
4. Trả về top `k=5` kết quả

**Mỗi result chứa:**
```python
{
    "source": "Local-DB",
    "score": 0.82,                    # Cosine similarity
    "evidence_chunk": "...",          # Text của chunk cụ thể
    "context_summary": "...",         # Summary toàn bài để LLM hiểu context
    "statement": "...",               # Title của article
    "url": "https://...",
    "trust_level": "SUPPORTED/REFUTED/NEI"  # Label từ dataset
}
```

#### 2.3. Decision Logic - Threshold Checking

```python
best_score = local_results[0]['score'] if local_results else 0.0

if best_score >= threshold:  # threshold = 0.65
    return local_results  # ✅ Tìm thấy trong DB
else:
    # ⚠️ Chuyển sang Internet Search
```

**Quyết định:**
- **Score >= 0.65:** Tin tưởng vào kết quả local, return ngay
- **Score < 0.65:** Claim quá mới/khác biệt, phải search Internet

---

#### 2.4. Internet Search (Fallback)

**File:** `services/internet_search.py` - Class `InternetSearcher`

**Kích hoạt khi:** Local search score < 0.65

**Quy trình:**

##### Step 1: Google Search qua Serper API
```python
payload = {"q": query, "gl": "vn", "hl": "vi", "num": 10}
response = serper_api.search(payload)
organic_results = response.get("organic", [])
```

##### Step 2: Authority Scoring
```python
def calculate_authority_score(url):
    # Phân loại độ tin cậy nguồn
    if "gov.vn" in domain: return 1.0
    if domain in TIER_1_DOMAINS: return 0.9  # VTV, VnExpress, etc.
    if domain in TIER_2_DOMAINS: return 0.75 # Dantri, VietnamNet
    if "edu.vn" in domain: return 0.6
    return 0.3  # Unknown
```

**Trusted Sources:**
- **Tier 1:** chinhphu.vn, vtv.vn, vnexpress.net, tuoitre.vn, thanhnien.vn...
- **Tier 2:** dantri.com.vn, vietnamnet.vn, zingnews.vn...

##### Step 3: Parallel Web Scraping
```python
with ThreadPoolExecutor(max_workers=5) as executor:
    for url in top_5_urls:
        full_text = _scrape_with_trafilatura(url)
```

**Trafilatura Logic:**
- Dùng `requests` với User-Agent giả lập browser
- Extract main content từ HTML (bỏ ads, menu, footer)
- **Fallback:** Nếu scrape thất bại → dùng snippet từ Google

**Output:**
```python
{
    "source": "Internet",
    "score": 0.9,                     # Authority score
    "evidence_chunk": "...[500]",     # Truncated preview
    "context_summary": "...[4000]",   # Full text cho LLM
    "url": "https://vtv.vn/...",
    "trust_level": "High/Low",
    "statement": "Article title"
}
```

---

### **BƯỚC 3: VERIFICATION PHASE - Xác Minh với LLM**

**File:** `services/llm_service.py` - Class `LLMFactChecker`

**Input:** 
- `claim`: Tuyên bố cần kiểm tra
- `evidence_list`: 5 bằng chứng từ retrieval phase

#### 3.1. Context Preparation
```python
context_str = ""
for i, item in enumerate(evidence_list):
    context_str += f"""
    [Source #{i+1}] ({item['source']} - {item['url']})
    EVIDENCE: {item['evidence_chunk']}
    CONTEXT SUMMARY: {item['context_summary']}
    ---
    """
```

#### 3.2. System Prompt (Instruction cho AI)
```
Bạn là fact checker chuyên nghiệp.

QUY TẮC:
1. Chỉ dùng thông tin từ NGUỒN TIN được cung cấp
2. Nếu nguồn khẳng định → "ĐÚNG"
3. Nếu nguồn mâu thuẫn/phủ định → "SAI"
4. Nếu nguồn không đề cập → "KHÔNG ĐỦ THÔNG TIN"

OUTPUT: JSON format
{
  "status": "ĐÚNG" | "SAI" | "KHÔNG ĐỦ THÔNG TIN",
  "explanation": "...",
  "confidence": 0.0 - 1.0
}
```

#### 3.3. LLM Call
```python
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    response_format={"type": "json_object"},
    temperature=0.0  # Deterministic
)
```

**Sử dụng:**
- **API:** OpenAI qua ShopAIKey proxy
- **Model:** GPT-4o
- **Temperature:** 0.0 (để nhất quán)
- **JSON Mode:** Bắt buộc output là JSON

---

### **BƯỚC 4: Trả Response về Client**

**File:** `main.py` (Line 91-97)

```python
return {
    "claim": claim,
    "status": llm_result.get("status", "LỖI"),
    "explanation": llm_result.get("explanation", "Không thể phân tích"),
    "confidence": llm_result.get("confidence", 0.0),
    "evidence": evidence_list  # 5 nguồn tin đã tìm được
}
```

**Response Example:**
```json
{
  "claim": "Phó Thủ tướng Trần Hồng Hà chúc mừng Đài Truyền hình Việt Nam",
  "status": "ĐÚNG",
  "explanation": "Theo nguồn #1 từ VTV.vn, Phó Thủ tướng Trần Hồng Hà đã đến chúc mừng Đài Truyền hình Việt Nam nhân dịp kỷ niệm...",
  "confidence": 0.95,
  "evidence": [
    {
      "source": "Local-DB",
      "score": 0.87,
      "evidence_chunk": "...",
      "url": "https://vtv.vn/...",
      "trust_level": "SUPPORTED"
    },
    // ... 4 nguồn khác
  ]
}
```

---

## 🔧 CÁC THÀNH PHẦN CHÍNH

### 1. **Vector Database (Qdrant)**
- **Location:** `vectordb/`
- **Collection:** `vifactcheck_chunks`
- **Data:** Dataset ViFactCheck đã được chunk và embed
- **Vector Dimension:** 1024 (BGE-M3)

### 2. **Embedding Model**
- **Model:** `BAAI/bge-m3`
- **Purpose:** Chuyển text → vector để semantic search
- **Load:** Khi server startup

### 3. **LLM (GPT-4o)**
- **Provider:** OpenAI qua ShopAIKey
- **Model:** gpt-4o
- **Purpose:** So sánh claim vs evidence → verdict

### 4. **Internet Search**
- **API:** Google Serper API
- **Scraper:** Trafilatura + Requests
- **Parallel:** ThreadPoolExecutor với 5 workers

---

## ⚙️ CẤU HÌNH QUAN TRỌNG

### Retrieval Phase
```python
k = 5              # Số lượng evidence tối đa
threshold = 0.65   # Ngưỡng chuyển sang Internet
```

### Internet Search
```python
TIER_1_DOMAINS = {
    "chinhphu.vn", "vtv.vn", "vnexpress.net", 
    "tuoitre.vn", "thanhnien.vn", ...
}
```

### LLM Settings
```python
model = "gpt-4o"
temperature = 0.0        # Deterministic
response_format = "json"
```

---

## 📊 DECISION TREE

```
Claim Input
    │
    ▼
Embed với BGE-M3
    │
    ▼
Search Local DB
    │
    ├─► Score >= 0.65 ──► Use Local Results
    │                         │
    │                         ▼
    │                     Feed to LLM
    │                         │
    └─► Score < 0.65 ──► Internet Search
                              │
                              ├─► Has Results? ──► Feed to LLM
                              │
                              └─► No Results ──► Return "KHÔNG ĐỦ THÔNG TIN"
```

---

## 🎯 KEY FEATURES

### ✅ Hybrid Search
- Ưu tiên local database (nhanh, accurate)
- Fallback Internet nếu không tìm thấy (coverage)

### ✅ Deduplication
- Tránh trùng lặp article trong kết quả
- Chỉ lấy chunk tốt nhất của mỗi article

### ✅ Authority Scoring
- Đánh giá độ tin cậy nguồn
- Ưu tiên nguồn chính thống (VTV, Chính phủ)

### ✅ Full Content Extraction
- Scrape toàn bộ article từ Internet
- Fallback snippet nếu thất bại

### ✅ LLM Verification
- GPT-4o phân tích context-aware
- JSON mode đảm bảo format chuẩn

---

## 🚀 STARTUP SEQUENCE

```python
# main.py - @app.on_event("startup")
1. Load BGE-M3 model         (15-30s)
2. Connect Qdrant DB         (1-2s)
3. Initialize LLM client     (instant)
4. Initialize InternetSearcher (instant)
```

**Tổng thời gian khởi động:** ~20-40 giây

---

## 📌 TÓM TẮT

| Giai đoạn       | Thành phần              | Thời gian      | Output                    |
|-----------------|-------------------------|----------------|---------------------------|
| **Input**       | Client POST             | Instant        | Claim string              |
| **Retrieval**   | BGE-M3 + Qdrant         | 0.5 - 2s       | Top 5 evidence (local)    |
|                 | Serper + Trafilatura    | 3 - 8s         | Top 5 evidence (internet) |
| **Verification**| GPT-4o                  | 2 - 5s         | Status + Explanation      |
| **Output**      | FastAPI JSON Response   | Instant        | Full verdict              |

**Total Response Time:**
- Local hit: **2-7 giây**
- Internet fallback: **5-13 giây**

---

🎉 **Hệ thống của bạn kết hợp tốt giữa tốc độ (local DB) và coverage (Internet search)!**
