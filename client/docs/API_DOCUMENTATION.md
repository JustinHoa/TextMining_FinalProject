# ViFactCheck API Documentation

## Overview
The ViFactCheck API provides an automated pipeline for verifying Vietnamese news and claims. It integrates a retrieval system (using Qdrant and BGE-M3) with an LLM (OpenAI) to analyze claims against evidence.

**Base URL**: `http://localhost:8000` (Default)

---

## Endpoints

### 1. Health Check
**GET** `/`

Verifies that the API server is running.

**Response:**
```json
{
  "message": "Welcome to ViFactCheck API. Use /check to verify news."
}
```

---

### 2. Verify Claim
**POST** `/check`

Submits a claim or statement for verification. The system retrieves relevant evidence from the local database (or optionally the internet if configured) and uses an LLM to adjucate the claim.

#### Request Headers
| Key | Value |
|---|---|
| `Content-Type` | `application/json` |

#### Request Body
| Field | Type | Required | Description |
|---|---|---|---|
| `claim` | `string` | Yes | The statement or claim to verify. Cannot be empty. |

**Example Request:**
```json
{
  "claim": "Vụ cháy chung cư mini Khương Hạ nguyên nhân do chập điện xe máy."
}
```

#### Response Body
| Field | Type | Description |
|---|---|---|
| `claim` | `string` | The original claim submitted. |
| `status` | `string` | The verdict (e.g., "True", "False", "Unverified"). |
| `explanation` | `string` | A detailed explanation from the LLM citing the evidence. |
| `confidence` | `float` | A confidence score (0.0 - 1.0) of the verdict. |
| `evidence` | `array` | A list of retrieved evidence objects used for verification. |

**Evidence Object Structure:**
| Field | Type | Description |
|---|---|---|
| `source` | `string` | Source of evidence ("Local-DB" or "Internet"). |
| `score` | `float` | Relevance score of the evidence. |
| `evidence_chunk` | `string` | A snippet or segment of the evidence text. |
| `context_summary` | `string` | A summary or full context of the source article. |
| `statement` | `string` | The headline or main statement derived from the source. |
| `url` | `string` | URL to the source article (if available). |
| `trust_level` | `string` | Credibility label of the source (e.g., "High", "Unknown"). |

**Example Response:**
```json
{
  "claim": "Vụ cháy chung cư mini Khương Hạ nguyên nhân do chập điện xe máy.",
  "status": "True",
  "explanation": "Theo báo cáo điều tra, nguyên nhân vụ cháy chung cư mini Khương Hạ được xác định là do chập mạch điện trên đường dây dẫn điện tại khu vực bình ắc quy của một xe tay ga đặt tại tầng 1.",
  "confidence": 0.95,
  "evidence": [
    {
      "source": "Local-DB",
      "score": 0.88,
      "evidence_chunk": "Nguyên nhân gây cháy được xác định do chập mạch điện trên đường dây dẫn điện tại khu vực bình ắc quy ...",
      "context_summary": "Công an TP Hà Nội công bố kết luận điều tra vụ cháy ...",
      "statement": "Công bố nguyên nhân vụ cháy chung cư mini ở Khương Hạ",
      "url": "https://vnexpress.net/...",
      "trust_level": "High"
    }
  ]
}
```

#### Error Responses
*   **400 Bad Request**: If the `claim` field is missing or empty.
*   **503 Service Unavailable**: If the searcher or LLM services failed to initialize.
*   **500 Internal Server Error**: If an unexpected error occurs during processing.

