import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class LLMFactChecker:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("⚠️ OPENAI_API_KEY not found in .env")
        
        # Using ShopAIKey as OpenAI Proxy
        self.client = OpenAI(
            api_key=OPENAI_API_KEY, 
            base_url="https://api.shopaikey.com/v1"
        )
        self.model = "gpt-4o" # Ensure this model is supported by your plan

    def verify_claim(self, claim, evidence_list):
        """
        Verify a claim against a list of retrieved evidence.
        
        Args:
            claim (str): The statement to check.
            evidence_list (list): List of dicts containing 'evidence_chunk', 'context_summary', 'source', 'url'.
            
        Returns:
            dict: {
                "status": "ĐÚNG" | "SAI" | "KHÔNG ĐỦ THÔNG TIN",
                "explanation": "...",
                "confidence": 0.0 - 1.0
            }
        """
        
        # 1. Prepare Context
        context_str = ""
        if not evidence_list:
             return {
                "status": "KHÔNG ĐỦ THÔNG TIN",
                "explanation": "Không tìm thấy thông tin liên quan từ cơ sở dữ liệu hoặc Internet.",
                "confidence": 0.0
            }

        for i, item in enumerate(evidence_list):
            context_str += f"""
            [Source #{i+1}] ({item.get('source', 'Unknown')} - {item.get('url', 'No URL')})
            EVIDENCE: {item.get('evidence_chunk', '')}
            CONTEXT SUMMARY: {item.get('context_summary', '')}
            ---
            """
        
        # 2. System Prompt
        system_prompt = """
        Bạn là một trợ lý kiểm chứng tin giả (Fact Checker) chuyên nghiệp và trung thực.
        Nhiệm vụ của bạn là so sánh "TUYÊN BỐ" (Claim) của người dùng với các "NGUỒN TIN" (Sources) được cung cấp.
        
        QUY TẮC CỐT LÕI:
        1. Chỉ sử dụng thông tin trong NGUỒN TIN được cung cấp. KHÔNG được bịa đặt hoặc dùng kiến thức bên ngoài.
        2. Nếu thông tin trong nguồn khẳng định tuyên bố là đúng -> Trả về "ĐÚNG".
        3. Nếu thông tin trong nguồn mâu thuẫn hoặc phủ định tuyên bố -> Trả về "SAI".
        4. Nếu các nguồn tin không đề cập hoặc thông tin quá mơ hồ -> Trả về "KHÔNG ĐỦ THÔNG TIN".
        
        OUTPUT FORMAT (JSON):
        {
            "status": "ĐÚNG" | "SAI" | "KHÔNG ĐỦ THÔNG TIN",
            "explanation": "Giải thích ngắn gọn, trích dẫn nguồn (VD: Theo nguồn #1...)",
            "confidence": <float từ 0.0 đến 1.0>
        }
        """

        user_prompt = f"""
        TUYÊN BỐ CẦN KIỂM CHỨNG:
        "{claim}"

        CÁC NGUỒN TIN THU THẬP ĐƯỢC:
        {context_str}
        
        Hãy phân tích và trả về định dạng JSON.
        """

        # 3. Call OpenAI
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return {
                "status": "LỖI",
                "explanation": f"Gặp lỗi khi gọi AI: {e}",
                "confidence": 0.0
            }
