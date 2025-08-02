#!/usr/bin/env python3

import asyncio
import httpx
from config import get_settings

async def test_openrouter_connection():
    """Test OpenRouter API connection"""
    settings = get_settings()
    
    print("🧪 Testing OpenRouter API Connection")
    print("=" * 40)
    
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/rag-pipeline",
        "X-Title": "RAG PDF QA Pipeline"
    }
    
    # Test payload
    payload = {
        "model": settings.generative_model,
        "messages": [
            {
                "role": "user",
                "content": "Hello! Can you confirm you're working properly? Please respond with 'Yes, I am Qwen and I am working correctly.'"
            }
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print(f"🔗 Connecting to OpenRouter...")
            print(f"📊 Model: {settings.generative_model}")
            
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            print(f"📈 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    answer = result["choices"][0]["message"]["content"]
                    print(f"✅ OpenRouter API is working!")
                    print(f"🤖 Model Response: {answer}")
                    return True
                else:
                    print("❌ No response generated")
                    return False
            else:
                print(f"❌ API Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_openrouter_connection())
    print(f"\n{'🎉 Test passed!' if success else '💥 Test failed!'}")
