import os
import asyncio
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

async def test_hf_embedding():
    api_key = os.getenv('HF_TOKEN')
    model_id = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
    if not api_key:
        print("❌ HF_TOKEN not set in environment.")
        return
    
    print(f"🔑 Using API key: {api_key[:10]}...")
    print(f"📊 Using model: {model_id}")
    
    client = InferenceClient(provider="nebius", token=api_key)
    test_text = "This is a test sentence for embedding."
    try:
        embedding = client.feature_extraction(test_text, model=model_id)
        print(f"✅ Embedding received! Type: {type(embedding)}")
        print(f"📊 Raw shape info: Length={len(embedding)}")
        
        # Debug the structure
        if isinstance(embedding, list) and len(embedding) > 0:
            print(f"🔍 First element type: {type(embedding[0])}")
            if isinstance(embedding[0], list):
                print(f"📏 Inner array length: {len(embedding[0])}")
                flat_embedding = embedding[0]
                print(f"🔢 First 5 values: {flat_embedding[:5]}")
            else:
                print(f"� Values: {embedding[:5]}")
        else:
            print(f"🔢 Raw embedding: {embedding}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

async def test_embedding_service():
    """Test the actual EmbeddingService class"""
    try:
        from ai_services import EmbeddingService
        
        print("\n🧪 Testing EmbeddingService class...")
        service = EmbeddingService()
        
        test_text = "Hello my name is dhruv."
        embedding = await service.generate_embedding(test_text)
        
        print(f"✅ EmbeddingService works! Embedding length: {len(embedding)}")
        print(f"🔢 First 5 values: {embedding[:5]}")
        
    except Exception as e:
        print(f"❌ EmbeddingService error: {e}")

if __name__ == "__main__":
    asyncio.run(test_hf_embedding())
    asyncio.run(test_embedding_service())
