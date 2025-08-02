#!/usr/bin/env python3

import asyncio
from config import get_settings
from vector_store import VectorStore

async def test_qdrant_cloud():
    """Test Qdrant Cloud connection"""
    print("🧪 Testing Qdrant Cloud Connection")
    print("=" * 40)
    
    settings = get_settings()
    
    print(f"🔗 Qdrant URL: {settings.qdrant_host}")
    print(f"🔑 API Key: {settings.qdrant_api_key[:20]}...")
    print(f"📚 Collection: {settings.qdrant_collection_name}")
    
    try:
        # Initialize vector store
        vector_store = VectorStore()
        
        # Test health check
        is_healthy = await vector_store.health_check()
        print(f"🏥 Health Check: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")
        
        # Get collection info
        collection_info = await vector_store.get_collection_info()
        print(f"📊 Collection Info: {collection_info}")
        
        print("\n🎉 Qdrant Cloud connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_qdrant_cloud())
    print(f"\n{'✅ Test passed!' if success else '💥 Test failed!'}")
