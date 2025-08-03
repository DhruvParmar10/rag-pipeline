import asyncio
import time
from rag_pipeline import RAGPipeline

async def test_optimized_rag():
    print("🧪 Testing Optimized RAG Pipeline")
    print("=" * 50)
    
    try:
        # Test 1: Pipeline initialization
        start_time = time.time()
        print("1. Initializing RAG pipeline...")
        rag = RAGPipeline()
        init_time = time.time() - start_time
        print(f"✅ Pipeline initialized in {init_time:.2f}s")
        
        # Test 2: Health check
        print("\n2. Testing health check...")
        health = await rag.health_check()
        print(f"✅ Health status: {health['rag_pipeline']}")
        for service, status in health['services'].items():
            print(f"   📊 {service}: {status}")
        
        # Test 3: Quick embedding test
        print("\n3. Testing embedding generation...")
        test_question = "What is the grace period for premium payment?"
        embedding = await rag.embedding_service.generate_embedding(test_question)
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        
        # Test 4: Document processing (if fast enough)
        print("\n4. Testing document processing...")
        start_time = time.time()
        
        # Use a simpler test - just try to process one question
        document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        questions = ["What is the grace period for premium payment?"]
        
        answers = await rag.process_document_and_answer(document_url, questions)
        processing_time = time.time() - start_time
        
        print(f"✅ Processing completed in {processing_time:.2f}s")
        print(f"📝 Answer: {answers[0][:100]}..." if len(answers[0]) > 100 else f"📝 Answer: {answers[0]}")
        
        print(f"\n🎉 Optimized pipeline test completed!")
        print(f"⏱️  Total time: {processing_time:.2f}s (vs previous 155s)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_optimized_rag())
