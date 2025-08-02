import asyncio
import httpx
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
BEARER_TOKEN = "7c695e780a6ab6eacffab7c9326e5d8e472a634870a6365979c5671ad28f003c"

async def test_rag_pipeline():
    """Comprehensive test of the RAG pipeline"""
    
    print("🧪 Testing Advanced RAG Pipeline")
    print("=" * 50)
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # Test 1: Health check
            print("\n1. Testing health check...")
            health_response = await client.get(f"{BASE_URL.replace('/api/v1', '')}/health")
            print(f"Health status: {health_response.status_code}")
            
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"Pipeline status: {health_data['status']}")
                print(f"Services: {health_data['services']}")
            
            # Test 2: System stats
            print("\n2. Testing system stats...")
            stats_response = await client.get(f"{BASE_URL}/stats", headers=headers)
            print(f"Stats status: {stats_response.status_code}")
            
            if stats_response.status_code == 200:
                stats_data = stats_response.json()
                print(f"Vector store info: {stats_data.get('vector_store', {})}")
            
            # Test 3: Main RAG pipeline with sample document
            print("\n3. Testing RAG pipeline with document processing...")
            
            test_request = {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
                "questions": [
                    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                    "What is the waiting period for pre-existing diseases (PED) to be covered?",
                    "Does this policy cover maternity expenses, and what are the conditions?",
                    "What is the waiting period for cataract surgery?",
                    "Are the medical expenses for an organ donor covered under this policy?"
                ]
            }
            
            print(f"Processing document: {test_request['documents']}")
            print(f"Number of questions: {len(test_request['questions'])}")
            
            start_time = datetime.now()
            
            response = await client.post(
                f"{BASE_URL}/hackrx/run",
                headers=headers,
                json=test_request
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            print(f"Response status: {response.status_code}")
            print(f"Processing time: {processing_time:.2f} seconds")
            
            if response.status_code == 200:
                result = response.json()
                print("\n✅ RAG Pipeline Test Results:")
                print("-" * 40)
                
                for i, (question, answer) in enumerate(zip(test_request['questions'], result['answers']), 1):
                    print(f"\nQ{i}: {question}")
                    print(f"A{i}: {answer}")
                    print("-" * 40)
                
                # Test 4: Document info
                print("\n4. Testing document info...")
                doc_info_response = await client.get(
                    f"{BASE_URL}/document/info",
                    headers=headers,
                    params={"document_url": test_request['documents']}
                )
                
                if doc_info_response.status_code == 200:
                    doc_info = doc_info_response.json()
                    print(f"Document processed: {doc_info.get('is_processed', False)}")
                    print(f"Document hash: {doc_info.get('document_hash', 'N/A')}")
                
                print(f"\n🎉 All tests completed successfully!")
                print(f"Total processing time: {processing_time:.2f} seconds")
                
            else:
                print(f"❌ Pipeline test failed: {response.text}")
                
        except httpx.TimeoutException:
            print("❌ Request timed out. The model might be downloading or initializing.")
        except Exception as e:
            print(f"❌ Test failed with error: {e}")

async def test_error_handling():
    """Test error handling scenarios"""
    print("\n🔧 Testing Error Handling")
    print("=" * 30)
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Test invalid document URL
        print("\n1. Testing invalid document URL...")
        invalid_request = {
            "documents": "https://invalid-url.com/nonexistent.pdf",
            "questions": ["What is this document about?"]
        }
        
        try:
            response = await client.post(
                f"{BASE_URL}/hackrx/run",
                headers=headers,
                json=invalid_request
            )
            print(f"Status: {response.status_code}")
            if response.status_code != 200:
                print(f"Error response: {response.text}")
        except Exception as e:
            print(f"Expected error: {e}")
        
        # Test unauthorized access
        print("\n2. Testing unauthorized access...")
        try:
            response = await client.post(
                f"{BASE_URL}/hackrx/run",
                headers={"Content-Type": "application/json"},
                json={"documents": "https://example.com/test.pdf", "questions": ["Test?"]}
            )
            print(f"Status: {response.status_code}")
        except Exception as e:
            print(f"Expected error: {e}")

if __name__ == "__main__":
    print("🚀 Starting comprehensive RAG pipeline tests...")
    asyncio.run(test_rag_pipeline())
    asyncio.run(test_error_handling())
