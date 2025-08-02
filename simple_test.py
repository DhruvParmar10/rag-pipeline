#!/usr/bin/env python3

from qdrant_client import QdrantClient

# Test with the exact format you provided
url = "https://7c5b51e7-db3e-4b63-bc21-2a1e0e3b4d8a.us-east4-0.gcp.cloud.qdrant.io:6333"
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wNzUSRN2K0wJvpBJV7VpgRmE-FMNLZOAwxgIwLiTMCo"

print(f"Testing URL: {url}")
print(f"API Key: {api_key[:30]}...")

try:
    qdrant_client = QdrantClient(
        url=url,
        api_key=api_key,
    )
    
    print("✅ Client created successfully")
    
    # Test basic connection
    collections = qdrant_client.get_collections()
    print(f"✅ Connection successful!")
    print(f"Collections: {collections}")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print(f"Error type: {type(e)}")
    
    # Additional debugging
    import requests
    try:
        response = requests.get(url + "/collections", headers={"api-key": api_key}, timeout=10)
        print(f"Direct HTTP test - Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
    except Exception as http_e:
        print(f"HTTP test failed: {http_e}")
