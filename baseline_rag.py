"""
Baseline RAG Implementation for E-commerce Customer Support
Part 1 of Agentic RAG Masterclass

This demonstrates a simple RAG system that retrieves policy documents
but cannot take actions or verify information.
"""

import os
from typing import List, Dict
import numpy as np
import httpx
from openai import OpenAI
import faiss

# ⚠️ SSL BYPASS FOR CORPORATE NETWORKS (DEMO ONLY - NOT FOR PRODUCTION)
# This disables SSL certificate verification to work with corporate proxies
# For production, configure proper SSL certificates instead
http_client = httpx.Client(verify=False)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=http_client  # SSL bypass for corporate networks
)

# Sample knowledge base documents
KNOWLEDGE_BASE = {
    "return_policy.md": """
# Return Policy

## Eligibility
- Items can be returned within 30 days of delivery
- Items must be in original condition with tags attached
- Damaged or defective items: Full refund
- Change of mind: Refund minus 15% restocking fee

## Process
1. Submit return request with order number
2. Print return label (provided via email)
3. Ship item within 5 business days
4. Refund processed within 7-10 business days after receipt

## Non-returnable Items
- Perishable goods
- Personal care items
- Digital downloads
""",
    "shipping_policy.md": """
# Shipping Policy

## Delivery Times
- Standard shipping: 5-7 business days
- Express shipping: 2-3 business days
- International: 10-15 business days

## Shipping Costs
- Orders over $50: Free standard shipping
- Orders under $50: $5.99 standard, $15.99 express
- International: Calculated at checkout

## Tracking
- Tracking number sent via email within 24 hours
- Track on our website or carrier website

## Regions
- We ship to USA, Canada, UK, Australia
- Some remote areas may have extended delivery times
""",
    "product_info.md": """
# Product Information

## Product Care
- Machine wash cold, tumble dry low
- Do not bleach or iron directly on prints
- Check individual product labels for specific care

## Sizing
- Refer to size chart on product pages
- If between sizes, we recommend sizing up
- Contact support for personalized sizing help

## Materials
- All cotton products are 100% organic cotton
- Synthetic blends clearly labeled
- Hypoallergenic materials available
"""
}


class BaselineRAG:
    """Simple RAG system using FAISS and OpenAI."""

    def __init__(self, embedding_model: str = "text-embedding-3-large"):
        self.embedding_model = embedding_model
        self.chunks = []
        self.chunk_metadata = []
        self.index = None
        self.dimension = 3072  # text-embedding-3-large dimension

    def chunk_documents(self, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Split documents into overlapping chunks."""
        chunks = []

        for doc_name, content in KNOWLEDGE_BASE.items():
            # Simple sentence-based chunking
            sentences = content.split('\n')
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) < chunk_size:
                    current_chunk += sentence + "\n"
                else:
                    if current_chunk.strip():
                        chunks.append({
                            "text": current_chunk.strip(),
                            "source": doc_name,
                            "chunk_id": len(chunks)
                        })
                    current_chunk = sentence + "\n"

            # Add remaining chunk
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": doc_name,
                    "chunk_id": len(chunks)
                })

        return chunks

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text using OpenAI."""
        response = client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def build_index(self):
        """Build FAISS index from document chunks."""
        print("📚 Chunking documents...")
        self.chunks = self.chunk_documents()
        print(f"   Created {len(self.chunks)} chunks")

        print("🔢 Generating embeddings...")
        embeddings = []
        for i, chunk in enumerate(self.chunks):
            if i % 5 == 0:
                print(f"   Processing chunk {i+1}/{len(self.chunks)}")
            embedding = self.embed_text(chunk["text"])
            embeddings.append(embedding)
            self.chunk_metadata.append(chunk)

        # Create FAISS index
        embeddings_matrix = np.vstack(embeddings)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_matrix)
        print(f"✅ Index built with {self.index.ntotal} vectors")

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve top-k relevant chunks."""
        query_embedding = self.embed_text(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                **self.chunk_metadata[idx],
                "similarity_score": float(1 / (1 + distance))  # Convert distance to similarity
            })

        return results

    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate response using retrieved context."""
        # Build context from chunks
        context = "\n\n".join([
            f"[Source: {chunk['source']}]\n{chunk['text']}"
            for chunk in context_chunks
        ])

        # Create prompt
        prompt = f"""You are a helpful customer support agent for an e-commerce company.
Answer the customer's question using ONLY the information provided in the context below.
If the context doesn't contain enough information, say so.

Context:
{context}

Customer Question: {query}

Answer:"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful customer support agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    def query(self, user_query: str, k: int = 3) -> Dict:
        """Full RAG pipeline: retrieve + generate."""
        print(f"\n🔍 Query: {user_query}")
        print("="*80)

        # Retrieve
        print("📄 Retrieving relevant context...")
        retrieved_chunks = self.retrieve(user_query, k=k)

        print(f"   Found {len(retrieved_chunks)} relevant chunks:")
        for chunk in retrieved_chunks:
            print(f"   - {chunk['source']} (score: {chunk['similarity_score']:.3f})")

        # Generate
        print("\n🤖 Generating response...")
        response = self.generate_response(user_query, retrieved_chunks)

        return {
            "query": user_query,
            "response": response,
            "sources": [chunk['source'] for chunk in retrieved_chunks],
            "retrieved_chunks": retrieved_chunks
        }


def demo_baseline_rag():
    """Demonstrate baseline RAG with sample queries."""
    print("="*80)
    print("BASELINE RAG DEMO - E-commerce Customer Support")
    print("="*80)

    # Initialize and build index
    rag = BaselineRAG()
    rag.build_index()

    # Test queries
    test_queries = [
        "What is your return policy?",
        "How long does shipping take?",
        "Can I return a damaged item?",
        # This one will show the limitation
        "I want to return order #12345 because it arrived damaged. Can you process this?"
    ]

    for query in test_queries:
        result = rag.query(query)
        print(f"\n💬 Response:\n{result['response']}")
        print("\n" + "="*80)
        input("\nPress Enter to continue to next query...")

    # Show limitation
    print("\n⚠️  LIMITATION IDENTIFIED:")
    print("The last query shows the key limitation of baseline RAG:")
    print("- It retrieves the return policy (good!)")
    print("- But it CANNOT actually check if order #12345 exists")
    print("- It CANNOT verify if the item is eligible for return")
    print("- It CANNOT process the return request")
    print("\n👉 This is where Agentic RAG comes in!")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)

    demo_baseline_rag()
