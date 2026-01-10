"""
Production-Ready Agentic RAG Implementation
Part 3 of Agentic RAG Masterclass

This demonstrates production features:
- Multi-level caching (exact + semantic)
- Comprehensive metrics and tracing
- Safety guardrails (PII, cost limits, step limits)
- FastAPI deployment wrapper
"""

import os
import json
import time
import hashlib
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import httpx
from openai import OpenAI
from pydantic import BaseModel, Field
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ⚠️ SSL BYPASS FOR CORPORATE NETWORKS (DEMO ONLY - NOT FOR PRODUCTION)
http_client = httpx.Client(verify=False)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=http_client
)


# ============================================================================
# CACHING LAYER
# ============================================================================

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    query: str
    response: str
    embedding: np.ndarray
    timestamp: datetime
    hits: int = 0
    cost_saved: float = 0.0


class SemanticCache:
    """Multi-level caching for agent responses."""

    def __init__(self, similarity_threshold: float = 0.95, ttl_hours: int = 24):
        self.exact_cache: Dict[str, CacheEntry] = {}
        self.semantic_entries: List[CacheEntry] = []
        self.similarity_threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)
        self.stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "total_cost_saved": 0.0
        }

    def _hash_query(self, query: str) -> str:
        """Generate hash for exact match."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() - entry.timestamp > self.ttl

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for semantic similarity."""
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"  # Cheaper for cache lookups
        )
        return np.array(response.data[0].embedding)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get(self, query: str, estimated_cost: float = 0.05) -> Optional[str]:
        """Try to retrieve from cache."""
        # Exact match
        query_hash = self._hash_query(query)
        if query_hash in self.exact_cache:
            entry = self.exact_cache[query_hash]
            if not self._is_expired(entry):
                entry.hits += 1
                entry.cost_saved += estimated_cost
                self.stats["exact_hits"] += 1
                self.stats["total_cost_saved"] += estimated_cost
                print(f"💾 [CACHE HIT - EXACT] Saved ${estimated_cost:.4f}")
                return entry.response

        # Semantic similarity
        query_embedding = self._get_embedding(query)
        for entry in self.semantic_entries:
            if self._is_expired(entry):
                continue

            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            if similarity >= self.similarity_threshold:
                entry.hits += 1
                entry.cost_saved += estimated_cost
                self.stats["semantic_hits"] += 1
                self.stats["total_cost_saved"] += estimated_cost
                print(f"💾 [CACHE HIT - SEMANTIC] Similarity: {similarity:.3f}, Saved ${estimated_cost:.4f}")
                return entry.response

        self.stats["misses"] += 1
        return None

    def set(self, query: str, response: str):
        """Store in cache."""
        query_hash = self._hash_query(query)
        embedding = self._get_embedding(query)

        entry = CacheEntry(
            query=query,
            response=response,
            embedding=embedding,
            timestamp=datetime.now()
        )

        self.exact_cache[query_hash] = entry
        self.semantic_entries.append(entry)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_queries = sum([self.stats["exact_hits"], self.stats["semantic_hits"], self.stats["misses"]])
        hit_rate = (self.stats["exact_hits"] + self.stats["semantic_hits"]) / total_queries if total_queries > 0 else 0

        return {
            **self.stats,
            "total_queries": total_queries,
            "hit_rate": hit_rate,
            "cache_size": len(self.exact_cache)
        }


# ============================================================================
# METRICS AND OBSERVABILITY
# ============================================================================

class AgentMetrics:
    """Track and log agent execution metrics."""

    def __init__(self):
        self.query_log: List[Dict[str, Any]] = []
        self.tool_latencies = defaultdict(list)
        self.tool_errors = defaultdict(int)

    def log_query(self, query_data: Dict[str, Any]):
        """Log a complete query execution."""
        self.query_log.append({
            **query_data,
            "timestamp": datetime.now().isoformat()
        })

    def log_tool_call(self, tool_name: str, latency: float, success: bool):
        """Log individual tool execution."""
        self.tool_latencies[tool_name].append(latency)
        if not success:
            self.tool_errors[tool_name] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated metrics."""
        if not self.query_log:
            return {"message": "No queries logged yet"}

        total_queries = len(self.query_log)
        avg_latency = np.mean([q.get("total_latency", 0) for q in self.query_log])
        avg_cost = np.mean([q.get("estimated_cost", 0) for q in self.query_log])
        avg_steps = np.mean([q.get("steps", 0) for q in self.query_log])

        tool_stats = {}
        for tool, latencies in self.tool_latencies.items():
            tool_stats[tool] = {
                "calls": len(latencies),
                "avg_latency_ms": np.mean(latencies) * 1000,
                "p95_latency_ms": np.percentile(latencies, 95) * 1000 if len(latencies) > 1 else latencies[0] * 1000,
                "errors": self.tool_errors.get(tool, 0)
            }

        return {
            "total_queries": total_queries,
            "avg_latency_sec": round(avg_latency, 3),
            "avg_cost_usd": round(avg_cost, 4),
            "avg_steps": round(avg_steps, 2),
            "tool_stats": tool_stats
        }

    def print_trace(self, query_index: int = -1):
        """Pretty print execution trace for a query."""
        if not self.query_log:
            print("No queries to trace")
            return

        query = self.query_log[query_index]

        print("\n" + "="*80)
        print("EXECUTION TRACE")
        print("="*80)
        print(f"Query: {query.get('query', 'N/A')}")
        print(f"Timestamp: {query.get('timestamp', 'N/A')}")
        print(f"Total Latency: {query.get('total_latency', 0):.3f}s")
        print(f"Estimated Cost: ${query.get('estimated_cost', 0):.4f}")
        print(f"Steps: {query.get('steps', 0)}")
        print(f"Citations: {query.get('citation_count', 0)}")

        if "tool_calls" in query:
            print(f"\nTool Calls ({len(query['tool_calls'])})")
            for i, tool_call in enumerate(query['tool_calls'], 1):
                print(f"  {i}. {tool_call['name']}({tool_call.get('args', {})})")
                print(f"     → Latency: {tool_call.get('latency', 0):.3f}s")
                print(f"     → Status: {'✅ Success' if tool_call.get('success', True) else '❌ Failed'}")

        print("="*80)


# ============================================================================
# SAFETY GUARDRAILS
# ============================================================================

class SafetyGuardrails:
    """Production safety controls."""

    def __init__(self, max_steps: int = 10, max_cost: float = 1.0):
        self.max_steps = max_steps
        self.max_cost = max_cost
        self.current_cost = 0.0
        self.pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
            (r'\b\d{16}\b', '[CARD]'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
        ]

    def check_step_limit(self, current_step: int) -> bool:
        """Prevent infinite loops."""
        if current_step >= self.max_steps:
            raise ValueError(f"🛑 Safety: Step limit reached ({self.max_steps}). Possible infinite loop.")
        return True

    def check_cost_limit(self, additional_cost: float) -> bool:
        """Prevent cost explosions."""
        self.current_cost += additional_cost
        if self.current_cost >= self.max_cost:
            raise ValueError(f"🛑 Safety: Cost limit reached (${self.max_cost}). Current: ${self.current_cost:.4f}")
        return True

    def mask_pii(self, text: str) -> str:
        """Mask personally identifiable information."""
        masked = text
        for pattern, replacement in self.pii_patterns:
            masked = re.sub(pattern, replacement, masked)
        return masked

    def validate_citations(self, response: str, sources: List[str]) -> bool:
        """Ensure all factual claims have citations."""
        citation_count = len(re.findall(r'\[Source: [^\]]+\]', response))

        if citation_count < len(set(sources)):
            print(f"⚠️  Warning: Low citation coverage ({citation_count} citations for {len(set(sources))} sources)")
            return False

        return True

    def reset_cost(self):
        """Reset cost counter (call per query)."""
        self.current_cost = 0.0


# ============================================================================
# PRODUCTION AGENT
# ============================================================================

# Initialize components
cache = SemanticCache(similarity_threshold=0.95, ttl_hours=24)
metrics = AgentMetrics()
guardrails = SafetyGuardrails(max_steps=10, max_cost=1.0)


def production_agent_query(query: str) -> Dict[str, Any]:
    """
    Production-ready agent with caching, metrics, and safety.
    """
    from agentic_rag import build_agent, AgentState

    start_time = time.time()
    guardrails.reset_cost()

    # Check cache
    cached_response = cache.get(query, estimated_cost=0.05)
    if cached_response:
        end_time = time.time()
        return {
            "query": query,
            "response": cached_response,
            "from_cache": True,
            "latency": end_time - start_time,
            "cost": 0.0
        }

    # Execute agent
    print(f"\n{'='*80}")
    print(f"❓ Query: {query}")
    print("="*80)

    agent = build_agent()
    initial_state = AgentState(
        query=query,
        messages=[{"role": "user", "content": query}]
    )

    try:
        result = agent.invoke(initial_state)

        # Safety validation
        guardrails.check_step_limit(result["steps"])
        guardrails.validate_citations(result["final_response"], result["sources"])

        # Mask PII
        safe_response = guardrails.mask_pii(result["final_response"])

        # Log metrics
        end_time = time.time()
        total_latency = end_time - start_time
        estimated_cost = 0.001 * result["steps"] + 0.01
        citation_count = len(re.findall(r'\[Source: [^\]]+\]', result["final_response"]))

        query_data = {
            "query": query,
            "steps": result["steps"],
            "total_latency": total_latency,
            "estimated_cost": estimated_cost,
            "citation_count": citation_count,
            "tool_calls": [
                {
                    "name": tr["tool"],
                    "args": tr["args"],
                    "success": "error" not in tr["result"],
                    "latency": 0.1
                }
                for tr in result["tool_results"]
            ]
        }
        metrics.log_query(query_data)

        # Store in cache
        cache.set(query, safe_response)

        # Display results
        print(f"\n💬 Final Response:")
        print("-"*80)
        print(safe_response)
        print("-"*80)

        print(f"\n📊 Execution Stats:")
        print(f"   Latency: {total_latency:.3f}s")
        print(f"   Est. Cost: ${estimated_cost:.4f}")
        print(f"   Steps: {result['steps']}")
        print(f"   Tools Used: {len(result['tool_results'])}")
        print(f"   Citations: {citation_count}")

        return {
            "query": query,
            "response": safe_response,
            "from_cache": False,
            "latency": total_latency,
            "cost": estimated_cost,
            "steps": result["steps"],
            "citations": citation_count
        }

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return {
            "query": query,
            "error": str(e),
            "from_cache": False
        }


# ============================================================================
# FASTAPI DEPLOYMENT
# ============================================================================

app = FastAPI(title="Agentic RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    response: str
    from_cache: bool
    latency_sec: float
    cost_usd: float
    metadata: Dict[str, Any]


@app.post("/v1/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Query the agentic RAG system.

    - **query**: The user's question
    - **user_id**: Optional user identifier for rate limiting
    """
    try:
        result = production_agent_query(request.query)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return QueryResponse(
            query=result["query"],
            response=result["response"],
            from_cache=result["from_cache"],
            latency_sec=result["latency"],
            cost_usd=result.get("cost", 0.0),
            metadata={
                "steps": result.get("steps", 0),
                "citations": result.get("citations", 0)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/metrics")
async def get_metrics():
    """Get agent performance metrics."""
    return JSONResponse(content={
        "agent": metrics.get_summary(),
        "cache": cache.get_stats()
    })


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============================================================================
# DEMO
# ============================================================================

def demo_production():
    """Demonstrate production features."""
    print("="*80)
    print("PRODUCTION-READY AGENTIC RAG DEMO")
    print("="*80)

    test_queries = [
        "Where is my order #12345?",
        "Where is my order #12345?",  # Exact cache hit
        "What's the status of order 12345?",  # Semantic cache hit
        "I want to return order #12345 because it's damaged. How much refund?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(test_queries)}")
        print("="*80)

        result = production_agent_query(query)

        if i < len(test_queries):
            input("\nPress Enter to continue...")

    # Show metrics
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)

    print("\nAgent Performance:")
    print(json.dumps(metrics.get_summary(), indent=2))

    print("\nCache Performance:")
    print(json.dumps(cache.get_stats(), indent=2))

    metrics.print_trace(-1)


if __name__ == "__main__":
    import sys

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        exit(1)

    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        # Run FastAPI server
        import uvicorn
        print("\n🚀 Starting production API server...")
        print("   Docs: http://localhost:8000/docs")
        print("   Metrics: http://localhost:8000/v1/metrics")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run demo
        demo_production()
