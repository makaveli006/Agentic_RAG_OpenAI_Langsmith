# Agentic RAG Masterclass - Code Package

This folder contains all the code needed to run the 3-part Agentic RAG masterclass.

---

## 📦 What's Included

### Jupyter Notebooks (Interactive - Recommended)
- `1_baseline_rag.ipynb` - Part 1: Baseline RAG system
- `2_agentic_rag.ipynb` - Part 2: Agentic RAG with tools
- `3_production_ready.ipynb` - Part 3: Production features

### Python Scripts (Standalone)
- `baseline_rag.py` - Part 1 script version
- `agentic_rag.py` - Part 2 script version
- `production_agent.py` - Part 3 script version

### Configuration
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template

---

## 🚀 Quick Start

### Step 1: Setup Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate        # Mac/Linux
# OR
venv\Scripts\activate           # Windows
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure OpenAI API Key

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Replace 'sk-your-api-key-here' with your actual key
```

### Step 4: Run the Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. 1_baseline_rag.ipynb
# 2. 2_agentic_rag.ipynb
# 3. 3_production_ready.ipynb
```

---

## 📚 Masterclass Structure

### Part 1: Baseline RAG (15 minutes)
**File**: `1_baseline_rag.ipynb`

**What You'll Learn**:
- Document chunking and embedding
- FAISS vector store setup
- Basic retrieval pipeline
- Why baseline RAG fails for action-oriented tasks

**Key Demo**: Can retrieve policy but can't actually check order status

---

### Part 2: Agentic RAG (20 minutes)
**File**: `2_agentic_rag.ipynb`

**What You'll Learn**:
- Tool definitions and function calling
- LangGraph agent architecture
- Multi-step planning with ReAct pattern
- Source citation tracking

**Key Demo**: Actually checks orders, calculates refunds, processes returns

---

### Part 3: Production Ready (10 minutes)
**File**: `3_production_ready.ipynb`

**What You'll Learn**:
- Multi-level caching (saves 50-80% costs)
- Comprehensive metrics and observability
- Safety guardrails (PII masking, cost limits)
- FastAPI deployment wrapper

**Key Demo**: Cost savings analysis, production API

---

## 💻 Alternative: Run Python Scripts

If you prefer running scripts instead of notebooks:

```bash
# Part 1: Baseline RAG
python baseline_rag.py

# Part 2: Agentic RAG
python agentic_rag.py

# Part 3: Production Agent (Demo)
python production_agent.py

# Part 3: Production Agent (API Server)
python production_agent.py --api
# Then visit http://localhost:8000/docs
```

---

## 📋 Requirements

- **Python**: 3.11 or higher
- **OpenAI API Key**: Required (get one at https://platform.openai.com/api-keys)
- **Dependencies**: Listed in `requirements.txt`

### Key Dependencies
- `jupyter` - Interactive notebooks
- `openai` - OpenAI API client
- `langgraph` - Agent framework
- `faiss-cpu` - Vector store
- `pydantic` - Data validation
- `fastapi` - API framework (Part 3)
- `httpx` - HTTP client (for SSL bypass if needed)

---

## 🔧 Troubleshooting

### Issue: SSL Certificate Error

If you see SSL certificate errors (common on corporate networks):

**Already Fixed**: The code includes SSL bypass for demo purposes using:
```python
http_client = httpx.Client(verify=False)
```

⚠️ **Note**: This is for demo/testing only. For production, configure proper SSL certificates.

---

### Issue: "Module not found" Error

**Solution**: Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

---

### Issue: "Invalid API Key" Error

**Solution**: Check your `.env` file:
```bash
# Make sure .env exists
ls -la .env

# Make sure it contains your actual API key
cat .env
# Should show: OPENAI_API_KEY=sk-...
```

---

### Issue: LangGraph AttributeError

If you see `'dict' object has no attribute 'final_response'`:

**Already Fixed**: The notebooks access results as dictionaries:
```python
result = agent.invoke(state)
print(result["final_response"])  # Correct
```

---

## 📊 What Each Part Demonstrates

### Part 1: The Problem
```
User: "Where is my order #12345?"
Baseline RAG: "According to our shipping policy, orders arrive in 5-7 days..."
❌ Generic, doesn't check actual order
```

### Part 2: The Solution
```
User: "Where is my order #12345?"
Agentic RAG:
  1. Calls check_order_status(12345)
  2. Finds: Shipped Jan 15, delivered Jan 17
  3. Responds: "Your order #12345 shipped on Jan 15 and was delivered on Jan 17 [Source: order_api]"
✅ Specific, verified, cited
```

### Part 3: Production Ready
```
Feature                 Before      After
Caching                 ❌          ✅ Multi-level (exact + semantic)
Metrics                 ❌          ✅ Latency, cost, quality
Safety                  ❌          ✅ PII masking, limits
API                     ❌          ✅ FastAPI wrapper
Cost (1K queries/day)   $50/day     $20/day (60% savings)
```

---

## 🎯 Expected Outputs

### When Running Part 1
```
📚 Chunking documents...
   Created 12 chunks
🔢 Generating embeddings...
   Processing chunk 1/12
   ...
✅ Index built with 12 vectors

🔍 Query: What is your return policy?
📄 Retrieving relevant context...
   Found 3 relevant chunks:
   - return_policy.md (score: 0.892)
   - return_policy.md (score: 0.856)
   - shipping_policy.md (score: 0.723)

🤖 Generating response...
💬 Response:
Items can be returned within 30 days of delivery...
```

### When Running Part 2
```
❓ Query: I want to return order #12345 because it's damaged. How much refund?

🧠 Planning (Step 1/10)...
🔧 Executing 2 tool(s)...
   → check_order_status(order_id=12345)
   → calculate_refund(order_id=12345, reason=damaged)

✍️  Generating final response with citations...

💬 Final Response:
Your order #12345 is eligible for return. You'll receive a full refund of $119.97
[Source: order_api]. Damaged items qualify for full refunds within 30 days
[Source: return_policy.md]. A return label has been sent to your email.

📊 Agent Statistics:
   Steps taken: 3
   Tools used: 2
   Sources cited: ['order_api', 'return_policy.md']
```

### When Running Part 3
```
Query 1: "Where is my order #12345?"
Status: CACHE MISS
Cost: $0.05
Latency: 3.2s

Query 2: "Where is my order #12345?" (exact repeat)
💾 [CACHE HIT - EXACT] Saved $0.0500
Cost: $0.00
Latency: 0.02s

Query 3: "What's the status of order 12345?" (paraphrase)
💾 [CACHE HIT - SEMANTIC] Similarity: 0.970, Saved $0.0500
Cost: ~$0.0001
Latency: 0.15s

================================================================================
METRICS SUMMARY
================================================================================
{
  "total_queries": 3,
  "avg_latency_sec": 1.124,
  "avg_cost_usd": 0.0167,
  "cache_hit_rate": 0.667,
  "total_cost_saved": 0.10
}
```

---

## 🎓 Learning Objectives

By the end of this masterclass, you'll be able to:

1. ✅ Explain why baseline RAG fails for action-oriented tasks
2. ✅ Build a complete RAG system with FAISS and OpenAI
3. ✅ Transform RAG into an agentic system with LangGraph
4. ✅ Implement tool calling and multi-step planning
5. ✅ Add production features: caching, metrics, safety
6. ✅ Deploy an agent as a REST API with FastAPI
7. ✅ Optimize costs with semantic caching (50-80% savings)

---

## 📖 Additional Resources

### Documentation
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Related Topics
- ReAct Pattern: https://arxiv.org/abs/2210.03629
- RAG Evaluation: https://github.com/explodinggradients/ragas
- LLM Observability: https://docs.smith.langchain.com/

---

## 💡 Tips for Presenters

1. **Run notebooks sequentially** - Each part builds on the previous
2. **Show the outputs** - The visual feedback is important
3. **Compare Part 1 vs Part 2** - Highlight the limitations and solutions
4. **Demo the cache** - Run same query twice to show cost savings
5. **Show metrics dashboard** - Real-time observability is impressive
6. **Keep API key secret** - Use environment variables, never hardcode

---

## 🤝 Support

If you encounter any issues:

1. Check the Troubleshooting section above
2. Verify all dependencies are installed
3. Ensure Python version is 3.11+
4. Confirm OpenAI API key is valid

---

## 📝 License

This code is provided for educational purposes as part of the Agentic RAG Masterclass.

---

**Ready to build production-grade agentic RAG systems? Let's get started!** 🚀

Run `jupyter notebook` and open `1_baseline_rag.ipynb` to begin.
