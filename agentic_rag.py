"""
Agentic RAG Implementation for E-commerce Customer Support
Part 2 of Agentic RAG Masterclass

This demonstrates an agentic system that:
- Plans multi-step tasks
- Uses tools (order API, returns API, calculator)
- Retrieves policies when needed
- Cites all sources
"""

import os
import json
from typing import List, Dict, Any, Literal
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import httpx
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from baseline_rag import BaselineRAG

# ⚠️ SSL BYPASS FOR CORPORATE NETWORKS (DEMO ONLY - NOT FOR PRODUCTION)
# This disables SSL certificate verification to work with corporate proxies
# For production, configure proper SSL certificates instead
http_client = httpx.Client(verify=False)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=http_client  # SSL bypass for corporate networks
)

# Mock database for orders
ORDERS_DB = {
    "12345": {
        "order_id": "12345",
        "customer_email": "john@example.com",
        "items": [
            {"name": "Blue Cotton T-Shirt", "price": 29.99, "quantity": 2},
            {"name": "Denim Jeans", "price": 59.99, "quantity": 1}
        ],
        "total": 119.97,
        "order_date": "2025-01-10",
        "ship_date": "2025-01-11",
        "delivery_date": "2025-01-15",
        "status": "delivered",
        "tracking_number": "1Z999AA10123456784"
    },
    "67890": {
        "order_id": "67890",
        "customer_email": "jane@example.com",
        "items": [
            {"name": "Red Hoodie", "price": 45.00, "quantity": 1}
        ],
        "total": 45.00,
        "order_date": "2025-01-20",
        "ship_date": "2025-01-21",
        "delivery_date": None,
        "status": "in_transit",
        "tracking_number": "1Z999AA10987654321"
    }
}


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

class OrderStatus(BaseModel):
    """Result from order status check."""
    order_id: str
    status: str
    order_date: str
    ship_date: str | None
    delivery_date: str | None
    tracking_number: str | None
    items: List[Dict[str, Any]]
    total: float


class ReturnRequest(BaseModel):
    """Result from return processing."""
    return_id: str
    order_id: str
    refund_amount: float
    status: str
    estimated_refund_date: str


def check_order_status(order_id: str) -> Dict[str, Any]:
    """
    Check the status of an order by order ID.

    Args:
        order_id: The order ID to check

    Returns:
        Order details including status, tracking, and items
    """
    if order_id not in ORDERS_DB:
        return {
            "error": f"Order {order_id} not found",
            "source": "order_api"
        }

    order = ORDERS_DB[order_id]
    return {
        "data": order,
        "source": "order_api",
        "retrieved_at": datetime.now().isoformat()
    }


def calculate_refund(order_id: str, reason: str) -> Dict[str, Any]:
    """
    Calculate refund amount based on order and return reason.

    Args:
        order_id: The order ID
        reason: Reason for return (damaged, defective, change_of_mind)

    Returns:
        Refund calculation details
    """
    if order_id not in ORDERS_DB:
        return {"error": f"Order {order_id} not found", "source": "refund_calculator"}

    order = ORDERS_DB[order_id]
    total = order["total"]

    # Apply business rules
    if reason.lower() in ["damaged", "defective"]:
        refund_amount = total
        restocking_fee = 0.0
    else:  # change_of_mind
        restocking_fee = total * 0.15
        refund_amount = total - restocking_fee

    return {
        "data": {
            "order_id": order_id,
            "original_amount": total,
            "restocking_fee": restocking_fee,
            "refund_amount": refund_amount,
            "reason": reason
        },
        "source": "refund_calculator",
        "rule": "Full refund for damaged/defective; 15% fee for change of mind"
    }


def process_return(order_id: str, reason: str, items: str = "all") -> Dict[str, Any]:
    """
    Process a return request for an order.

    Args:
        order_id: The order ID
        reason: Reason for return
        items: Which items to return (default: all)

    Returns:
        Return request confirmation
    """
    if order_id not in ORDERS_DB:
        return {"error": f"Order {order_id} not found", "source": "returns_api"}

    order = ORDERS_DB[order_id]

    # Check if order is delivered and within 30 days
    if order["delivery_date"]:
        delivery_date = datetime.fromisoformat(order["delivery_date"])
        days_since_delivery = (datetime.now() - delivery_date).days

        if days_since_delivery > 30:
            return {
                "error": "Return window expired (30 days)",
                "source": "returns_api",
                "policy_reference": "return_policy.md#eligibility"
            }

    # Calculate refund
    refund_calc = calculate_refund(order_id, reason)
    if "error" in refund_calc:
        return refund_calc

    # Generate return ID
    return_id = f"RET-{order_id}-{datetime.now().strftime('%Y%m%d')}"
    estimated_refund = (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')

    return {
        "data": {
            "return_id": return_id,
            "order_id": order_id,
            "status": "approved",
            "refund_amount": refund_calc["data"]["refund_amount"],
            "estimated_refund_date": estimated_refund,
            "next_steps": "Return label sent to email. Ship within 5 business days."
        },
        "source": "returns_api",
        "processed_at": datetime.now().isoformat()
    }


def retrieve_policy(query: str) -> Dict[str, Any]:
    """
    Retrieve relevant policy information from knowledge base.

    Args:
        query: The policy question

    Returns:
        Relevant policy excerpts
    """
    # Use the baseline RAG for policy retrieval
    rag = BaselineRAG()
    rag.build_index()
    results = rag.retrieve(query, k=2)

    return {
        "data": {
            "query": query,
            "excerpts": [
                {"text": chunk["text"], "source": chunk["source"]}
                for chunk in results
            ]
        },
        "source": "knowledge_base",
        "retrieved_at": datetime.now().isoformat()
    }


# Tool registry for LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Check the status of an order by order ID. Use this when customer asks about order status, tracking, or delivery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to check"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate the refund amount for a return based on reason. Use this to determine how much customer will receive back.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID"},
                    "reason": {
                        "type": "string",
                        "enum": ["damaged", "defective", "change_of_mind"],
                        "description": "Reason for return"
                    }
                },
                "required": ["order_id", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "process_return",
            "description": "Process a return request. Only call this after verifying eligibility and calculating refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID"},
                    "reason": {"type": "string", "description": "Reason for return"},
                    "items": {"type": "string", "description": "Items to return", "default": "all"}
                },
                "required": ["order_id", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_policy",
            "description": "Retrieve policy information from knowledge base. Use when customer asks about policies, procedures, or general questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The policy question"}
                },
                "required": ["query"]
            }
        }
    }
]


# ============================================================================
# AGENT STATE
# ============================================================================

class AgentState(BaseModel):
    """State for the agent conversation."""
    query: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    steps: int = 0
    max_steps: int = 10
    final_response: str | None = None


# ============================================================================
# AGENT NODES
# ============================================================================

def should_continue(state: AgentState) -> Literal["tools", "respond"]:
    """Decide whether to use tools or generate final response."""
    last_message = state.messages[-1]

    # Check if we hit max steps
    if state.steps >= state.max_steps:
        return "respond"

    # Check if LLM wants to use tools
    if last_message.get("tool_calls"):
        return "tools"

    return "respond"


def call_model(state: AgentState) -> AgentState:
    """Call LLM to plan next action."""
    print(f"\n🧠 Planning (Step {state.steps + 1}/{state.max_steps})...")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an intelligent customer support agent with access to tools.

Your capabilities:
- Check order status and tracking
- Calculate refunds based on return policy
- Process return requests
- Retrieve policy information

Always:
1. Use tools to get accurate, real-time information
2. Cite sources for every fact (e.g., [Source: order_api])
3. Be helpful and professional
4. If you don't have enough information, ask clarifying questions

When processing returns:
1. First check order status
2. Then calculate refund
3. Finally process return if customer confirms

Never make up order details or policy information."""
            }
        ] + state.messages,
        tools=TOOLS,
        tool_choice="auto"
    )

    message = response.choices[0].message
    state.messages.append({
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in (message.tool_calls or [])
        ] if message.tool_calls else None
    })
    state.steps += 1

    return state


def execute_tools(state: AgentState) -> AgentState:
    """Execute tool calls from LLM."""
    last_message = state.messages[-1]
    tool_calls = last_message.get("tool_calls", [])

    if not tool_calls:
        return state

    print(f"🔧 Executing {len(tool_calls)} tool(s)...")

    tool_map = {
        "check_order_status": check_order_status,
        "calculate_refund": calculate_refund,
        "process_return": process_return,
        "retrieve_policy": retrieve_policy
    }

    for tool_call in tool_calls:
        function_name = tool_call["function"]["name"]
        function_args = json.loads(tool_call["function"]["arguments"])

        print(f"   → {function_name}({', '.join(f'{k}={v}' for k, v in function_args.items())})")

        # Execute tool
        tool_func = tool_map[function_name]
        result = tool_func(**function_args)

        # Track source
        if "source" in result:
            state.sources.append(result["source"])

        # Add tool result to messages
        state.messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "name": function_name,
            "content": json.dumps(result)
        })

        state.tool_results.append({
            "tool": function_name,
            "args": function_args,
            "result": result
        })

    return state


def generate_response(state: AgentState) -> AgentState:
    """Generate final response with citations."""
    print("\n✍️  Generating final response with citations...")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=state.messages + [
            {
                "role": "system",
                "content": """Generate a final response to the customer.

CRITICAL: Every factual claim MUST include a citation in square brackets.
Examples:
- "Your order shipped on Jan 15 [Source: order_api]"
- "Full refund for damaged items [Source: return_policy.md]"
- "Refund amount: $89.99 [Source: refund_calculator]"

If you cannot cite a source for a claim, do not make the claim.
Be helpful, professional, and complete."""
            }
        ],
        temperature=0.3
    )

    state.final_response = response.choices[0].message.content
    return state


# ============================================================================
# BUILD AGENT GRAPH
# ============================================================================

def build_agent():
    """Build the LangGraph agent."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", call_model)
    workflow.add_node("tools", execute_tools)
    workflow.add_node("responder", generate_response)

    # Add edges
    workflow.set_entry_point("planner")
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {
            "tools": "tools",
            "respond": "responder"
        }
    )
    workflow.add_edge("tools", "planner")  # Loop back to planner after tools
    workflow.add_edge("responder", END)

    return workflow.compile()


# ============================================================================
# DEMO
# ============================================================================

def demo_agentic_rag():
    """Demonstrate agentic RAG with sample queries."""
    print("="*80)
    print("AGENTIC RAG DEMO - E-commerce Customer Support")
    print("="*80)

    agent = build_agent()

    test_queries = [
        # Same query that breaks baseline RAG - now it will work!
        "I want to return order #12345 because it arrived damaged. Can you process this?",
        "Where is my order #12345?",
        "What is your return policy for defective items?",
        "Can I return order #67890? It's still in transit but I changed my mind."
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"❓ Query: {query}")
        print("="*80)

        # Initialize state
        initial_state = AgentState(
            query=query,
            messages=[{"role": "user", "content": query}]
        )

        # Run agent - returns a dict, not AgentState object
        result = agent.invoke(initial_state)

        # Display results
        print(f"\n💬 Final Response:")
        print("-"*80)
        print(result["final_response"])
        print("-"*80)

        print(f"\n📊 Agent Statistics:")
        print(f"   Steps taken: {result['steps']}")
        print(f"   Tools used: {len(result['tool_results'])}")
        print(f"   Sources cited: {list(set(result['sources']))}")

        print(f"\n🔍 Tool Execution Trace:")
        for i, tool_result in enumerate(result['tool_results'], 1):
            print(f"   {i}. {tool_result['tool']}({tool_result['args']})")
            print(f"      → {tool_result['result'].get('source', 'N/A')}")

        input("\nPress Enter to continue to next query...")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        exit(1)

    demo_agentic_rag()
