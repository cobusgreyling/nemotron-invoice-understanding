"""
Nemotron Nano 2 VL As A Multimodal Agentic Tool
================================================

Grok 3 Fast orchestrates an expense policy enforcement agent.
Nemotron Nano 2 VL reads invoice images as a LangChain tool.
Two models, division of labor: vision specialist + decision maker.

Requirements:
    pip install langchain-openai langgraph openai pyarrow pandas Pillow certifi
    export NVIDIA_API_KEY="your-key"
    export GROK_API_KEY="your-key"

Usage:
    python3 nemotron_agentic_tool_demo.py
"""

import asyncio
import io
import os
import ssl
import base64
import json

import certifi
import pyarrow.parquet as pq
import urllib.request
from PIL import Image
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Fix macOS SSL certificates
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# --- API Keys ---

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("Set NVIDIA_API_KEY env var — get one free at build.nvidia.com")

GROK_API_KEY = os.environ.get("GROK_API_KEY")
if not GROK_API_KEY:
    settings_path = os.path.expanduser("~/.grok/user-settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            GROK_API_KEY = json.load(f).get("apiKey")
if not GROK_API_KEY:
    raise ValueError("Set GROK_API_KEY env var or add apiKey to ~/.grok/user-settings.json")

# --- Nemotron Client (Vision Specialist) ---

nemotron_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

# --- Load Invoice Dataset ---

PARQUET_URL = "https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1/resolve/main/data/test-00000-of-00001-56af6bd5ff7eb34d.parquet"
PARQUET_FILE = "/tmp/claude/invoices.parquet"

print("Downloading invoice dataset from HuggingFace...")
urllib.request.urlretrieve(PARQUET_URL, PARQUET_FILE)
table = pq.read_table(PARQUET_FILE)
df = table.to_pandas()
print(f"Loaded {len(df)} invoices.\n")

INVOICE_INDICES = [6, 8, 10]


def encode_pil_to_jpeg_data_url(pil_image):
    """Convert a PIL image to a base64 JPEG data URL."""
    if pil_image.mode not in ("RGB",):
        if pil_image.mode in ("RGBA", "LA") or (
            pil_image.mode == "P" and "transparency" in getattr(pil_image, "info", {})
        ):
            background = Image.new("RGB", pil_image.size, (255, 255, 255))
            converted = pil_image.convert("RGBA")
            background.paste(converted, mask=converted.split()[-1])
            pil_image = background
        else:
            pil_image = pil_image.convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# Pre-encode all invoice images
invoice_images = {}
for idx in INVOICE_INDICES:
    img_bytes = df.loc[idx, "image"]["bytes"]
    img = Image.open(io.BytesIO(img_bytes))
    invoice_images[idx] = encode_pil_to_jpeg_data_url(img)

print(f"Pre-encoded {len(invoice_images)} invoice images (indices: {INVOICE_INDICES})\n")


# --- LangChain Tool: Nemotron Vision ---

@tool
def analyze_invoice_image(invoice_index: int, query: str) -> str:
    """Analyze an invoice image using Nemotron Nano 2 VL vision model.

    Use this tool to read and extract information from invoice images.
    Available invoice indices: 6, 8, 10.

    Args:
        invoice_index: The dataset index of the invoice to analyze (6, 8, or 10).
        query: What to extract or analyze from the invoice image.
    """
    if invoice_index not in invoice_images:
        return f"Error: invoice_index must be one of {INVOICE_INDICES}"

    image_url = invoice_images[invoice_index]

    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": query},
        ]},
    ]

    full_response = ""
    chat_response = nemotron_client.chat.completions.create(
        model="nvidia/nemotron-nano-12b-v2-vl",
        messages=messages,
        max_tokens=4096,
        temperature=0.6,
        stream=True,
    )

    for chunk in chat_response:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content

    return full_response


# --- Expense Policy ---

EXPENSE_POLICY = """
COMPANY EXPENSE POLICY
======================
1. Meal expenses must not exceed $75 per person per meal.
2. Entertainment and gaming purchases are PROHIBITED and will not be reimbursed.
3. Any single invoice above $500 must include itemized line items.
4. Office supplies and standard business materials are approved up to $5,000.
5. All invoices must include: vendor name, date, itemized list, and total amount.
"""

# --- Agent Setup (Grok 3 Fast as Orchestrator) ---

llm = ChatOpenAI(
    model="grok-3-fast",
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1",
)

agent = create_react_agent(
    model=llm,
    tools=[analyze_invoice_image],
)


# --- Run ---

SEPARATOR = "=" * 60

AGENT_PROMPT = f"""You are an expense policy enforcement agent. Your job is to review invoice images and determine whether each one complies with company policy.

{EXPENSE_POLICY}

You have access to 3 invoices (indices: 6, 8, 10). For EACH invoice:
1. Use the analyze_invoice_image tool to extract: vendor name, date, all line items with amounts, and the total.
2. Check every extracted detail against the expense policy.
3. Determine: APPROVED or REJECTED, with specific reasons.

After reviewing all 3 invoices, provide a final summary table with columns:
Invoice | Vendor | Total | Decision | Reason

Be thorough. Check every policy rule against every invoice."""


async def run_demo():
    print(SEPARATOR)
    print("  Nemotron As Multimodal Agentic Tool")
    print("  Orchestrator: Grok 3 Fast  |  Vision: Nemotron Nano 2 VL")
    print(SEPARATOR)
    print(f"\n[EXPENSE POLICY]{EXPENSE_POLICY}")
    print(f"[AGENT TASK] Review 3 invoices against expense policy\n")
    print(SEPARATOR)

    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": AGENT_PROMPT}]},
    )

    # Print the full agent trace
    for msg in response["messages"]:
        role = getattr(msg, "type", "unknown")
        if role == "human":
            continue  # Skip the long prompt
        elif role == "ai":
            if msg.content:
                print(f"\n[AGENT]\n{msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n[TOOL CALL] {tc['name']}(invoice_index={tc['args'].get('invoice_index')}, query=\"{tc['args'].get('query', '')[:80]}...\")")
        elif role == "tool":
            print(f"\n[NEMOTRON RESPONSE]\n{msg.content[:500]}...")

    # Print the final answer
    last_message = response["messages"][-1]
    print(f"\n{SEPARATOR}")
    print("  FINAL DECISION")
    print(SEPARATOR)
    print(last_message.content)
    print(SEPARATOR)


if __name__ == "__main__":
    asyncio.run(run_demo())
