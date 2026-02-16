# NVIDIA Nemotron For Invoice Understanding

A practical example of using NVIDIA's Nemotron Nano 2 VL vision language model for invoice understanding, internal audit and duplicate detection — including an agentic architecture where Nemotron acts as a vision tool inside a LangGraph ReAct agent.

## Files

### Demo 1: Direct API — Invoice Understanding
- **`nemotron-invoice-understanding-blog.md`** — Blog post covering the exercise
- **`nemotron_invoice_demo.py`** — Working demo that runs two invoice analysis queries

### Demo 2: Agentic Tool — Expense Policy Enforcement
- **`nemotron-agentic-tool-blog.md`** — Blog post on the agentic architecture
- **`nemotron_agentic_tool_demo.py`** — Grok 3 Fast agent with Nemotron as a vision tool

## What It Does

### Demo 1: Direct API
1. **Sums totals across invoices** — The model reads raw invoice images and extracts financial figures with step-by-step reasoning
2. **Detects duplicate invoices** — Compares invoices across dates, sellers, items and amounts to identify genuine duplicates

### Demo 2: Agentic Tool
3. **Expense policy enforcement** — A Grok 3 Fast agent reviews invoices against company policy, using Nemotron to read each document and then applying business rules to approve or reject

## Quick Start

### Demo 1: Direct API
```bash
pip install pyarrow pandas openai Pillow certifi
export NVIDIA_API_KEY="your-key-here"
python3 nemotron_invoice_demo.py
```

### Demo 2: Agentic Tool
```bash
pip install langchain-openai langgraph openai pyarrow pandas Pillow certifi
export NVIDIA_API_KEY="your-key-here"
export GROK_API_KEY="your-key-here"
python3 nemotron_agentic_tool_demo.py
```

Get a free NVIDIA API key at [build.nvidia.com](https://build.nvidia.com).

## Architecture

### Demo 1: Direct API
```
Invoice Images (from HuggingFace dataset)
        |
        v
+---------------------------+
|   Nemotron Nano 2 VL      |
|   (12B Vision LLM)        |
|                            |
|   /think mode ON           |
|   Reasoning visible        |
|   Step-by-step audit       |
+-------------+-------------+
              |
              v
   Structured answers with
   full reasoning chain
```

### Demo 2: Agentic Tool
```
User: "Review these invoices against our expense policy"
                    |
                    v
         +---------------------+
         |   Grok 3 Fast       |  <-- Orchestrator
         |   (LangGraph Agent) |
         +----------+----------+
                    | calls tool per invoice
                    v
         +---------------------+
         |  Nemotron Nano 2 VL |  <-- Vision specialist
         |  (LangChain @tool)  |
         +----------+----------+
                    |
                    v
         Agent chains vision output
         -> policy check -> approve/reject
```

Nemotron can run via NVIDIA API or locally on NVIDIA hardware for full data sovereignty.
