# NVIDIA Nemotron For Invoice Understanding

A practical example of using NVIDIA's Nemotron Nano 2 VL vision language model for invoice understanding, internal audit and duplicate detection.

## Files

- **`nemotron-invoice-understanding-blog.md`** — Blog post covering the exercise
- **`nemotron_invoice_demo.py`** — Working demo that runs two invoice analysis queries

## What It Does

1. **Sums totals across invoices** — The model reads raw invoice images and extracts financial figures with step-by-step reasoning
2. **Detects duplicate invoices** — Compares invoices across dates, sellers, items and amounts to identify genuine duplicates

## Quick Start

```bash
pip install pyarrow pandas openai Pillow certifi
export NVIDIA_API_KEY="your-key-here"
python3 nemotron_invoice_demo.py
```

Get a free NVIDIA API key at [build.nvidia.com](https://build.nvidia.com).

## Architecture

```
Invoice Images (from HuggingFace dataset)
        │
        ▼
┌──────────────────────────┐
│   Nemotron Nano 2 VL     │
│   (12B Vision LLM)       │
│                          │
│   /think mode ON         │
│   Reasoning visible      │
│   Step-by-step audit     │
└──────────┬───────────────┘
           │
           ▼
   Structured answers with
   full reasoning chain
```

Nemotron can run via NVIDIA API or locally on NVIDIA hardware for full data sovereignty.
