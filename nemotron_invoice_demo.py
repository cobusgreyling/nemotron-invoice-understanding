"""
Nemotron Nano 2 VL — Invoice Understanding Demo
=================================================
Runs two invoice analysis queries using NVIDIA's Nemotron model:
1. Sum up all totals across receipts
2. Detect potential duplicate invoices
"""

import pyarrow.parquet as pq
import io
import base64
import os
import ssl
import certifi
import urllib.request
from PIL import Image
from openai import OpenAI

# Fix macOS SSL certificates
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# --- Setup ---
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "YOUR_NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)


def encode_pil_to_jpeg_data_url(pil_image):
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


def call_nemotron_nano_2_vl(image_urls, text_prompt, reasoning_mode=True, temperature=0.6, max_tokens=32000):
    if reasoning_mode:
        reasoning_token = "/think"
    else:
        reasoning_token = "/no_think"

    messages = [
        {"role": "system", "content": reasoning_token},
        {"role": "user", "content": []},
    ]

    for image_url in image_urls:
        messages[1]["content"].append(
            {"type": "image_url", "image_url": {"url": image_url}}
        )

    messages[1]["content"].append({"type": "text", "text": text_prompt})

    full_response = ""
    reasoning_text = ""

    chat_response = client.chat.completions.create(
        model="nvidia/nemotron-nano-12b-v2-vl",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    for chunk in chat_response:
        reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
        if reasoning:
            reasoning_text += reasoning
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content

    return reasoning_text, full_response


# --- Download dataset ---
PARQUET_URL = "https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1/resolve/main/data/test-00000-of-00001-56af6bd5ff7eb34d.parquet"
PARQUET_FILE = "/tmp/claude/invoices.parquet"

print("Downloading invoice dataset from HuggingFace...")
urllib.request.urlretrieve(PARQUET_URL, PARQUET_FILE)

table = pq.read_table(PARQUET_FILE)
df = table.to_pandas()
print(f"Loaded {len(df)} invoices.\n")

# --- Load 4 invoice images ---
invoice_indices = [6, 8, 10, 12]
images = []
image_urls = []

for idx in invoice_indices:
    img_bytes = df.loc[idx, "image"]["bytes"]
    img = Image.open(io.BytesIO(img_bytes))
    images.append(img)
    image_urls.append(encode_pil_to_jpeg_data_url(img))

print(f"Loaded {len(images)} invoice images (indices: {invoice_indices})\n")

# --- Query 1: Sum totals ---
SEPARATOR = "=" * 60
print(SEPARATOR)
print("  QUERY 1: Sum up all the totals across the receipts")
print(SEPARATOR)

reasoning1, response1 = call_nemotron_nano_2_vl(
    image_urls=image_urls,
    text_prompt="Sum up all the totals across the reciepts.",
    reasoning_mode=True,
    temperature=0.6,
    max_tokens=32000,
)

print("\n[REASONING]")
print(reasoning1)
print("\n[ANSWER]")
print(response1)

# --- Query 2: Duplicate detection ---
print(f"\n{SEPARATOR}")
print("  QUERY 2: Duplicate invoice detection")
print(SEPARATOR)

reasoning2, response2 = call_nemotron_nano_2_vl(
    image_urls=image_urls,
    text_prompt="Here are 4 invoices flagged as potential duplicates — are they actually the same document with minor layout differences?",
    reasoning_mode=True,
    temperature=0.6,
    max_tokens=32000,
)

print("\n[REASONING]")
print(reasoning2)
print("\n[ANSWER]")
print(response2)

print(f"\n{SEPARATOR}")
print("  Done.")
print(SEPARATOR)
