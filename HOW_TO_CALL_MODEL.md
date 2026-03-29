# How to Call Typhoon Model for Data Generation

## 📞 Step-by-Step Model Calling Process

### Step 1: Load the Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Download & load model from HuggingFace
model_name = "typhoon-ai/llama3.2-typhoon2-3b"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load model with smart device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,      # Memory efficient
    device_map="auto"                # GPU with CPU offload if needed
)
```

**What happens:**
- Downloads Typhoon 3B from HuggingFace (first time only)
- Caches it locally (~6-7 GB)
- Loads into CUDA GPU memory
- `device_map="auto"` handles out-of-memory by offloading to CPU

---

### Step 2: Build the Prompt

**For QA Task:**
```python
def _build_prompt(task="qa", label="qa_mode", seed_user="กองทุนรวมคืออะไร"):
    return f"""You are generating Thai-language synthetic Q&A data.
IMPORTANT: Output ONLY one JSON object. No markdown. No explanations.

Rules:
- Use Thai language only.
- task_type must be: qa
- label must be: qa_mode
- user: the Thai question
- assistant: a DIRECT ANSWER or SUMMARY in Thai
- DO NOT respond with questions! Give explanatory statements instead.

Seed input: {seed_user}
Generate a Thai question and provide a clear, factual Thai answer.

Example format:
  user: 'บอกความหมายของคำว่า...' → assistant: '[Topic] คือ...'
NOT: user: '...' → assistant: '[Topic] คืออะไร? [questions]?'

JSON:"""

prompt = _build_prompt("qa", "qa_mode", "กองทุนรวมคืออะไร")
```

**For CHAT Task:**
```python
prompt = """You are generating Thai-language synthetic training data for a chatbot.
IMPORTANT: Output ONLY one JSON object. No markdown.

Rules:
- Use Thai language
- task_type can be: route, chitchat
- label must be either: chat_mode or qa_mode

Seed user text: สวัสดี

JSON:"""
```

**Output:** A template text that guides the model on what to generate

---

### Step 3: Call model.generate()

**The actual model invocation:**
```python
def _generate_one(model, tokenizer, prompt, config):
    """
    Call the LLM to generate one example
    """
    # 1. Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 2. Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 3. Generate output
    output = model.generate(
        **inputs,
        max_new_tokens=96,           # ← Limit output length
        do_sample=True,              # ← Use sampling (not greedy)
        temperature=0.7,             # ← Balance creativity vs consistency
        top_p=0.9,                   # ← Nucleus sampling (quality)
        repetition_penalty=1.05,     # ← Avoid repetition
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # 4. Decode back to text
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 5. Extract JSON part only
    idx = decoded.rfind("JSON:")
    if idx >= 0:
        decoded = decoded[idx + len("JSON:"):]
    
    return decoded.strip()

# Call it!
completion = _generate_one(model, tokenizer, prompt, config)
# Result: '{"task_type": "qa", "user": "กองทุนรวมคืออะไร", "assistant": "...", ...}'
```

---

## 🎯 Key Parameters Explained

| Parameter | Value | What it does |
|-----------|-------|------------|
| **max_new_tokens** | 96 | Max length of generated text (1.7x faster than 192) |
| **temperature** | 0.7 | Randomness: 0=deterministic, 1.0=very random |
| **top_p** | 0.9 | Nucleus sampling: keep top 90% probability tokens |
| **do_sample** | True | Use sampling instead of greedy decoding |
| **repetition_penalty** | 1.05 | Reduce chance of repeating same phrases |
| **torch_dtype** | float16 | 16-bit precision (vs 32-bit) = 2x memory efficient |
| **device_map** | "auto" | Automatically use GPU+CPU if needed |

---

## 📊 Complete Generation Loop

```python
# From generate_typhoon.py main()

with open(output_file, "w") as fout:
    for pool in pools:  # pools = [route_seeds, chitchat_seeds, qa_seeds]
        for i in range(n_examples):  # Generate n examples per task
            
            # Get a seed (e.g., "กองทุนรวมคืออะไร")
            task_type, label, style, seed_user = pool[i % len(pool)]
            
            # Build prompt
            prompt = _build_prompt(task_type, label, style, seed_user)
            
            # Try up to 3 times (retries)
            for attempt in range(retries):
                
                # ===== CALL THE MODEL =====
                completion = _generate_one(model, tokenizer, prompt, config)
                
                # Extract JSON from completion
                obj = _extract_json(completion)
                
                # Validate structure
                ex = _ensure_fields(obj)
                if not ex:
                    continue
                
                # Validate Thai language (≥70% Thai chars)
                if not _thai_gate(ex, min_thai_ratio=0.70):
                    continue
                
                # ✅ PASS! Write to file
                output_line = json.dumps(ex, ensure_ascii=False)
                fout.write(output_line + "\n")
                written += 1
                break
            
            # Fallback: if generation fails, use seed as example
            if attempt == retries - 1:
                fallback = {
                    "id": uuid.uuid4(),
                    "task_type": task_type,
                    "user": seed_user,
                    "assistant": "seed text",
                    "source": "seed_fallback"
                }
                fout.write(json.dumps(fallback) + "\n")
```

---

## 🔄 Full Flow Diagram

```
INPUT
  ↓
[Seed: "กองทุนรวมคืออะไร"]
  ↓
BUILD PROMPT
  "You are generating Thai Q&A data...
   Seed input: กองทุนรวมคืออะไร
   JSON:"
  ↓
TOKENIZE
  [input_ids, attention_mask] → Move to GPU
  ↓
MODEL.GENERATE()  ← ⭐ THE MODEL CALL
  ├─ temperature=0.7 (creative but consistent)
  ├─ max_new_tokens=96 (short & fast)
  ├─ top_p=0.9 (quality sampling)
  └─ do_sample=True (randomness)
  ↓
OUTPUT (tokens)
  [1503, 2948, 1859, ...]
  ↓
DECODE
  "{"task_type": "qa", "user": "กองทุนรวมคืออะไร", 
    "assistant": "กองทุนรวม คือ เครื่องมือการลงทุน...", ...}"
  ↓
EXTRACT JSON
  {"task_type": "qa", "user": "...", "assistant": "..."}
  ↓
VALIDATE
  ✓ Thai chars ≥70%?
  ✓ Has required fields?
  ✓ Valid task_type?
  ↓
OUTPUT
  Write to raw.jsonl
```

---

## 🎮 How to Run It

**Command Line:**
```bash
python synthetic/generate_typhoon.py \
  --model typhoon-ai/llama3.2-typhoon2-3b \
  --device cuda \
  --dtype float16 \
  --out ./data/thai_synth/raw.jsonl \
  --n 20 \
  --task all \
  --max-new-tokens 96 \
  --min-thai-ratio 0.70
```

**Or Programmatically:**
```python
from synthetic.generate_typhoon import main

# Set arguments and run
import sys
sys.argv = [
    "generate_typhoon.py",
    "--model", "typhoon-ai/llama3.2-typhoon2-3b",
    "--device", "cuda",
    "--dtype", "float16",
    "--out", "./data/thai_synth/raw.jsonl",
    "--n", "20",
    "--task", "all",
    "--max-new-tokens", "96",
]

main()
```

---

## 💡 Key Points About Model Calling

### 1. **Tokenization** (Text → Numbers)
```python
inputs = tokenizer(prompt, return_tensors="pt")
# "You are generating..." → [101, 2054, 2003, ...]
```

### 2. **Generation** (Numbers → More Numbers)
```python
output_ids = model.generate(input_ids, max_new_tokens=96)
# [101, 2054, ...] + [generated tokens]
```

### 3. **Decoding** (Numbers → Text)
```python
text = tokenizer.decode(output_ids)
# [output tokens] → "{"task_type": "qa", ...}"
```

### 4. **Validation** (Check Quality)
```python
obj = json.loads(text)  # Parse JSON
if thai_char_ratio(obj["assistant"]) >= 0.70:  # Check Thai
    save(obj)  # Keep it
```

---

## ⚡ Performance Tips

| Optimization | Impact | How |
|---|---|---|
| **float16** | 2x faster, 2x less VRAM | `torch_dtype=torch.float16` |
| **max_new_tokens=96** | 2x faster than 192 | Shorter outputs |
| **device_map="auto"** | Works on 6GB GPU | Offload to CPU if needed |
| **Batch generation** | 3x faster | Generate 3 examples in parallel |
| **Cache model locally** | No re-download | First run downloads, rest cached |

---

## 🐛 Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| **CUDA out of memory** | Model too large | Use `device_map="auto"` or float16 |
| **Model not found** | Bad model name | Check: `typhoon-ai/llama3.2-typhoon2-3b` |
| **Output not Thai** | Poor prompt | Improve prompt with Thai examples |
| **JSON parse error** | Invalid JSON | Use `_extract_json()` to find start |
| **Very slow generation** | CPU inference | Check: are you using CUDA? |

---

## 📝 Example Output

When you call the model for a QA task with seed "กองทุนรวมคืออะไร":

```
INPUT PROMPT:
────────────────────────────────────────
You are generating Thai-language synthetic Q&A data.
Rules:
- Use Thai language only.
- task_type must be: qa
- user: the Thai question
- assistant: a DIRECT ANSWER in Thai
- DO NOT respond with questions!

Seed input: กองทุนรวมคืออะไร

JSON:

MODEL OUTPUT:
────────────────────────────────────────
{
  "task_type": "qa",
  "user": "กองทุนรวมคืออะไร",
  "assistant": "กองทุนรวม คือ เครื่องมือการลงทุนทางอ้อมของผู้ลงทุน 
               ที่รวมเงินจากนักลงทุนหลายคนแล้วนำไปลงทุนในหลักทรัพย์ต่างชนิด",
  "label": "qa_mode",
  "tags": ["gen"]
}

VALIDATION:
────────────────────────────────────────
✓ Valid JSON
✓ Thai chars: 93/98 = 94.9% (> 70%)
✓ Has "user" and "assistant"
✓ Save to raw.jsonl!
```

---

**Summary:** Model calling is a 3-step process:
1. **Tokenize** prompt (text → token IDs)
2. **Generate** with `model.generate()` (smart sampling)
3. **Decode** output (token IDs → JSON text)

Then **validate** with Thai character checking to keep only quality examples!
