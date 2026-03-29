# Thai Synthetic Data Generation Pipeline

A comprehensive pipeline for generating, filtering, evaluating, and analyzing Thai synthetic data using the Typhoon 3B language model.

## Overview

This project automates the creation and validation of high-quality Thai synthetic data through a sequential pipeline:
1. **Generate** - Create synthetic examples using Typhoon 3B LLM
2. **Filter** - Keep only examples with ≥70% Thai characters
3. **Extract** - Extract routing information for classification training
4. **Evaluate** - Run API evaluation on cleaned data
5. **Analyze** - Validate quality with 6-point scoring system
6. **Report** - Generate comprehensive summary statistics

## Requirements

- Python 3.8+
- Docker & Docker Compose (for Elasticsearch)
- CUDA-capable GPU (recommended for Typhoon 3B inference)
- ~4GB free disk space

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AimAim25475/thai_synthetic_data_generator.git
cd thai_synthetic_data_generator
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r chitchat_api/requirements.txt
```

### 4. Start Elasticsearch (required for QA evaluation)
```bash
docker-compose -f chitchat_api/docker-compose.yml up -d
```

## Quick Start

### Option 1: Interactive Mode (Recommended for Testing)

Generate Thai QA examples with an interactive terminal interface:

```bash
python chitchat_api/synthetic/generate_typhoon.py --interactive
```

Then follow the prompts:
- Model: `typhoon-ai/llama3.2-typhoon2-3b`
- Task: `4` (for QA) or `1` (for all)
- Examples: `5` (quick test) or `50` (full batch)
- Device: `1` (auto-detect)
- Start: `y`

Expected time: 2-5 minutes for 5 examples, 30+ minutes for 50 examples.

### Option 2: Run the Complete Pipeline

1. Open `Thai_synthetic_data_generated.ipynb` in Jupyter
2. Configure settings in Cell 2 (model name, device, API URL)
3. Ensure `raw.jsonl` exists with source examples
4. Run all cells sequentially

### Expected Output
- `raw.jsonl` - Generated synthetic data
- `filtered.jsonl` - Thai-language filtered data (≥70% Thai chars)
- Quality metrics:
  - Thai Answer Rate: >96%
  - Average Quality Score: >85/100

## Interactive QA Generation

Generate Thai synthetic QA examples using an interactive terminal interface:

### Run Interactive Mode

```bash
python chitchat_api/synthetic/generate_typhoon.py --interactive
```

### Step-by-Step Guide

The script will prompt you for:

1. **Model Name** (required)
   ```
   [*] Model name: typhoon-ai/llama3.2-typhoon2-3b
   ```

2. **Seeds File Path** (default provided)
   ```
   [*] Seeds file path [chitchat_api/synthetic/seeds_thai.json]: 
   (Press Enter to use default)
   ```

3. **Output JSONL Path** (where to save results)
   ```
   [*] Output JSONL file path [raw.jsonl]: qa_examples.jsonl
   ```

4. **Task Type** (what to generate)
   ```
   [Task Type]
   1. all      - Generate route, chitchat, QA
   2. route    - Generate routing examples only
   3. chitchat - Generate chitchat examples only
   4. qa       - Generate QA examples only
   
   Select [1-4] or type (all/route/chitchat/qa) [1]: 4
   ```

5. **Number of Examples** (quick test: use 5)
   ```
   [*] Number of examples per category [50]: 5
   ```

6. **Max Tokens Per Example** (default 512 is good)
   ```
   [*] Max tokens per example [512]: 
   (Press Enter for default)
   ```

7. **Temperature** (default 0.7 balances creativity and consistency)
   ```
   [*] Temperature (0.1-2.0) [0.7]: 
   (Press Enter for default)
   ```

8. **Device** (GPU or CPU)
   ```
   [Device]
   1. auto  - Auto-detect GPU/CPU
   2. cuda  - Force GPU (if available)
   3. cpu   - Force CPU only
   
   Select [1-3] or type (auto/cuda/cpu) [1]: 1
   ```

9. **Data Type** (precision level)
   ```
   [Data Type]
   1. auto     - Auto-detect (recommended)
   2. float16  - Half precision (faster, less memory)
   3. float32  - Full precision (more memory)
   
   Select [1-3] or type (auto/float16/float32) [1]: 1
   ```

10. **Min Thai Character Ratio** (quality filter)
    ```
    [*] Min Thai character ratio (0.0-1.0) [0.70]:
    (Press Enter for default - requires 70% Thai)
    ```

11. **Confirmation** (review and start)
    ```
    Configuration Summary:
    ======================================================================
    Model:          typhoon-ai/llama3.2-typhoon2-3b
    Seeds:          chitchat_api/synthetic/seeds_thai.json
    Output:         qa_examples.jsonl
    Task:           qa
    Examples:       5 per category
    Max Tokens:     512
    Temperature:    0.7
    Device:         auto
    Data Type:      auto
    Thai Ratio:     0.7
    ======================================================================
    
    [?] Start generation? (y/n) [y]: y
    ```

### Expected Output

Generated file (`qa_examples.jsonl`) with entries like:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "task_type": "qa",
  "user": "บอกความหมายของการลงทุนในหุ้น",
  "assistant": "การลงทุนในหุ้นคือการซื้อส่วนแบ่ง (share) ของบริษัท เพื่อให้คุณมีส่วนได้ส่วนเสีย ในกำไร/ขาดทุนของบริษัทนั้น...",
  "label": "qa_mode",
  "style": null,
  "tags": ["gen", "financial"],
  "source": "synthetic",
  "quality_score": 0.92
}
```

### Verify Quality

View the generated data:
```powershell
Get-Content qa_examples.jsonl -Head 3
```

Or parse as JSON:
```powershell
Get-Content qa_examples.jsonl | ConvertFrom-Json | Format-List
```

### Command-Line Mode (Alternative)

If you prefer non-interactive mode:

```bash
python chitchat_api/synthetic/generate_typhoon.py \
    --model typhoon-ai/llama3.2-typhoon2-3b \
    --task qa \
    --n 10 \
    --out qa_examples.jsonl \
    --device auto \
    --dtype float16
```

### Quick Test Settings

For a fast validation test (2-3 minutes):
```
Model: typhoon-ai/llama3.2-typhoon2-3b
Task: qa (option 4)
Examples: 5
Tokens: 512
Temperature: 0.7
Device: auto
Data Type: auto
Thai Ratio: 0.70
```

### Next: Filter & Evaluate

After generation, filter the data:
```bash
python chitchat_api/synthetic/filter_thai.py \
    --input qa_examples.jsonl \
    --output filtered_qa.jsonl \
    --min-ratio 0.70
```

Then evaluate with the API:
```bash
python chitchat_api/eval/eval_thai_api.py \
    --base-url http://127.0.0.1:3001 \
    --jsonl filtered_qa.jsonl
```

## Project Structure

```
chitchat_api/
├── bot_api.py              # FastAPI server
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Elasticsearch setup
├── synthetic/              # Data generation scripts
│   ├── generate_typhoon.py
│   ├── filter_thai.py
│   └── make_route_dataset.py
├── eval/                   # Evaluation scripts
│   ├── eval_thai_api.py
│   └── save_qa_responses.py
└── libs/                   # Core modules
    ├── Chitchat.py
    ├── Classification.py
    └── QA.py

Thai_synthetic_data_generated.ipynb  # Main pipeline notebook
raw.jsonl                             # Input data (not in repo)
```

## Configuration

Edit Cell 2 in `Thai_synthetic_data_generated.ipynb`:
```python
CONFIG = {
    "MODEL_NAME": "typhoon-ai/llama3.2-typhoon2-3b",
    "DEVICE": "cuda",  # or "cpu" if no GPU
    "DTYPE": "float16",
    "API_BASE_URL": "http://127.0.0.1:3001",
    "MAX_NEW_TOKENS": 96,
    "N_EXAMPLES": 20,
    "MIN_THAI_RATIO": 0.70,
}
```

## Dependencies

### Core
- pandas - Data manipulation
- torch - PyTorch framework
- transformers - Hugging Face models
- fastapi - API framework

### Optional
- elasticsearch - Required for QA evaluation
- docker - For containerized services

## Troubleshooting

### ModuleNotFoundError: No module named 'torch'

**Solution:** Reinstall dependencies:
```bash
pip install -r chitchat_api/requirements.txt
```

### CUDA/GPU Not Found

**Solution:** Use CPU mode in CONFIG:
```python
CONFIG["DEVICE"] = "cpu"
```

Or install CUDA and PyTorch:
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)

### Elasticsearch Connection Error

**Solution:** Start Elasticsearch with Docker:
```bash
docker-compose -f chitchat_api/docker-compose.yml up -d
docker ps  # Verify running
```

### Port 3001 Already in Use

**Solution:** Change the API port:
```python
CONFIG["API_BASE_URL"] = "http://127.0.0.1:3002"
```

### OutOfMemory Error on GPU

**Solution:** Reduce parameters or use float16:
```python
CONFIG["DTYPE"] = "float16"  # Reduces memory by 50%
CONFIG["MAX_NEW_TOKENS"] = 64  # Shorter generations
```

## Next Steps

1. **Interactive QA Generation (Recommended for Testing):**
   ```bash
   python chitchat_api/synthetic/generate_typhoon.py --interactive
   # Follow the prompts for model name, task type, examples, etc.
   ```

2. **Generate Raw Data (Command-Line Mode):**
   ```bash
   python chitchat_api/synthetic/generate_typhoon.py \
       --model typhoon-ai/llama3.2-typhoon2-3b \
       --task qa \
       --n 20 \
       --out chitchat_api/data/thai_synth/raw.jsonl
   ```

3. **Filter for Quality:**
   ```bash
   python chitchat_api/synthetic/filter_thai.py \
       --input raw.jsonl \
       --output filtered.jsonl \
       --min-ratio 0.70
   ```

4. **Start API Server:**
   ```bash
   python chitchat_api/bot_api.py
   ```

5. **Evaluate Generated Examples:**
   ```bash
   python chitchat_api/eval/eval_thai_api.py \
       --base-url http://127.0.0.1:3001 \
       --jsonl filtered.jsonl
   ```

6. **Run Complete Pipeline:**
   ```bash
   jupyter notebook Thai_synthetic_data_generated.ipynb
   ```

7. **Fine-tune Models:**
   - Use `route_train.csv` for classification fine-tuning
   - Use `filtered.jsonl` for QA model fine-tuning

## License

MIT License - See [LICENSE](LICENSE) file for details.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions.

## Contact

- **Author:** Rapeepat Ounkhom (AimAim25475)
- **GitHub:** [AimAim25475](https://github.com/AimAim25475)
- **Issues:** For bugs and feature requests, please open an [issue](https://github.com/AimAim25475/thai_synthetic_data_generator/issues)
