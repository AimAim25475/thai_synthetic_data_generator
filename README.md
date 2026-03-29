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

### 1. Clone the repository
```bash
git clone <repository-url>
cd chitchat
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

### Run the complete pipeline
1. Open `Thai_synthetic_data_generated.ipynb` in Jupyter
2. Configure settings in Cell 2 (model name, device, API URL)
3. Ensure `raw.jsonl` exists with source examples
4. Run all cells sequentially

### Expected Output
- `filtered.jsonl` - Thai-language filtered data
- `route_train.csv` - Extracted routing examples for fine-tuning
- Quality report with metrics:
  - Thai Answer Rate: >96%
  - Average Quality Score: >85/100
  - Example filters: ≥70% Thai chars, 10-100 char length

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
    'model_name': 'Typhoon-1.0-7b',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'api_url': 'http://127.0.0.1:3001',
    'raw_data_path': 'raw.jsonl',
    ...
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

- **ModuleNotFoundError**: Ensure all requirements installed: `pip install -r chitchat_api/requirements.txt`
- **GPU not found**: Set `device='cpu'` in CONFIG or install CUDA
- **Elasticsearch connection error**: Start docker: `docker-compose -f chitchat_api/docker-compose.yml up -d`
- **Port 3001 already in use**: Change `api_url` in CONFIG to available port

## Next Steps

1. Verify Elasticsearch is running: `docker ps`
2. Start bot API: `python chitchat_api/bot_api.py`
3. Run notebook cells sequentially
4. Check logs in terminal for any errors

## License

[Add your license here]

## Contact

[Add contact information]
