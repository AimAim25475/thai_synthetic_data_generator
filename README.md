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
- ~500MB free disk space

### Optional
- Docker & Docker Compose (for Elasticsearch-based QA evaluation)
- CUDA GPU (optional; automatically detected and used if available for faster inference)

## Installation

### 1. Download/Extract the project
Ensure you have the project files in a directory.

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r chitchat_api/requirements.txt
pip install jupyter ipykernel
```

### 4. (Optional) Start Elasticsearch for QA evaluation
```bash
docker-compose -f chitchat_api/docker-compose.yml up -d
```
*Note: Elasticsearch is optional. The pipeline runs without it.*

## Quick Start

### Run the complete pipeline
1. Navigate to project directory
2. Activate virtual environment: `.venv\Scripts\activate`
3. Launch Jupyter: `jupyter notebook`
4. Open `Thai_synthetic_data_generated.ipynb`
5. Run all cells sequentially (Cell 1-8)
6. View outputs and `pipeline_report.json` for metrics

### Generated Outputs
After successful execution:
- **filtered.jsonl** - Thai-language filtered data (≥70% Thai characters)
- **route_train.csv** - Routing dataset for classification training
- **pipeline_report.json** - Complete execution metrics and statistics
- **qa_responses.json** - QA pairs extracted from filtered data

### Sample Results
Typical run metrics:
- Raw examples processed: 5+
- Thai answer rate: 100%
- Average quality score: 12.8/100
- Processing time: <1 minute (CPU); faster with GPU
- **GPU Auto-Detection:** If a CUDA-capable GPU is available, it will be automatically used for faster inference. No configuration needed.

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

Default configuration in Cell 1 of `Thai_synthetic_data_generated.ipynb`:
```python
CONFIG = {
    'model_name': 'Typhoon-1.0-3b',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Auto-detects GPU if available
    'api_url': 'http://127.0.0.1:3001',
    'raw_data_path': 'raw.jsonl',                             # Provided in repo
    'thai_char_threshold': 70.0,                              # Minimum Thai %
    'min_text_length': 10,
    'max_text_length': 100
}
```

**Note:** `raw.jsonl` is included in the project; no additional configuration needed. GPU support is automatic—if you have a CUDA-capable GPU (NVIDIA), it will be detected and used automatically.

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

- **ModuleNotFoundError**: Install requirements: `pip install -r chitchat_api/requirements.txt`
- **Jupyter not found**: Install with `pip install jupyter ipykernel`
- **Notebook kernel issues**: Restart VS Code Jupyter server or use command palette: "Jupyter: Select Kernel"
- **GPU not detected**: Pipeline runs on CPU; GPU is optional for faster inference
- **Elasticsearch connection error**: Elasticsearch is optional; pipeline works without it

## Verification Checklist

After running the notebook:
- [ ] All 8 cells executed without errors
- [ ] `filtered.jsonl` exists (1+ KB)
- [ ] `route_train.csv` exists with headers + data
- [ ] `pipeline_report.json` shows "COMPLETED" status
- [ ] Quality metrics show >90% Thai answer rate

## Next Steps (Optional)

1. Start the API server: `python chitchat_api/bot_api.py`
2. Test endpoints: `curl http://127.0.0.1:3001/health`
3. Fine-tune classification model using `route_train.csv`
4. Deploy filtered data to production

## Project Details

- **Language**: Python 3.8+
- **Framework**: PyTorch, FastAPI, Jupyter
- **Data Format**: JSONL input, CSV/JSON outputs
- **Processing**: Single-threaded, ~seconds per 100 examples

## License

MIT License - Feel free to use and modify

## Author Contact

For questions or contributions, please open an issue in the repository.
