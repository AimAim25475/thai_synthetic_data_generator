# Thai Synthetic Data (Workflow)

This folder contains a minimal, repo-aligned workflow to satisfy Topic 3 requirements:

- Create Thai synthetic data (<=7B generation model)
- Fine-tune / adapt components (routing classifier + chitchat + QA prompts)
- Verify it works in Thai (automatic gates + evaluation harness)

## What this repo currently does

- `libs/Classification.py`: routes `chat_mode` vs `qa_mode` (CountVectorizer + LogisticRegression)
- `libs/Chitchat.py`: generative model via Hugging Face (`4s4ki/doodownnakumkuing`)
- `libs/QA.py`: Haystack extractive QA (Elasticsearch + `FARMReader`)

## What we add

- A schema for synthetic examples
- Seed sets for Thai
- Scripts (next steps) to generate, filter, and evaluate

See `../eval/` for evaluation scripts.
