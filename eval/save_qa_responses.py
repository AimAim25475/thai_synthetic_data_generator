"""
Save actual API-generated Q&A responses to JSON and CSV.
This captures the real answers generated at runtime.
"""

import json
import csv
import sys
import time
import requests
from pathlib import Path
from argparse import ArgumentParser

def thai_char_ratio(text: str) -> float:
    """Calculate Thai character ratio."""
    if not text:
        return 0.0
    thai_count = sum(1 for c in text if '\u0E01' <= c <= '\u0E5B')
    return thai_count / len(text)

def main():
    ap = ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:3001")
    ap.add_argument("--jsonl", required=True, help="Input filtered.jsonl")
    ap.add_argument("--out-json", default="eval_qa_responses.json")
    ap.add_argument("--out-csv", default="eval_qa_responses.csv")
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    jsonl_path = Path(args.jsonl)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    print(f"📖 Reading: {jsonl_path}")
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"🚀 Calling API at {base_url}")
    responses = []
    errors = 0

    for i, ex in enumerate(examples):
        task_type = ex.get('task_type', '')
        user_input = ex.get('user', '')
        label = ex.get('label', 'chat_mode')
        
        try:
            # Call API with GET parameters
            # Use label to determine mode (chat_mode vs qa_mode)
            mode = label if label in {"chat_mode", "qa_mode"} else "chat_mode"
            
            response = requests.get(
                f"{base_url}/chat",
                params={
                    "line": user_input,
                    "mode": mode,
                    "ret_tk": 3,
                    "red_tk": 1
                },
                timeout=60
            )
            
            if response.status_code == 200:
                bot_reply = response.text.strip()
            else:
                bot_reply = f"ERROR {response.status_code}"
                errors += 1
            
            # Calculate Thai ratio
            thai_ratio = thai_char_ratio(bot_reply)
            is_thai = thai_ratio >= 0.70
            
            result = {
                "id": ex.get('id', ''),
                "task_type": task_type,
                "user_input": user_input,
                "bot_response": bot_reply,
                "thai_ratio": round(thai_ratio, 3),
                "is_thai": is_thai,
                "label": ex.get('label', ''),
                "source": ex.get('source', '')
            }
            responses.append(result)
            
            status = "✓" if is_thai else "✗"
            print(f"[{i+1}/{len(examples)}] {status} {task_type[:8]:8} | {user_input[:40]:40} | {thai_ratio:.1%}")
            
            time.sleep(0.1)  # Rate limit
            
        except Exception as e:
            print(f"[{i+1}/{len(examples)}] ✗ ERROR: {str(e)[:50]}")
            errors += 1
            responses.append({
                "id": ex.get('id', ''),
                "task_type": task_type,
                "user_input": user_input,
                "bot_response": f"ERROR: {str(e)}",
                "thai_ratio": 0.0,
                "is_thai": False,
                "label": ex.get('label', ''),
                "source": ex.get('source', '')
            })

    # Save JSON
    print(f"\n💾 Saving to {out_json}")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    # Save CSV
    print(f"💾 Saving to {out_csv}")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "user_input", "bot_response", "thai_ratio", "is_thai", "label", "source"])
        writer.writeheader()
        for r in responses:
            writer.writerow({
                "task_type": r["task_type"],
                "user_input": r["user_input"][:100],
                "bot_response": r["bot_response"][:200],
                "thai_ratio": r["thai_ratio"],
                "is_thai": r["is_thai"],
                "label": r["label"],
                "source": r["source"]
            })

    # Summary
    thai_count = sum(1 for r in responses if r["is_thai"])
    print(f"\n✅ Results:")
    print(f"  Total: {len(responses)}")
    print(f"  Thai: {thai_count} ({100*thai_count/len(responses):.1f}%)")
    print(f"  Errors: {errors}")

if __name__ == "__main__":
    main()
