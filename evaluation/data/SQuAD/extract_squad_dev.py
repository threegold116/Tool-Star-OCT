import json
import sys
from pathlib import Path

# Input and output paths
BASE = Path(__file__).parent
INPUT = BASE / 'dev-v2.0.json'
OUTPUT = BASE / 'dev_extracted.jsonl'

def extract():
    if not INPUT.exists():
        print(f'Input file not found: {INPUT}', file=sys.stderr)
        return 1
    # File is a single line giant JSON object (per user). Read whole.
    with INPUT.open('r', encoding='utf-8') as f:
        data = json.loads(f.read())
    # SQuAD v2 schema: {"data":[{"title":"...","paragraphs":[{"context":"...","qas":[{"question":"...","answers":[{"text":"...","answer_start":123}],"is_impossible":false}, ...]}]}], "version": "2.0"}
    count = 0
    with OUTPUT.open('w', encoding='utf-8') as out:
        for article in data.get('data', []):
            for para in article.get('paragraphs', []):
                for qa in para.get('qas', []):
                    q = qa.get('question', '').strip()
                    # answers list may be empty in v2.0 (unanswerable). Use empty list.
                    answers = qa.get('answers', []) or []
                    # Each answer object has 'text'; collect unique non-empty texts preserving order.
                    ans_texts = []
                    seen = set()
                    for a in answers:
                        t = (a.get('text') or '').strip()
                        if t and t not in seen:
                            seen.add(t)
                            ans_texts.append(t)
                    obj = {"question": q, "answer": ans_texts}
                    out.write(json.dumps(obj, ensure_ascii=False) + '\n')
                    count += 1
    print(f'Wrote {count} QA pairs to {OUTPUT}')
    return 0

if __name__ == '__main__':
    raise SystemExit(extract())
