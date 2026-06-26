import os
import random
import json
import requests
from collections import defaultdict
import redis
from sklearn.metrics import roc_auc_score, classification_report

random.seed(42)
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

authors = defaultdict(list)
for key in r.scan_iter("dataset:reuters5050:author:*:doc:*"):
    parts = key.split(':')
    author = parts[3]
    authors[author].append(key)

author_list = list(authors.keys())


def make_pairs(n=500):
    pairs = []
    eligible_authors = [x for x in author_list if len(authors[x]) >= 2]
    for _ in range(n // 2):
        a = random.choice(eligible_authors)
        k1, k2 = random.sample(authors[a], 2)
        pairs.append((r.hget(k1, 'text'), r.hget(k2, 'text'), 1))

        a, b = random.sample(author_list, 2)
        k1 = random.choice(authors[a])
        k2 = random.choice(authors[b])
        pairs.append((r.hget(k1, 'text'), r.hget(k2, 'text'), 0))
    random.shuffle(pairs)
    return pairs


def evaluate_baseline(pairs):
    results = []
    for ta, tb, label in pairs:
        resp = requests.post(
            'http://localhost:8000/compare_two_texts',
            json={'text_a': ta, 'text_b': tb},
        ).json()
        score = resp['overall_similarity']
        predicted = 1 if score >= 0.5 else 0
        results.append({'label': label, 'score': score, 'pred': predicted})
    return results


def metrics(results):
    tp = sum(1 for res in results if res['label'] == 1 and res['pred'] == 1)
    fp = sum(1 for res in results if res['label'] == 0 and res['pred'] == 1)
    tn = sum(1 for res in results if res['label'] == 0 and res['pred'] == 0)
    fn = sum(1 for res in results if res['label'] == 1 and res['pred'] == 0)
    accuracy = (tp + tn) / len(results) if results else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


if __name__ == '__main__':
    pairs = make_pairs(n=500)
    print(f'evaluating {len(pairs)} pairs...')
    results = evaluate_baseline(pairs)
    print(metrics(results))

    y_true = [res['label'] for res in results]
    y_score = [res['score'] for res in results]
    y_pred = [res['pred'] for res in results]
    print('AUC-ROC:', roc_auc_score(y_true, y_score))
    print(classification_report(y_true, y_pred, digits=4))

    os.makedirs('results', exist_ok=True)
    with open('results/baseline_eval.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote results/baseline_eval.json')
