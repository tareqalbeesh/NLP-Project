import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import redis
import requests
from sklearn.metrics import classification_report, roc_auc_score

WEBHOOK_URL = 'http://localhost:5678/webhook/eval-verify'
VERDICT_RE = re.compile(r'VERDICT:\s*(SAME|DIFFERENT|UNKNOWN)', re.IGNORECASE)
SCORE_RE = re.compile(r'SCORE:\s*([0-9]*\.?[0-9]+)')


def parse_verdict(response_text):
    v = VERDICT_RE.search(response_text)
    s = SCORE_RE.search(response_text)
    verdict = v.group(1).upper() if v else None
    score = float(s.group(1)) if s else None
    if verdict == 'UNKNOWN' or verdict is None:
        return None, score
    return (1 if verdict == 'SAME' else 0), score


def collect_authors(r):
    authors = defaultdict(list)
    for key in r.scan_iter('dataset:reuters5050:author:*:doc:*'):
        author = key.split(':')[3]
        authors[author].append(key)
    return authors


def make_pairs(r, authors, n):
    author_list = list(authors.keys())
    eligible = [a for a in author_list if len(authors[a]) >= 2]
    pairs = []
    for _ in range(n // 2):
        a = random.choice(eligible)
        k1, k2 = random.sample(authors[a], 2)
        pairs.append((r.hget(k1, 'text'), r.hget(k2, 'text'), 1))

        a, b = random.sample(author_list, 2)
        k1 = random.choice(authors[a])
        k2 = random.choice(authors[b])
        pairs.append((r.hget(k1, 'text'), r.hget(k2, 'text'), 0))
    random.shuffle(pairs)
    return pairs


def call_agent(text_a, text_b, timeout):
    resp = requests.post(
        WEBHOOK_URL,
        json={'text_a': text_a, 'text_b': text_b},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.text


def evaluate(pairs, timeout, progress_every=5):
    results = []
    t0 = time.time()
    for i, (ta, tb, label) in enumerate(pairs, 1):
        try:
            t_call = time.time()
            response_text = call_agent(ta, tb, timeout)
            elapsed = time.time() - t_call
            pred, score = parse_verdict(response_text)
            results.append({
                'label': label,
                'pred': pred,
                'score': score,
                'elapsed_sec': round(elapsed, 2),
                'response_preview': response_text[-200:],
            })
        except Exception as e:
            results.append({
                'label': label,
                'pred': None,
                'score': None,
                'elapsed_sec': None,
                'error': str(e),
            })
        if i % progress_every == 0 or i == len(pairs):
            total = time.time() - t0
            avg = total / i
            remaining = avg * (len(pairs) - i)
            print(f'  [{i}/{len(pairs)}] avg {avg:.1f}s/pair, ~{remaining/60:.1f} min remaining')
    return results


def metrics(results):
    scored = [r for r in results if r.get('pred') is not None]
    skipped = len(results) - len(scored)
    tp = sum(1 for r in scored if r['label'] == 1 and r['pred'] == 1)
    fp = sum(1 for r in scored if r['label'] == 0 and r['pred'] == 1)
    tn = sum(1 for r in scored if r['label'] == 0 and r['pred'] == 0)
    fn = sum(1 for r in scored if r['label'] == 1 and r['pred'] == 0)
    accuracy = (tp + tn) / len(scored) if scored else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        'n_total': len(results),
        'n_scored': len(scored),
        'n_skipped_or_failed': skipped,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def health_check(r):
    try:
        n_keys = sum(1 for _ in r.scan_iter('dataset:reuters5050:author:*:doc:*'))
    except Exception as e:
        print(f'!! redis unreachable: {e}', file=sys.stderr)
        return False
    if n_keys == 0:
        print('!! redis has no reuters keys. Run data-loading/reuters-dataset-loading.py first.', file=sys.stderr)
        return False
    print(f'redis: {n_keys} reuters docs indexed')

    try:
        ping = requests.post(WEBHOOK_URL, json={
            'text_a': 'This is a short ping to verify the eval webhook is active.',
            'text_b': 'This is another short ping for the same purpose.',
        }, timeout=120)
    except Exception as e:
        print(f'!! webhook unreachable at {WEBHOOK_URL}: {e}', file=sys.stderr)
        print('   Did you import + ACTIVATE the eval_agent_verify workflow?', file=sys.stderr)
        return False
    if ping.status_code != 200:
        print(f'!! webhook returned status {ping.status_code}', file=sys.stderr)
        return False
    print('webhook: ping returned 200 OK')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=20, help='number of pairs (default 20 for smoke test)')
    ap.add_argument('--timeout', type=float, default=180, help='per-call timeout in seconds')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default='results/agent_eval.json')
    ap.add_argument('--skip-health-check', action='store_true')
    args = ap.parse_args()

    random.seed(args.seed)
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    if not args.skip_health_check and not health_check(r):
        sys.exit(1)

    authors = collect_authors(r)
    pairs = make_pairs(r, authors, args.n)
    print(f'evaluating {len(pairs)} pairs against the agent...')

    results = evaluate(pairs, timeout=args.timeout)
    m = metrics(results)
    print(json.dumps(m, indent=2))

    scored = [r for r in results if r.get('pred') is not None and r.get('score') is not None]
    if len(scored) >= 2 and len(set(r['label'] for r in scored)) == 2:
        y_true = [r['label'] for r in scored]
        y_pred = [r['pred'] for r in scored]
        y_score = [r['score'] for r in scored]
        try:
            print('AUC-ROC:', roc_auc_score(y_true, y_score))
        except ValueError as e:
            print(f'AUC-ROC unavailable: {e}')
        print(classification_report(y_true, y_pred, digits=4))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'metrics': m, 'results': results}, f, indent=2)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
