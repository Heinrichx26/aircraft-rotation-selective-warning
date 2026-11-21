import argparse
import json
from pathlib import Path

from scipy.stats import beta


def one_sided_upper(false_alerts, issued_alerts, family_size, alpha=0.05):
    if false_alerts < 0 or issued_alerts < false_alerts or family_size < 1:
        raise ValueError('Supply valid false-alert counts and a positive family size.')
    if issued_alerts == 0:
        return None
    if false_alerts == issued_alerts:
        return 100.0
    value = beta.ppf(1 - alpha / family_size, false_alerts + 1, issued_alerts - false_alerts)
    return round(float(value) * 100, 2)


def evaluate(source, alpha=0.05):
    rows = source.get('budgets', [])
    methods = sorted({row['policy'] for row in rows})
    if not rows or not methods:
        raise ValueError('The input must contain budget outcomes for each fixed policy.')

    method_families = {}
    for method in methods:
        family = sorted((row for row in rows if row['policy'] == method), key=lambda row: row['budget'])
        budgets = [row['budget'] for row in family]
        if len(budgets) != len(set(budgets)):
            raise ValueError(f'Duplicate budgets for {method}.')
        checked = []
        for row in family:
            pointwise = one_sided_upper(row['screen_false'], row['screen_issued'], 1, alpha)
            simultaneous = one_sided_upper(row['screen_false'], row['screen_issued'], len(family), alpha)
            checked.append({
                'budget': row['budget'],
                'screen_false': row['screen_false'],
                'screen_issued': row['screen_issued'],
                'pointwise_upper_percent': pointwise,
                'familywise_upper_percent': simultaneous,
                'within_budget': simultaneous is not None and simultaneous <= row['budget'],
            })
        method_families[method] = {
            'policy_count': len(family),
            'policies': checked,
        }

    all_policies = sorted(rows, key=lambda row: (row['policy'], row['budget']))
    combined = []
    for row in all_policies:
        bound = one_sided_upper(row['screen_false'], row['screen_issued'], len(all_policies), alpha)
        combined.append({
            'policy': row['policy'],
            'budget': row['budget'],
            'screen_false': row['screen_false'],
            'screen_issued': row['screen_issued'],
            'familywise_upper_percent': bound,
            'within_budget': bound is not None and bound <= row['budget'],
        })

    return {
        'method': 'One-sided Clopper-Pearson upper limits with Bonferroni allocation.',
        'familywise_error_rate': alpha,
        'per_method_families': method_families,
        'all_policy_families': {
            'policy_count': len(all_policies),
            'policies': combined,
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate pointwise and simultaneous binomial limits for screened policies.')
    parser.add_argument('input', type=Path)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error('--alpha must be between 0 and 1.')
    result = evaluate(json.loads(args.input.read_text(encoding='utf-8')), args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Checked {sum(len(group['policies']) for group in result['per_method_families'].values())} policies in {len(result['per_method_families'])} method-specific frontiers and their combined family.")


if __name__ == '__main__':
    main()
