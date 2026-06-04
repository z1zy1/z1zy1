import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.semantic_warmup import get_effective_lambda_semantic


def print_values(title, iterations, **kwargs):
    print(title)
    for current_iter in iterations:
        value = get_effective_lambda_semantic(current_iter=current_iter, **kwargs)
        print('iter=%-6d effective_lambda_semantic=%.4f' % (current_iter, value))


def main():
    lambda_semantic = 0.1
    semantic_warmup_start = 3000
    semantic_warmup_end = 7000
    semantic_start_iter = 5000

    print_values(
        'warmup mode',
        [0, 2999, 3000, 5000, 7000, 10000],
        lambda_semantic=lambda_semantic,
        use_semantic_warmup=True,
        semantic_warmup_start=semantic_warmup_start,
        semantic_warmup_end=semantic_warmup_end,
        semantic_warmup_type='linear',
        semantic_late_start=False,
        semantic_start_iter=semantic_start_iter,
    )
    print('')
    print_values(
        'late start mode',
        [0, 4999, 5000, 10000],
        lambda_semantic=lambda_semantic,
        use_semantic_warmup=False,
        semantic_warmup_start=semantic_warmup_start,
        semantic_warmup_end=semantic_warmup_end,
        semantic_warmup_type='linear',
        semantic_late_start=True,
        semantic_start_iter=semantic_start_iter,
    )


if __name__ == '__main__':
    main()
