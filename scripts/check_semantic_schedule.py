import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.semantic_warmup import get_effective_lambda_semantic


def print_values(title, iterations, semantic_start_iter):
    print(title)
    for current_iter in iterations:
        value = get_effective_lambda_semantic(
            current_iter=current_iter,
            lambda_semantic=0.10,
            use_semantic_warmup=False,
            semantic_warmup_start=3000,
            semantic_warmup_end=7000,
            semantic_warmup_type='linear',
            semantic_late_start=True,
            semantic_start_iter=semantic_start_iter,
        )
        print('iter=%-6d effective_lambda_semantic=%.4f' % (current_iter, value))


def check_detach_source():
    card_path = os.path.join(PROJECT_ROOT, 'models', 'CARD.py')
    with open(card_path, 'r', encoding='utf-8') as f:
        source = f.read()
    has_semantic_detach = 'semantic_input = diff_features.detach() if self.use_semantic_detach else diff_features' in source
    has_caption_output = 'return caption_input' in source
    print('')
    print('E3 semantic detach:')
    print('use_semantic_detach=True')
    if not has_semantic_detach:
        raise RuntimeError('semantic branch input does not use detach() in CARD.forward.')
    if not has_caption_output:
        raise RuntimeError('CARD.forward no longer returns the original caption branch output.')
    print('semantic branch input uses detach()')
    print('caption branch input is not detached')


def main():
    print_values(
        'E1 latestart4000:',
        [0, 3999, 4000, 10000],
        semantic_start_iter=4000,
    )
    print('')
    print_values(
        'E2 latestart6000:',
        [0, 5999, 6000, 10000],
        semantic_start_iter=6000,
    )
    check_detach_source()


if __name__ == '__main__':
    main()
