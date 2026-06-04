def get_effective_lambda_semantic(
    current_iter: int,
    lambda_semantic: float,
    use_semantic_warmup: bool = False,
    semantic_warmup_start: int = 3000,
    semantic_warmup_end: int = 7000,
    semantic_warmup_type: str = 'linear',
    semantic_late_start: bool = False,
    semantic_start_iter: int = 5000,
) -> float:
    if lambda_semantic <= 0:
        return 0.0

    if semantic_late_start:
        if current_iter < semantic_start_iter:
            return 0.0
        return lambda_semantic

    if not use_semantic_warmup:
        return lambda_semantic

    if current_iter < semantic_warmup_start:
        return 0.0

    if current_iter >= semantic_warmup_end:
        return lambda_semantic

    denom = max(1, semantic_warmup_end - semantic_warmup_start)
    progress = (current_iter - semantic_warmup_start) / float(denom)
    progress = max(0.0, min(1.0, progress))

    if semantic_warmup_type == 'linear':
        return lambda_semantic * progress
    raise ValueError('Unsupported semantic_warmup_type: %s' % semantic_warmup_type)
