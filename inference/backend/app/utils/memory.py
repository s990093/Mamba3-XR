from __future__ import annotations

import gc
import time

_last_flush_ts = 0.0
_FLUSH_DEBOUNCE_SECONDS = 3.0


def flush_memory() -> dict[str, object]:
    global _last_flush_ts

    now = time.monotonic()
    if now - _last_flush_ts < _FLUSH_DEBOUNCE_SECONDS:
        return {
            'ok': False,
            'message': f'Flush is debounced. Retry in {round(_FLUSH_DEBOUNCE_SECONDS - (now - _last_flush_ts), 2)}s.',
        }

    torch_cleared = False
    mlx_cleared = False

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch_cleared = True
    except Exception:
        pass

    try:
        import mlx.core as mx  # type: ignore

        mx.clear_cache()
        mlx_cleared = True
    except Exception:
        pass

    gc.collect()
    _last_flush_ts = now

    return {
        'ok': True,
        'message': 'Memory cache cleared.',
        'torch_cleared': torch_cleared,
        'mlx_cleared': mlx_cleared,
    }
