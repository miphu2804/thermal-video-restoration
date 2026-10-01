"""Registry and factory for restoration methods.

Every method has the signature ``fn(frames: np.ndarray, **params) -> np.ndarray``
with ``(T, H, W)`` float32 frames in and out. Experiments build methods by name
through :func:`get_method`, so new methods only need to be registered.
"""

import functools
import inspect
from collections.abc import Callable

import numpy as np

Method = Callable[..., np.ndarray]

_REGISTRY: dict[str, Method] = {}


def register(name: str) -> Callable[[Method], Method]:
    """Decorator that registers a restoration method under ``name``.

    Args:
        name: Unique method name used in configs.

    Returns:
        The decorator, which returns the function unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(fn: Method) -> Method:
        if name in _REGISTRY:
            raise ValueError(f"Method {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def available_methods() -> list[str]:
    """Return the sorted names of all registered methods."""
    return sorted(_REGISTRY)


def get_method(name: str, **params) -> Callable[[np.ndarray], np.ndarray]:
    """Build a method by name with its parameters bound.

    Args:
        name: Registered method name.
        **params: Keyword parameters passed to the method.

    Returns:
        A callable ``frames -> restored_frames``.

    Raises:
        ValueError: If ``name`` is not registered, if ``params`` are invalid
            for the registered method, or if the returned callable receives
            frames with a shape other than ``(T, H, W)``.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown method {name!r}. Available: {available_methods()}")
    fn = _REGISTRY[name]
    try:
        inspect.signature(fn).bind(None, **params)
    except TypeError as err:
        raise ValueError(f"Invalid parameters for method {name!r}: {err}") from err

    @functools.wraps(fn)
    def method(frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames)
        if frames.ndim != 3:
            raise ValueError(
                f"Method {name!r} expects frames of shape (T, H, W), got {frames.shape}"
            )
        return fn(frames, **params)

    return method
