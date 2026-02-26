from __future__ import annotations

from typing import Any, Callable, Optional


class _LazyWrapper:
    def __init__(self, materialization: Callable[[], Any], hint: Optional[str] = None) -> None:
        self._materialization = materialization
        self._instance = None
        self._hint = hint

    def __getattribute__(self, name: str) -> Any:
        if name in ['_instance', '_materialization', '_hint']:
            return super().__getattribute__(name)
        if self._instance is None:
            try:
                self._instance = self._materialization()
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    e.msg +
                    f"\nPlease install the package first:\n'{self._hint}'"
                )
        return getattr(self._instance, name)

lazy_wrapper = _LazyWrapper
