from __future__ import annotations

import contextlib
from typing import Iterable, Iterator, TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.theme import Theme

from .screen import _ScreenHandle

T = TypeVar('T')

class _EquivProtocol:
    def __call__(self, item: Iterable[T], /, **kwargs) -> Iterable[T]:
        return item

class ConsoleProxy:

    def __init__(self) -> None:
        self._console = Console(
            theme=Theme({
                'progress.download': 'red',
                'progress.percentage': 'magenta',
                'progress.elapsed': 'cyan',
                'progress.remaining': 'bright_yellow'
            }),
        )
        self._time = None
        self._live = False

    def _set_live(self) -> None:
        assert not self._live, "Duplicated console context."
        self._live = True


PROXY = ConsoleProxy()


@contextlib.contextmanager
def status(desc: str, *, screen: bool = False) -> Iterator[None]:
    if not screen:
        live = PROXY._console.status('[bold][bright_yellow]' + desc, spinner='bouncingBall')
        PROXY._set_live()
        live.start()
    else:
        PROXY._set_live()
        live = _ScreenHandle(
            _desc=desc,
            _console=PROXY._console,
            _update_interval=0.5,
        )
        live.start()
    try:
        yield None
    finally:
        PROXY._live = False
        live.stop()


@contextlib.contextmanager
def progress(
    desc: str,
    *,
    transient: bool = False,
    wrap_file: bool = False,
    enabled: bool = True,
) -> Iterator[_EquivProtocol]:
    if enabled:
        progress = Progress(
            TextColumn('[bold][bright_yellow]' + desc),
            BarColumn(None),
            *([TransferSpeedColumn(), DownloadColumn()] if wrap_file else [MofNCompleteColumn()]),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=PROXY._console,
            transient=transient
        )
        PROXY._set_live()
        progress.start()
    try:
        if not enabled:
            yield _EquivProtocol()
        elif wrap_file:
            yield progress.wrap_file
        else:
            yield progress.track
    finally:
        if enabled:
            progress.stop()
            PROXY._live = False


@contextlib.contextmanager
def screen(desc: str = 'Waiting...', *, refresh_interval: float = 2,) -> Iterator[_ScreenHandle]:
    PROXY._set_live()
    handle = _ScreenHandle(
        _desc=desc,
        _console=PROXY._console,
        _update_interval=refresh_interval,
    )
    handle._start()
    try:
        yield handle
    finally:
        handle._stop()
        PROXY._live = False

def print(info: str, /) -> None:
    PROXY._console.print(info)
