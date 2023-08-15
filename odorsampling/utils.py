from __future__ import annotations

import threading
from contextlib import contextmanager

import numpy as np

from odorsampling import config

from typing import TYPE_CHECKING, Protocol
if TYPE_CHECKING:
    from typing import Optional, Any, Generator


RNG = np.random.default_rng(config.RANDOM_SEED)
"""
Instance of `np.random.default_rng`
"""

# Want selections to fail fast
class DistributionFunc(Protocol):
    """Protocol for distribution types."""
    def __call__(self, a: Optional[float] = None, b: Optional[float] = None,
                 mu: Optional[float] = None, sigma: Optional[float] = None,
                 lambd: Optional[float] = None) -> float: ...

def init_dist_func_kwargs(kwargs: dict[str, Any], **defaults):
    """
    Initializes a dictionary of args to be used with a DistributionFunc.
    """
    # defaults
    kwargs.setdefault('a', 0.), kwargs.setdefault('b', 1.)
    if 'mean' in kwargs:
        assert (kwargs['mean']+kwargs['sd']) <= 1 and kwargs['mean']-kwargs['sd'] >= 0, \
            "Mean and SD are too high or low"
        kwargs.setdefault('mu', kwargs['mean'])
        try:
            kwargs.setdefault('lambd', 1/kwargs['mean'])
        except ZeroDivisionError:
            # FIXME: Shouldn't silently fail
            pass
    if 'sd' in kwargs:
        kwargs.setdefault('sigma', kwargs['sd'])

# Not the cleanest, but not bad considering what we needed to do in layers.py
def uniform_activation(a, b, **_):
    return RNG.uniform(a, b)
def gaussian_activation(mu, sigma, **_):
    return RNG.normal(mu, sigma)
def choice_gauss_activation(mu, sigma, **_):
    return RNG.choice([1,-1])*gaussian_activation(mu, sigma)
def expovar_activation(lambd, **_):
    return RNG.exponential(1/lambd)



class ReaderWriterSuite:
    """
    Reader Writer lock suite. This implementation prefers writers.
    """
    # Based on wikipedia:
    #   https://en.wikipedia.org/wiki/Readers%E2%80%93writer_lock

    g = threading.Lock()
    writer_active_con = threading.Condition(g)
    writer_active = False
    num_writers_waiting = 0
    num_readers_active = 0

    @contextmanager
    def reader(self) -> Generator[None, Any, None]:
        self.acquire_reader()
        yield
        self.release_reader()

    @contextmanager
    def writer(self) -> Generator[None, Any, None]:
        self.acquire_writer()
        yield
        self.release_writer()

    def acquire_reader(self) -> None:
        """
        Called to acquire a reader slot.
        """
        with self.g:
            while self.num_writers_waiting > 0 or self.writer_active:
                self.writer_active_con.wait()
            self.num_readers_active += 1

    def release_reader(self) -> None:
        """
        Called to release a reader slot.
        """
        with self.g:
            self.num_readers_active -= 1
            if self.num_readers_active == 0:
                self.writer_active_con.notify_all()

    def acquire_writer(self) -> None:
        with self.g:
            self.num_writers_waiting += 1
            while self.num_readers_active > 0 or self.writer_active:
                self.writer_active_con.wait()
            self.num_writers_waiting -= 1
            self.writer_active = True

    def release_writer(self) -> None:
        with self.g:
            self.writer_active = False
            self.writer_active_con.notify_all()
