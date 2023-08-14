from __future__ import annotations

import numpy as np

from odorsampling import config

from typing import TYPE_CHECKING, Protocol
if TYPE_CHECKING:
    from typing import Optional, Any


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
