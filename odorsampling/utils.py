from __future__ import annotations

from random import Random
import math

from typing import TYPE_CHECKING, Protocol
if TYPE_CHECKING:
    from typing import Sequence, Optional, Any


RANDOM_SEED = None
DEFAULT_RANDOM_GEN = Random(RANDOM_SEED) if RANDOM_SEED else Random()

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
    if 'sd' in kwargs:
        kwargs.setdefault('sigma', kwargs['sd'])
    kwargs.setdefault('lambd', 1/kwargs['mean'])

# Not the cleanest, but not bad considering what we needed to do in layers.py
uniform_activation: DistributionFunc = lambda a, b, **_: DEFAULT_RANDOM_GEN.uniform(a, b) if (a,b) != (0,1) else DEFAULT_RANDOM_GEN.random()
"""
If a,b == (0,1), will resort to Random.random to not include b.
"""
gaussian_activation: DistributionFunc = lambda mu, sigma, **_: DEFAULT_RANDOM_GEN.gauss(mu, sigma)
choice_gauss_activation: DistributionFunc = lambda mu, sigma, **_: \
    DEFAULT_RANDOM_GEN.choice([1,-1])*gaussian_activation(mu, sigma)
expovar_activation: DistributionFunc = lambda lambd, **_: DEFAULT_RANDOM_GEN.expovariate(lambd)
