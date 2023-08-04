from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING, Protocol
if TYPE_CHECKING:
    from typing import Optional, Any


RANDOM_SEED = None
RNG = np.random.default_rng(RANDOM_SEED)

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
# TODO: Check if ternary is necessary for excluding endpoint (`b`).
uniform_activation: DistributionFunc = lambda a, b, **_: RNG.uniform(a, b) if (a,b) != (0,1) else RNG.random()
gaussian_activation: DistributionFunc = lambda mu, sigma, **_: RNG.normal(mu, sigma)
"""
If a,b == (0,1), will resort to Random.random to not include b.
"""
choice_gauss_activation: DistributionFunc = lambda mu, sigma, **_: \
    RNG.choice([1,-1])*gaussian_activation(mu, sigma)
expovar_activation: DistributionFunc = lambda lambd, **_: RNG.exponential(1/lambd)
