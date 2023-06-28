# Glomeruli and Mitral Cell Objects
# Mitchell Gronowitz
# Spring 2015

# Reimplementation by Christopher De Jesus
# Summer 2023

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections import Counter
import random
import copy
import re
from enum import Enum

from scipy.stats import multivariate_normal as mvn

import config

# Type checking
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from typing import MutableMapping, Optional
    from RnO import Receptor

logger = logging.getLogger(__name__)
config.default_log_setup(logger)


class DistributionType(Enum):
    """
    Enum for distribution types.
    """
    UNIFORM = random.random
    """
    Activation levels are drawn from a random distribution.
    """
    GAUSSIAN = random.gauss
    """
    Activation levels are drawn from a Gaussian distribution with mean and sd.
    """
    EXPONENTIAL = random.expovariate
    """
    Activation levels are drawn from exponenial distribution with mean.
    """

id_tracker = Counter()
def get_id(key, type=True):
    """
    Parameters
    ----------
    key - 
        key to get id for
    type -
        if True, key is a type, else key is an instance
    """
    id_tracker[key if type else type(key)] += 1
    return id_tracker[key if type else type(key)]

@dataclass
class Cell:
    """
    Base class for cells.
    """
    id: int = field(default_factory=lambda: get_id(Cell))
    loc: tuple[float, float] = (0.0, 0.0)
    conn: int = 0
    activation: float = 0.0

    def __str__(self):
        """Returns a Glomural object description with activation energy"""
        return f"Id: {self.id} activ: {self.activation}"

class Glom(Cell):
    id: int = field(default_factory=lambda: get_id(Glom))
    dim: tuple[int, int] = (0, 0)
    recConn: MutableMapping[Receptor, float] = field(default_factory=dict)


class Mitral(Cell):
    id: int = field(default_factory=lambda: get_id(Mitral))
    glomConn: MutableMapping[Glom, float] = field(default_factory=dict)

    def __str__(self):
        """Returns a Mitral object description with activation energy and ID's 
        of connected glomeruli."""
        # *********************** From Python 3.6 onwards, the standard dict type maintains insertion order by default *****************************
        gstring = self.glomConn.keys()
        return f"Mitral ID: {self.id} Mitral Activ: {self.activation} Glom: {gstring}"

@dataclass
class QSpace:
    dims: list[tuple[int, int]] = field(default_factory=list)
    
    @property
    def q(self) -> int:
        return len(self.dims)


class Receptor:
    
    def __init__(self, qspace, mean, sda, sde, id_=None):
        self._id = id_ if id_ is not None else get_id(Receptor)
        self._qspace = qspace
        self._mean = mean
        self._sdA = sda
        self._sdE = sde
    
    def _distribute(self, qspace: QSpace, constMean, scaleAff, scaleEff):
        """Returns a list of means randomly distributed within the qspace based on the Type"""
        if constMean:
            self._mean = tuple(qspace.q[i][1]/2.0 for i in range(qspace.q))
        else:
            if config.DIST_TYPE_UNIF:
                self._mean = tuple(random.uniform(*qspace.q[i]) for i in range(qspace.q))
            elif config.DIST_TYPE_GAUSS:
                mean = []
                for i in range(qspace.q):
                    while True:
                        g = random.gauss(config.MU, config.SIG)
                        if g <= qspace.q[i][1] and g >= 0:
                            break
                    mean.append(g)
                self._mean = tuple(mean)
            else:
                raise Exception("No distribution type selected in config.")
        self._sdA = tuple(random.uniform(*scaleAff) for _ in range(qspace.q))
        self._sdE = tuple(random.uniform(*scaleEff) for _ in range(qspace.q))


    @property
    def id(self) -> int:
        return self._id
    
    @property
    def qspace(self) -> QSpace:
        """
        Returns a deep copy of the QSpace object.
        """
        return copy.deepcopy(self._qspace)

    # @classmethod
    # def create(cls, qspace: QSpace, scaleAff, scaleEff):
        

    @property
    def mean(self):
        """Returns mean of receptor."""
        return self._mean

    @mean.setter
    def mean(self, value: tuple[float, float]):
        """
        Sets mean equal to value.
        """
        logger.debug("Receptor mean changed: [%s->%s]", self._mean, value)
        self._mean = value
        self._mean_sd_change = True
    
    @property
    def sdA(self) -> tuple[float, float]:
        """Returns the standard deviations for Affinity."""
        return self._sdA

    @sdA.setter
    def sdA(self, value: tuple[float, float]):
        """
        Sets sdA equal to value.
        """
        logger.debug("Receptor sdA changed: [%s->%s]", self._sdA, value)
        self._sdA = value
        self._mean_sd_change = True

    # TODO: Make thread safe
    def _update_cov_scale(self) -> None:
        self._covA = [_sdA**2 for _sdA in self.sdA]
        self._covE = [_sdE**2 for _sdE in self.sdE]
        self._affScale = float(mvn.pdf(self.mean, self.mean, self._covA))
        self._effScale = float(mvn.pdf(self.mean, self.mean, self._covE))
        self._mean_sd_change = False

    @property
    def sdE(self) -> tuple:
        """Returns the standard deviations for Efficacy."""
        return self._sdE

    @sdE.setter
    def sdE(self, value: tuple):
        """
        Sets sdE equal to value.
        """
        logger.debug("Receptor sdE changed: [%s->%s]", self._sdE, value)
        self._sdE = value
        self._mean_sd_change = True
    
    @property
    def covA(self):
        """Returns the covariance for affinity"""
        if self._mean_sd_change:
            self._update_cov_scale()
        return self._covA
    
    @property
    def covE(self):
        """Returns the covariance for Efficacy"""
        if self._mean_sd_change:
            self._update_cov_scale()
        return self._covE
    
    @property
    def affScale(self):
        """Returns scale of receptor."""
        if self._mean_sd_change:
            self._update_cov_scale()
        return self._affScale

    @property
    def effScale(self):
        """Returns eff scale of receptor."""
        if self._mean_sd_change:
            self._update_cov_scale()
        return self._effScale

class Ligand:
    pass

class Odorscene(list[Ligand]):
    pass

#### Not defined in paper but useful structures for modelling
class GlomLayer(list[Glom]):
    """
    """
    def __init__(self, cells: Iterable[Glom] = tuple()):
        super().__init__(cells)
        

    def clear_activations(self):
        """Clears activation levels for all glom in layer."""
        for glom in self:
            glom.activation = 0.0
            glom.recConn.clear()
        logger.debug("Glom cell layer activations cleared.")
    
    def activate(self, activate_func=random.random):
        """Activates all glom in layer."""
        for glom in self:
            glom.activation = activate_func()
        logger.debug("Glom cell layer activated.")


class GlomLayers(list[GlomLayer]):

    # Already the default behavior
    # def __init__(self, cells: Iterable[Glom] = tuple()):
    #     super().__init__(cells)
    
    def clear_activations(self):
        """Clears acitvation levels for all glom layers."""
        for layer in self:
            layer.clear_activations()
        logger.debug("Glom cell layer activations cleared.")

    def activate(self, activate_func=random.random):
        """Activates all glom cells in all layers."""
        for layer in self:
            layer.activate(activate_func)
        logger.debug("Glom cell layer activated.")

    def save(self, name: str):
        """
        Saves glom layers to file.
        """
        content = "\n".join((f"{i},{glom.id},{glom.activation},{glom.loc[0]}:{glom.loc[1]},{glom.conn};"
                             for i, layer in enumerate(self) for glom in layer))
        filename = f"{name}.{config.GL_EXT}"
        with open(filename, 'w') as f:
            logger.info("Glom layer saved to `%s`.", filename)
            f.write(content)

    @classmethod
    def load(cls, name: str) -> GlomLayer:
        """Returns GL with given name from directory
        precondition: name is a string with correct extension"""
        pattern = re.compile(r"(\d+),(\d+),(\d+\.\d*),(\d+\.\d*):(\d+\.\d*),(\d+)")
        with open(name) as f:
            data = f.read()
        glom_layer = [
            Glom(int(match.group(2)), float(match.group(3)), (float(match.group(4)), float(match.group(5))), int(match.group(6)))
            for match in pattern.finditer(data)]
        logger.info("Glom layer loaded from `%s`.", name)
        return cls(glom_layer)

    def add_noise(self, noise_func=DistributionType.UNIFORM, mean=0, sd=0):
        """Adds noise to all glom cells in all layers."""
        if noise_func is DistributionType.UNIFORM:
            inc = noise_func(0, mean)
        elif noise_func is DistributionType.GAUSSIAN:
            inc = noise_func(mean, sd)
        else:
            inc = noise_func(1/mean)

        for layer in self:
            for glom in layer:
                glom.activation += min(max(glom.activation + random.choice([1,-1])*inc, 0.0), 1.0)
        logger.debug("Glom cell layer noise added.")
        