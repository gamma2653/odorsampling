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
import re
from enum import Enum

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

cell_counter = Counter()
def inc_and_get(key):
    cell_counter[type(key)] += 1
    return cell_counter[type(key)]

@dataclass
class Cell:
    """
    Baseclass for cells.
    """
    id: int = field(default_factory=lambda: inc_and_get(Cell))
    loc: tuple[float, float] = (0.0, 0.0)
    conn: int = 0
    activation: float = 0.0

    def __str__(self):
        """Returns a Glomural object description with activation energy"""
        return f"Id: {self.id} activ: {self.activation}"

class Glom(Cell):
    id: int = field(default_factory=lambda: inc_and_get(Glom))
    dim: tuple[int, int] = (0, 0)
    recConn: MutableMapping[Receptor, float] = field(default_factory=dict)


class Mitral(Cell):
    id: int = field(default_factory=lambda: inc_and_get(Mitral))
    glomConn: MutableMapping[Glom, float] = field(default_factory=dict)

    def __str__(self):
        """Returns a Mitral object description with activation energy and ID's 
        of connected glomeruli."""
        # *********************** From Python 3.6 onwards, the standard dict type maintains insertion order by default *****************************
        gstring = self.glomConn.keys()
        return f"Mitral ID: {self.id} Mitral Activ: {self.activation} Glom: {gstring}"

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
        