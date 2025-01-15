"""Prototype of a multidisciplinary predictive engineering model (PEM) for a Hall thruster."""

import numpy as _np
import numpy.typing as _npt

__version__ = "0.2.0"

# Custom types that are used frequently
type ExpData = dict[str, _npt.NDArray[_np.float64]]
