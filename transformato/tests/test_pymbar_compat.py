"""Compatibility tests for the pymbar 3/4 shims in analysis.py.

pymbar 4 renamed ``getFreeEnergyDifferences`` -> ``compute_free_energy_differences``
and ``computeOverlap`` -> ``compute_overlap`` (and dropped the ``return_dict``
argument; v4 always returns a dict). The returned dict keys ("Delta_f",
"dDelta_f", "matrix") are identical across versions. transformato routes these
calls through ``_mbar_free_energy_differences`` / ``_mbar_overlap`` so it runs on
both major versions.

These tests build a small two-state MBAR from synthetic data (no trajectories
needed) and check both the shims and the FreeEnergyCalculator accessors that use
them. They pass on whichever pymbar (3 or 4) is installed.
"""

import numpy as np
import pytest
from pymbar import MBAR

from transformato.analysis import (
    FreeEnergyCalculator,
    _mbar_free_energy_differences,
    _mbar_overlap,
)


def _two_state_mbar():
    """A trivial MBAR over two unit-variance Gaussians offset by 1 kT."""
    rng = np.random.default_rng(0)
    n = 400
    samples = np.concatenate([rng.normal(0.0, 1.0, n), rng.normal(1.0, 1.0, n)])
    # reduced potential of each state evaluated at every sample
    u_kn = np.array([0.5 * (samples - 0.0) ** 2, 0.5 * (samples - 1.0) ** 2])
    N_k = np.array([n, n])
    return MBAR(u_kn, N_k)


@pytest.mark.postprocessing
def test_mbar_shims_return_expected_keys():
    m = _two_state_mbar()

    fed = _mbar_free_energy_differences(m)
    assert {"Delta_f", "dDelta_f"}.issubset(fed)
    assert np.asarray(fed["Delta_f"]).shape == (2, 2)
    assert np.asarray(fed["dDelta_f"]).shape == (2, 2)

    overlap = _mbar_overlap(m)
    assert "matrix" in overlap
    assert np.asarray(overlap["matrix"]).shape == (2, 2)


@pytest.mark.postprocessing
def test_free_energy_calculator_accessors_use_shims():
    # Bypass __init__ (it needs a full configuration); inject a ready MBAR object
    # so we exercise only the accessor methods that go through the shims.
    fec = object.__new__(FreeEnergyCalculator)
    fec.mbar_results = {"vacuum": _two_state_mbar()}

    dG = np.asarray(fec.free_energy_differences(env="vacuum"))
    ddG = np.asarray(fec.free_energy_difference_uncertainties(env="vacuum"))
    overlap = np.asarray(fec.free_energy_overlap(env="vacuum"))

    assert dG.shape == (2, 2)
    assert ddG.shape == (2, 2)
    assert overlap.shape == (2, 2)
    # the two states are ~1 kT apart by construction; just a sanity range
    assert abs(dG[0, 1]) < 5.0
    assert ddG[0, 1] >= 0.0
