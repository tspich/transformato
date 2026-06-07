"""Unit tests for graceful common-core bond/angle/torsion *breaking* (Phase 1a
of the pseudouridine / bond-migration work -- see DESIGN_pseudouridine.md).

A common-core bonded term that exists in ligand 1 but has no counterpart in
ligand 2 (a migrating/breaking bond) must scale its force constant to 0 over
lambda when ``allow_bonded_breaking`` is set, and otherwise raise (preserving
the historical strict check). These tests drive CommonCoreTransformation in
isolation with lightweight fakes -- no parmed/psf objects required.
"""

from collections import namedtuple

import pytest

from transformato.mutate import CommonCoreTransformation


# --- lightweight stand-ins for the parmed objects the methods touch ----------
class _Atom:
    def __init__(self, name, type_, idx=0):
        self.name = name
        self.type = type_
        self.idx = idx


_BT = namedtuple("BT", "k req")
_AT = namedtuple("AT", "k theteq")


class _Bond:
    def __init__(self, a1, a2, k, req):
        self.atom1, self.atom2 = a1, a2
        self.type = _BT(k, req)


class _Angle:
    def __init__(self, a1, a2, a3, k, theteq):
        self.atom1, self.atom2, self.atom3 = a1, a2, a3
        self.type = _AT(k, theteq)


class _View:
    def __init__(self, bonds=(), angles=(), dihedrals=()):
        self.bonds = list(bonds)
        self.angles = list(angles)
        self.dihedrals = list(dihedrals)


class _Psf:
    """psf.view[":TLC"] -> _View; .bonds/.angles used for the ligand2 template."""

    def __init__(self, view, bonds=(), angles=()):
        self._view = view
        self.bonds = list(bonds)
        self.angles = list(angles)

    @property
    def view(self):
        return {":XXX": self._view}


def _make_cct(allow_breaking, view, ligand2_psf):
    cct = object.__new__(CommonCoreTransformation)
    cct.tlc_cc1 = "XXX"
    cct.allow_bonded_breaking = allow_breaking
    # both atom names map to themselves and are "in the cc"
    cct.atom_names_mapping = {"C1'": "C1'", "N1": "N1", "C2": "C2"}
    cct.ligand2_psf = ligand2_psf
    cct._psf = _Psf(view)  # the psf being mutated; type != AmberParm so no parmed calls
    return cct


def test_bond_breaks_to_zero_when_allowed():
    bond = _Bond(_Atom("C1'", "CN7B"), _Atom("N1", "NN2"), k=300.0, req=1.48)
    view = _View(bonds=[bond])
    cct = _make_cct(True, view, _Psf(_View(), bonds=[]))  # no counterpart in ligand2

    cct._mutate_bonds(cct._psf, lambda_value=1.0)
    assert bond.mod_type.k == pytest.approx(300.0)  # full at lambda=1
    assert bond.mod_type.req == pytest.approx(1.48)

    cct._mutate_bonds(cct._psf, lambda_value=0.5)
    assert bond.mod_type.k == pytest.approx(150.0)  # half-way

    cct._mutate_bonds(cct._psf, lambda_value=0.0)
    assert bond.mod_type.k == pytest.approx(0.0)  # gone at the common core


def test_bond_missing_counterpart_raises_when_not_allowed():
    bond = _Bond(_Atom("C1'", "CN7B"), _Atom("N1", "NN2"), k=300.0, req=1.48)
    cct = _make_cct(False, _View(bonds=[bond]), _Psf(_View(), bonds=[]))
    with pytest.raises(RuntimeError, match="No corresponding bond"):
        cct._mutate_bonds(cct._psf, lambda_value=0.5)


def test_angle_breaks_to_zero_when_allowed():
    angle = _Angle(_Atom("C2", "CN1"), _Atom("C1'", "CN7B"), _Atom("N1", "NN2"), k=50.0, theteq=109.5)
    cct = _make_cct(True, _View(angles=[angle]), _Psf(_View(), angles=[]))

    cct._mutate_angles(cct._psf, lambda_value=1.0)
    assert angle.mod_type.k == pytest.approx(50.0)
    assert angle.mod_type.theteq == pytest.approx(109.5)

    cct._mutate_angles(cct._psf, lambda_value=0.0)
    assert angle.mod_type.k == pytest.approx(0.0)


def test_angle_missing_counterpart_raises_when_not_allowed():
    angle = _Angle(_Atom("C2", "CN1"), _Atom("C1'", "CN7B"), _Atom("N1", "NN2"), k=50.0, theteq=109.5)
    cct = _make_cct(False, _View(angles=[angle]), _Psf(_View(), angles=[]))
    with pytest.raises(RuntimeError, match="No corresponding angle"):
        cct._mutate_angles(cct._psf, lambda_value=0.5)
