"""Unit tests for common-core bond/angle/torsion *forming* (Phase 1b of the
pseudouridine / bond-migration work -- see DESIGN_pseudouridine.md).

A common-core bonded term that exists in the target ligand (cc2) but not in the
ligand being mutated (cc1) -- the other half of a migrating bond -- must be
inserted into the psf and ramped on (k: 0 -> full) over lambda. These tests use
small hand-built parmed Structures so the *real* insertion path runs, and check:
the term appears, k ramps correctly, the geometry target is the cc2 value, and
re-applying across lambda-states is idempotent (no duplicate terms).
"""

import parmed as pm
import pytest

from transformato.mutate import CommonCoreTransformation


def _ligand(atom_specs, bonds=(), angles=(), dihedrals=()):
    """atom_specs: list of (name, type). bonds: (n1, n2, k, req). etc."""
    s = pm.Structure()
    at = {}
    for name, typ in atom_specs:
        a = pm.Atom(name=name, type=typ, charge=0.0, mass=12.0)
        s.add_atom(a, "LIG", 1)
        at[name] = a
    for n1, n2, k, req in bonds:
        bt = pm.BondType(k, req)
        s.bond_types.append(bt)
        s.bonds.append(pm.Bond(at[n1], at[n2], type=bt))
    for n1, n2, n3, k, th in angles:
        ang_t = pm.AngleType(k, th)
        s.angle_types.append(ang_t)
        s.angles.append(pm.Angle(at[n1], at[n2], at[n3], type=ang_t))
    for n1, n2, n3, n4, phi_k, per, phase in dihedrals:
        dt = pm.DihedralType(phi_k, per, phase, 1.0, 1.0)
        s.dihedral_types.append(dt)
        s.dihedrals.append(pm.Dihedral(at[n1], at[n2], at[n3], at[n4], type=dt))
    return s


ATOMS = [("C1'", "CN7B"), ("N1", "NN2"), ("C5", "CN3"), ("C2", "CN1"), ("O4'", "ON6")]
# identity name mapping for the common core
MAPPING = {n: n for n, _ in ATOMS}


def _make_cct(cc1, cc2):
    cct = object.__new__(CommonCoreTransformation)
    cct.tlc_cc1 = "LIG"
    cct.allow_bonded_breaking = True
    cct.atom_names_mapping = dict(MAPPING)
    cct.ligand2_psf = cc2
    return cct


def test_forming_bond_ramps_on_and_is_idempotent():
    # cc1 (uridine-like): glycosidic bond C1'-N1; cc2 (Psi-like): C1'-C5
    cc1 = _ligand(ATOMS, bonds=[("C1'", "N1", 300.0, 1.47)])
    cc2 = _ligand(ATOMS, bonds=[("C1'", "C5", 250.0, 1.50)])
    cct = _make_cct(cc1, cc2)

    def forming_bond():
        return [b for b in cc1.bonds if getattr(b, "_tf_forming", False)]

    # lambda = 1.0 -> cc1 endstate: forming bond present but off
    cct._mutate_bonds(cc1, 1.0)
    assert len(forming_bond()) == 1
    fb = forming_bond()[0]
    assert {fb.atom1.name, fb.atom2.name} == {"C1'", "C5"}
    assert fb.mod_type.k == pytest.approx(0.0)
    assert fb.mod_type.req == pytest.approx(1.50)  # geometry target = cc2

    # the breaking bond (C1'-N1) is full here
    bb = [b for b in cc1.bonds if {b.atom1.name, b.atom2.name} == {"C1'", "N1"}][0]
    assert bb.mod_type.k == pytest.approx(300.0)

    # lambda = 0.0 -> common core: forming bond full, breaking bond gone
    cct._mutate_bonds(cc1, 0.0)
    assert len(forming_bond()) == 1  # idempotent: no duplicate inserted
    assert forming_bond()[0].mod_type.k == pytest.approx(250.0)
    assert bb.mod_type.k == pytest.approx(0.0)

    # midpoint: both partially on (overlapping linear -> base never detaches)
    cct._mutate_bonds(cc1, 0.5)
    assert len(forming_bond()) == 1
    assert forming_bond()[0].mod_type.k == pytest.approx(125.0)
    assert bb.mod_type.k == pytest.approx(150.0)


def test_forming_angle_ramps_on_and_is_idempotent():
    cc1 = _ligand(ATOMS, angles=[("C2", "C1'", "N1", 50.0, 110.0)])
    cc2 = _ligand(ATOMS, angles=[("C2", "C1'", "C5", 40.0, 112.0)])
    cct = _make_cct(cc1, cc2)

    def forming():
        return [a for a in cc1.angles if getattr(a, "_tf_forming", False)]

    cct._mutate_angles(cc1, 1.0)
    assert len(forming()) == 1
    assert forming()[0].mod_type.k == pytest.approx(0.0)
    assert forming()[0].mod_type.theteq == pytest.approx(112.0)

    cct._mutate_angles(cc1, 0.0)
    assert len(forming()) == 1
    assert forming()[0].mod_type.k == pytest.approx(40.0)


def test_forming_torsion_uses_sequenced_schedule_and_is_idempotent():
    cc1 = _ligand(ATOMS, dihedrals=[("O4'", "C1'", "N1", "C2", 1.0, 2, 180.0)])
    cc2 = _ligand(ATOMS, dihedrals=[("O4'", "C1'", "C5", "C2", 2.0, 2, 180.0)])
    cct = _make_cct(cc1, cc2)

    def forming():
        return [d for d in cc1.dihedrals if getattr(d, "_tf_forming", False)]

    # lambda >= 0.5 -> forming torsion stays off (sequenced: comes on in 2nd half)
    cct._mutate_torsions(cc1, 1.0)
    assert len(forming()) == 1
    assert forming()[0].mod_type == []  # f = 0

    cct._mutate_torsions(cc1, 0.5)
    assert len(forming()) == 1
    assert forming()[0].mod_type == []  # still off at 0.5

    # lambda = 0.0 -> fully on
    cct._mutate_torsions(cc1, 0.0)
    assert len(forming()) == 1
    assert len(forming()[0].mod_type) == 1
    assert forming()[0].mod_type[0].phi_k == pytest.approx(2.0)


def test_no_forming_when_topologies_match():
    # same bond in both -> nothing is formed
    cc1 = _ligand(ATOMS, bonds=[("C1'", "N1", 300.0, 1.47)])
    cc2 = _ligand(ATOMS, bonds=[("C1'", "N1", 320.0, 1.46)])
    cct = _make_cct(cc1, cc2)
    cct._mutate_bonds(cc1, 0.5)
    assert [b for b in cc1.bonds if getattr(b, "_tf_forming", False)] == []
