"""Realistic Phase 4 scaffolding test: uridine (URA) <-> pseudouridine (PSU).

Uses the *real* CHARMM atom names and base connectivity (from
toppar top_all36_na.rtf for URA and toppar_all36_na_rna_modified.str for PSU) to
check that, given the name-based common-core mapping, the migrating glycosidic
bond is handled correctly among the full uracil ring:

  * uridine:        C1'-N1 glycosidic, C5 carries H5
  * pseudouridine:  C1'-C5 glycosidic, N1 carries H1

So with mol1 = URA, mol2 = PSU and H5/H1 as the swapped dummies, the URA->CC
transform must **break C1'-N1**, **form C1'-C5**, and leave the rest of the ring
(matched bonds/angles) intact.

This runs on hand-built parmed Structures (no CHARMM-GUI inputs needed); the
end-to-end run on a solvated/vacuum nucleoside is the remaining Phase 4 step.
"""

import numpy as np
import parmed as pm
import pytest

from transformato.mutate import CommonCoreTransformation, ProposeMutationRoute

# --- real CHARMM atom names/types for the uracil base + glycosidic anchor ---
URA_ATOMS = [
    ("C1'", "CN7B"),
    ("N1", "NN2B"),
    ("C2", "CN1T"),
    ("O2", "ON1"),
    ("N3", "NN2U"),
    ("H3", "HN2"),
    ("C4", "CN1"),
    ("O4", "ON1"),
    ("C5", "CN3"),
    ("H5", "HN3"),
    ("C6", "CN3"),
    ("H6", "HN3"),
]
URA_BONDS = [
    ("C1'", "N1"),
    ("N1", "C2"),
    ("N1", "C6"),
    ("C2", "O2"),
    ("C2", "N3"),
    ("N3", "H3"),
    ("N3", "C4"),
    ("C4", "O4"),
    ("C4", "C5"),
    ("C5", "C6"),
    ("C5", "H5"),
    ("C6", "H6"),
]
URA_ANGLES = [("C1'", "N1", "C2"), ("C1'", "N1", "C6"), ("N1", "C2", "O2")]

PSU_ATOMS = [
    ("C1'", "CN7B"),
    ("N1", "NG2R61"),
    ("H1", "HGP1"),
    ("C2", "CG2R63"),
    ("O2", "OG2D4"),
    ("N3", "NG2R61"),
    ("H3", "HGP1"),
    ("C4", "CG2R63"),
    ("O4", "OG2D4"),
    ("C5", "CG2R62"),
    ("C6", "CG2R62"),
    ("H6", "HGR62"),
]
PSU_BONDS = [
    ("C1'", "C5"),
    ("N1", "C2"),
    ("N1", "C6"),
    ("N1", "H1"),
    ("C2", "O2"),
    ("C2", "N3"),
    ("N3", "H3"),
    ("N3", "C4"),
    ("C4", "O4"),
    ("C4", "C5"),
    ("C5", "C6"),
    ("C6", "H6"),
]
PSU_ANGLES = [("C1'", "C5", "C4"), ("C1'", "C5", "C6"), ("N1", "C2", "O2")]


def _build(atoms, bonds, angles, dihedrals=(), resname="LIG", k_bond=300.0, k_ang=50.0, phi_k=2.0, coords=None):
    s = pm.Structure()
    at = {}
    for name, typ in atoms:
        a = pm.Atom(name=name, type=typ, charge=0.0, mass=12.0)
        s.add_atom(a, resname, 1)
        at[name] = a
    for n1, n2 in bonds:
        bt = pm.BondType(k_bond, 1.45)
        s.bond_types.append(bt)
        s.bonds.append(pm.Bond(at[n1], at[n2], type=bt))
    for n1, n2, n3 in angles:
        angt = pm.AngleType(k_ang, 110.0)
        s.angle_types.append(angt)
        s.angles.append(pm.Angle(at[n1], at[n2], at[n3], type=angt))
    for n1, n2, n3, n4 in dihedrals:
        dt = pm.DihedralType(phi_k, 2, 180.0, 1.2, 2.0)
        s.dihedral_types.append(dt)
        s.dihedrals.append(pm.Dihedral(at[n1], at[n2], at[n3], at[n4], type=dt))
    if coords is not None:
        s.coordinates = np.array([coords[name] for name, _ in atoms], dtype=float)
    return s


class _FakeMol:
    def __init__(self, n):
        self._n = n

    def GetNumAtoms(self):
        return self._n


# ---------------------------------------------------------------------------
# Phase 0: the name-based mapping helper produces the right common core
# ---------------------------------------------------------------------------
def test_mapping_by_name_excludes_swapped_hydrogens():
    ura = _build(URA_ATOMS, [], [])
    psu = _build(PSU_ATOMS, [], [])
    route = object.__new__(ProposeMutationRoute)
    route.asfe = False
    route.psfs = {"m1": ura, "m2": psu}
    route.mols = {"m1": _FakeMol(len(URA_ATOMS)), "m2": _FakeMol(len(PSU_ATOMS))}
    route._substructure_match = {"m1": [], "m2": []}
    route.added_indeces = {"m1": [], "m2": []}
    route.removed_indeces = {"m1": [], "m2": []}

    route.set_common_core_mapping_by_name(exclude_mol1=["H5"], exclude_mol2=["H1"])

    cc1_names = [URA_ATOMS[i][0] for i in route.get_common_core_idx_mol1()]
    cc2_names = [PSU_ATOMS[i][0] for i in route.get_common_core_idx_mol2()]
    # H5 (uridine) and H1 (pseudouridine) are left out; everything else is in
    assert "H5" not in cc1_names
    assert "H1" not in cc2_names
    assert set(cc1_names) == {"C1'", "N1", "C2", "O2", "N3", "H3", "C4", "O4", "C5", "C6", "H6"}
    # the mapping is name-parallel
    assert cc1_names == cc2_names
    assert route.allow_bonded_topology_change is True


# ---------------------------------------------------------------------------
# Phase 1: the migrating bond is correctly broken+formed within the full ring
# ---------------------------------------------------------------------------
def _make_cct(ura, psu):
    common = [n for n, _ in URA_ATOMS if n not in ("H5",)]
    cct = object.__new__(CommonCoreTransformation)
    cct.tlc_cc1 = "LIG"
    cct.allow_bonded_breaking = True
    cct.atom_names_mapping = {n: n for n in common}  # identity by name
    cct.ligand2_psf = psu
    return cct


def _bond(struct, a, b):
    hits = [x for x in struct.bonds if {x.atom1.name, x.atom2.name} == {a, b}]
    return hits[0] if hits else None


def test_glycosidic_bond_migrates_within_full_uracil_ring():
    ura = _build(URA_ATOMS, URA_BONDS, [])
    psu = _build(PSU_ATOMS, PSU_BONDS, [])
    cct = _make_cct(ura, psu)

    # lambda = 1.0  (uridine endstate)
    cct._mutate_bonds(ura, 1.0)
    assert _bond(ura, "C1'", "N1").mod_type.k == pytest.approx(300.0)  # breaking bond full
    formed = _bond(ura, "C1'", "C5")
    assert formed is not None and getattr(formed, "_tf_forming", False)
    assert formed.mod_type.k == pytest.approx(0.0)  # forming bond off

    # lambda = 0.0  (common core = pseudouridine connection)
    cct._mutate_bonds(ura, 0.0)
    assert _bond(ura, "C1'", "N1").mod_type.k == pytest.approx(0.0)  # broken
    assert _bond(ura, "C1'", "C5").mod_type.k == pytest.approx(300.0)  # formed full

    # exactly one forming bond inserted (idempotent across the two calls)
    assert sum(getattr(b, "_tf_forming", False) for b in ura.bonds) == 1

    # a matched ring bond (present in both) is never treated as forming
    assert not getattr(_bond(ura, "N1", "C2"), "_tf_forming", False)


def test_junction_angles_break_and_form():
    ura = _build(URA_ATOMS, URA_BONDS, URA_ANGLES)
    psu = _build(PSU_ATOMS, PSU_BONDS, PSU_ANGLES)
    cct = _make_cct(ura, psu)

    def angle(struct, names):
        want = set(names)
        hits = [a for a in struct.angles if {a.atom1.name, a.atom2.name, a.atom3.name} == want]
        return hits[0] if hits else None

    cct._mutate_angles(ura, 0.0)
    # breaking angle C1'-N1-C2 -> k 0
    assert angle(ura, ("C1'", "N1", "C2")).mod_type.k == pytest.approx(0.0)
    # forming angle C1'-C5-C4 inserted and full
    formed = angle(ura, ("C1'", "C5", "C4"))
    assert formed is not None and getattr(formed, "_tf_forming", False)
    assert formed.mod_type.k == pytest.approx(50.0)


def _torsion(struct, names):
    want = list(names)
    for d in struct.dihedrals:
        seq = [d.atom1.name, d.atom2.name, d.atom3.name, d.atom4.name]
        if seq == want or seq == want[::-1]:
            return d
    return None


def test_junction_torsions_always_carry_a_parameter():
    """Regression: a torsion that stays in the psf topology must keep a (possibly
    zero) parameter at *every* lambda.

    Forming and breaking torsions ramp on opposite halves of lambda. The bug:
    in the half where a junction torsion's force constant is 0, ``mod_type`` was
    set to an empty list while the dihedral remained in the topology -> state.py
    wrote no parameter line -> OpenMM raised MissingParameter at createSystem
    (only on the migration leg's first or last transform states). Here we assert
    ``mod_type`` is a non-empty list at lambda = 1.0 / 0.5 / 0.0 for both a
    forming torsion (C1'-C5-C4-O4, off in the first half) and a breaking torsion
    (O2-C2-N1-C1', off in the second half).
    """
    ura = _build(URA_ATOMS, URA_BONDS, [], dihedrals=[("O2", "C2", "N1", "C1'")])
    psu = _build(PSU_ATOMS, PSU_BONDS, [], dihedrals=[("C1'", "C5", "C4", "O4")])
    cct = _make_cct(ura, psu)

    for lam in (1.0, 0.5, 0.0):
        cct._mutate_torsions(ura, lam)

        formed = _torsion(ura, ["C1'", "C5", "C4", "O4"])
        assert formed is not None and getattr(formed, "_tf_forming", False)
        assert isinstance(formed.mod_type, list) and len(formed.mod_type) >= 1, (
            f"forming torsion has no parameter at lambda={lam} (would crash OpenMM)"
        )

        broken = _torsion(ura, ["O2", "C2", "N1", "C1'"])
        assert broken is not None  # breaking torsion stays in the topology
        assert isinstance(broken.mod_type, list) and len(broken.mod_type) >= 1, (
            f"breaking torsion has no parameter at lambda={lam} (would crash OpenMM)"
        )

    # exactly one forming torsion inserted across the repeated calls (idempotent)
    assert sum(getattr(d, "_tf_forming", False) for d in ura.dihedrals) == 1
    # at the common-core endpoint (lambda=0) the breaking torsion is fully off...
    assert max(t.phi_k for t in _torsion(ura, ["O2", "C2", "N1", "C1'"]).mod_type) == pytest.approx(0.0)
    # ...and the forming torsion is fully on (its cc2 phi_k)
    assert max(t.phi_k for t in _torsion(ura, ["C1'", "C5", "C4", "O4"]).mod_type) == pytest.approx(2.0)


def test_forming_bond_sequences_geometry_ahead_of_stiffness():
    """Regression: the forming bond must POSITION before it STIFFENS.

    Ramping req and k together (linearly) left, in the back half of the transform, a stiff
    spring whose rest length was still moving -- every window pinned to a distinct geometry
    and adjacent MBAR overlap collapsed (uridine states 11..22 overlapped < 0.03 down to 0).
    The sequenced schedule walks req from the measured initial separation (C1'-C5 = 3.6 A
    here) to the cc2 target (1.45 A) while k is soft, reaching the target by the positioning
    fraction (default lambda=0.5); only then does k climb to full, with req fixed. So req
    LEADS k: at lambda=0.5 req is already at target while k is still only ~K_SOFT of full.
    """
    coords = {name: (5.0 + 2.0 * i, 0.0, 0.0) for i, (name, _) in enumerate(URA_ATOMS)}
    coords["C1'"] = (0.0, 0.0, 0.0)
    coords["C5"] = (3.6, 0.0, 0.0)
    ura = _build(URA_ATOMS, URA_BONDS, [], coords=coords)
    psu = _build(PSU_ATOMS, PSU_BONDS, [])  # provides the forming C1'-C5 target (req 1.45, k 300)
    cct = _make_cct(ura, psu)

    r_init, target, k_full = 3.6, 1.45, 300.0
    k_soft = CommonCoreTransformation._FORMING_K_SOFT       # 0.1
    assert CommonCoreTransformation._FORMING_REQ_FRACTION == pytest.approx(0.5)

    def fb(lam):
        cct._mutate_bonds(ura, lam)
        return _bond(ura, "C1'", "C5").mod_type

    # lambda=1.0 (cc1 endstate): bond essentially absent -- req at the start, k ~0
    m = fb(1.0)
    assert getattr(_bond(ura, "C1'", "C5"), "_tf_forming", False)
    assert m.req == pytest.approx(r_init, abs=1e-6)
    assert m.k == pytest.approx(0.0, abs=1e-6)

    # halfway through positioning (p=0.25 -> lambda=0.75): req half-walked, k still tiny
    m = fb(0.75)
    assert m.req == pytest.approx(0.5 * r_init + 0.5 * target)
    assert m.k == pytest.approx(k_soft * 0.5 * k_full)        # 15.0

    # end of positioning (p=0.5 -> lambda=0.5): req AT target, k only soft -> req leads k
    m = fb(0.5)
    assert m.req == pytest.approx(target)
    assert m.k == pytest.approx(k_soft * k_full)             # 30.0
    assert m.k < 0.2 * k_full                                 # k clearly lags req here

    # common core (lambda=0.0): req at target, k full
    m = fb(0.0)
    assert m.req == pytest.approx(target)
    assert m.k == pytest.approx(k_full)

    # no coordinates -> req falls back to the target throughout (k schedule still applies)
    ura_nc = _build(URA_ATOMS, URA_BONDS, [])
    cct_nc = _make_cct(ura_nc, _build(PSU_ATOMS, PSU_BONDS, []))
    cct_nc._mutate_bonds(ura_nc, 1.0)
    assert _bond(ura_nc, "C1'", "C5").mod_type.req == pytest.approx(target)


def test_breaking_bond_releases_by_positioning_fraction():
    """Regression: the breaking bond must RELEASE early (front-loaded), reaching k=0 by the
    positioning fraction (default lambda=0.5) -- not only at the common core.

    Linear lambda*k kept the breaking C1'-N1 bond near full strength through the positioning
    phase, pinning C1' on the N1 side until it abruptly let go (the (11,12) overlap gap), and
    left a force-constant change at the final window (the (21,22) gap). Now it releases in
    lockstep with the forming bond's req descent (and the breaking torsions), so it is fully
    gone before the stiffening phase and before the common core.
    """
    ura = _build(URA_ATOMS, URA_BONDS, [])
    psu = _build(PSU_ATOMS, PSU_BONDS, [])
    cct = _make_cct(ura, psu)
    k_full = 300.0  # _build's BondType k
    s = CommonCoreTransformation._FORMING_REQ_FRACTION  # 0.5

    def bk(lam):
        cct._mutate_bonds(ura, lam)
        return _bond(ura, "C1'", "N1").mod_type.k

    assert bk(1.0) == pytest.approx(k_full)                      # full at the start
    assert bk(0.75) == pytest.approx((1.0 - 0.25 / s) * k_full)  # 150.0, releasing
    assert bk(0.5) == pytest.approx(0.0)                         # released by positioning fraction
    assert bk(0.25) == pytest.approx(0.0)                        # stays released
    assert bk(0.0) == pytest.approx(0.0)                         # fully broken at the common core


def test_cc_lambda_schedule_densifies_commitment_band():
    """Non-uniform cc-transform schedule: uniform by default, but n_extra inserts windows
    only inside the [center-halfwidth, center+halfwidth] band -- used to densify the
    migration commitment point (lambda~0.5) that a uniform grid leaves at ~0 overlap."""
    from transformato.mutate import _cc_lambda_schedule

    uni = _cc_lambda_schedule(30)
    assert uni == _cc_lambda_schedule(30, n_extra=0)        # n_extra=0 == uniform
    assert len(uni) == 30
    assert max(uni) < 1.0 and min(uni) == 0.0              # 1.0 dropped, 0.0 kept
    assert uni == sorted(set(uni), reverse=True)           # descending, unique

    dens = _cc_lambda_schedule(30, n_extra=8, center=0.5, halfwidth=0.1)
    assert len(dens) > len(uni)
    assert dens == sorted(set(dens), reverse=True)
    extra = set(dens) - set(uni)
    assert extra and all(0.4 <= x <= 0.6 for x in extra)   # additions only inside the band
    assert any(0.5 < x < 0.533333 for x in dens)           # bridges the (0.533->0.5) gap
