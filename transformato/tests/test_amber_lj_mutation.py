"""Regression tests for Lennard-Jones mutations on Amber/GAFF topologies.

These guard a subtle Amber-only correctness bug in the mutation code: when a
single atom's LJ parameters are changed, the change must apply to *that atom
only*. ParmEd's ``changeLJSingleType`` edits the whole ``nb_idx`` atom type, so
it leaks into every same-typed atom (corrupting the common-core LJ in every
intermediate state); ``addLJType`` instead splits a new type for just the
masked atom. The correct ``addLJType`` call also has to pass ``radius``/
``epsilon`` as keyword tokens -- positional values are silently dropped and the
atom keeps its original LJ.

The fixtures are tiny in-memory GAFF-like topologies built with ParmEd alone,
so these tests need no external CHARMM-GUI/Amber input tree. The observable is
the LJ as it ends up in the *written and reloaded* parm7 (the artifact that is
actually simulated); reading ``atom.rmin``/``atom.epsilon`` directly would give
a false pass because ``addLJType`` leaves the stale ``atom_type`` in place.
"""

import os
import tempfile

import parmed as pm
import pytest
from parmed import Atom, AtomType, Bond, BondType

from transformato.mutate import DummyRegion, Mutation

# GAFF-like reference parameters (rmin is Rmin/2, the ParmEd convention).
C3 = dict(eps=0.1094, rmin=1.9080)  # sp3 carbon
HC = dict(eps=0.0157, rmin=1.4870)  # aliphatic hydrogen


def _build_ethane_amberparm(tlc: str = "ETH") -> pm.amber.AmberParm:
    """Minimal GAFF-like ethane built entirely in memory (no external files).

    Both carbons share LJ type ``c3`` and all six hydrogens share ``hc``. That
    shared ``nb_idx`` is exactly what a per-type LJ edit would corrupt, so it is
    what lets these tests distinguish an atom-local edit from a type-wide one.
    """
    structure = pm.Structure()
    c3 = AtomType("c3", 1, 12.01, atomic_number=6)
    c3.set_lj_params(**C3)
    hc = AtomType("hc", 2, 1.008, atomic_number=1)
    hc.set_lj_params(**HC)

    names = ["C1", "C2", "H1", "H2", "H3", "H4", "H5", "H6"]
    types = [c3, c3, hc, hc, hc, hc, hc, hc]
    for name, atom_type in zip(names, types):
        atom = Atom(
            name=name,
            type=atom_type.name,
            atomic_number=atom_type.atomic_number,
            mass=atom_type.mass,
            charge=0.0,
        )
        atom.atom_type = atom_type
        atom.epsilon = atom_type.epsilon
        atom.rmin = atom_type.rmin
        structure.add_atom(atom, tlc, 1)

    bond_type = BondType(300.0, 1.5)
    structure.bond_types.append(bond_type)
    structure.bond_types.claim()
    for i, j in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]:
        structure.bonds.append(
            Bond(structure.atoms[i], structure.atoms[j], type=bond_type)
        )

    parm = pm.amber.AmberParm.from_structure(structure)
    parm.fill_LJ()

    # Mirror SystemStructure._determine_offset_and_set_possible_dummy_properties,
    # which the mutation code relies on.
    parm.number_of_dummys = 0
    parm.mutations_to_default = 0
    for atom in parm.atoms:
        atom.initial_charge = atom.charge
        atom.initial_epsilon = atom.epsilon
        atom.initial_rmin = atom.rmin

    return parm


def _lj_from_written_parm7(parm: pm.amber.AmberParm, name: str):
    """Return (Rmin/2, epsilon) of atom ``name`` as serialized in the parm7."""
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "lig.parm7")
    parm.write_parm(path)
    reloaded = pm.load_file(path)
    atom = next(a for a in reloaded.atoms if a.name == name)
    return (round(atom.rmin, 4), round(atom.epsilon, 4))


@pytest.mark.amber
def test_vdw_to_default_is_atom_local_and_applied():
    """``_mutate_vdw`` (DDX 'mutate to default') must change only the targeted
    atom and must actually write rmin=1.5/eps=0.15.

    Catches both failure modes: the original ``changeLJSingleType`` (a same-type
    bystander also changes) and a positional ``addLJType`` call (the target is
    left unchanged because radius/epsilon are dropped)."""
    parm = _build_ethane_amberparm()
    dummy_region = DummyRegion(
        mol_name="m",
        match_termin_real_and_dummy_atoms={},
        connected_dummy_regions=[],
        tlc="ETH",
        lj_default=[],
    )
    mutation = Mutation(
        atoms_to_be_mutated=[2, 3, 4, 5, 6, 7], dummy_region=dummy_region
    )

    # Act on H1 (index 2) only.
    mutation._mutate_vdw(
        parm, lambda_value=0.0, vdw_atom_idx=[2], offset=0, to_default=True
    )

    assert _lj_from_written_parm7(parm, "H1") == (
        1.5,
        0.15,
    ), "DDX LJ was not applied to the target atom"
    assert _lj_from_written_parm7(parm, "H4") == (
        HC["rmin"],
        HC["eps"],
    ), "LJ of a same-typed bystander atom leaked"


@pytest.mark.amber
def test_vdw_to_dummy_scales_only_target():
    """``_mutate_vdw`` to a dummy (lambda=0) must zero the target's LJ while
    leaving same-typed bystanders untouched."""
    parm = _build_ethane_amberparm()
    dummy_region = DummyRegion(
        mol_name="m",
        match_termin_real_and_dummy_atoms={},
        connected_dummy_regions=[],
        tlc="ETH",
        lj_default=[],
    )
    mutation = Mutation(
        atoms_to_be_mutated=[2, 3, 4, 5, 6, 7], dummy_region=dummy_region
    )

    mutation._mutate_vdw(
        parm, lambda_value=0.0, vdw_atom_idx=[2], offset=0, to_default=False
    )

    assert _lj_from_written_parm7(parm, "H1") == (
        0.0,
        0.0,
    ), "dummy LJ was not zeroed on the target atom"
    assert _lj_from_written_parm7(parm, "H4") == (
        HC["rmin"],
        HC["eps"],
    ), "LJ of a same-typed bystander leaked"


@pytest.mark.amber
@pytest.mark.filterwarnings("error::parmed.tools.exceptions.UnhandledArgumentWarning")
def test_addljtype_call_contract():
    """Locks in the correct ``addLJType`` call pattern used in mutate.py.

    Keyword ``radius=``/``epsilon=`` apply both values to only the masked atom.
    The typo (``espsilon=``) and positional forms emit an
    ``UnhandledArgumentWarning`` (here promoted to an error) and silently fail
    to apply the value -- this test fails if the calls ever regress to them."""
    parm = _build_ethane_amberparm()
    pm.tools.actions.addLJType(parm, ":1@3", radius=1.0, epsilon=0.05).execute()
    parm.load_atom_info()

    assert _lj_from_written_parm7(parm, "H1") == (
        1.0,
        0.05,
    ), "addLJType did not apply rmin and epsilon"
    assert _lj_from_written_parm7(parm, "H2") == (
        HC["rmin"],
        HC["eps"],
    ), "addLJType leaked into a bystander atom"
