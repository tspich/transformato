"""Unit tests for the explicit common-core mapping (Phase 0 of the
pseudouridine / bond-migration work -- see DESIGN_pseudouridine.md).

These tests exercise ProposeMutationRoute.set_common_core_mapping in isolation,
without building real SystemStructure objects, by injecting the few attributes
the method touches (mirrors the object.__new__ approach in test_pymbar_compat).
"""

import pytest

from transformato.mutate import ProposeMutationRoute


class _FakeMol:
    """Minimal stand-in for an RDKit mol -- only GetNumAtoms is used."""

    def __init__(self, n):
        self._n = n

    def GetNumAtoms(self):
        return self._n


def _make_route(n1=10, n2=11):
    route = object.__new__(ProposeMutationRoute)
    route.asfe = False
    route.mols = {"m1": _FakeMol(n1), "m2": _FakeMol(n2)}
    route._substructure_match = {"m1": [], "m2": []}
    route.added_indeces = {"m1": [], "m2": []}
    route.removed_indeces = {"m1": [], "m2": []}
    return route


def test_mapping_sets_parallel_substructure_match():
    route = _make_route()
    mapping = [(0, 1), (2, 3), (4, 0)]
    route.set_common_core_mapping(mapping)

    assert route._substructure_match["m1"] == [0, 2, 4]
    assert route._substructure_match["m2"] == [1, 3, 0]
    # the public getters derive from the substructure match and preserve order
    assert route.get_common_core_idx_mol1() == [0, 2, 4]
    assert route.get_common_core_idx_mol2() == [1, 3, 0]


def test_mapping_accepts_dict():
    route = _make_route()
    route.set_common_core_mapping({0: 0, 1: 2, 3: 4})
    assert route._substructure_match["m1"] == [0, 1, 3]
    assert route._substructure_match["m2"] == [0, 2, 4]


def test_mapping_resets_prior_manual_edits():
    route = _make_route()
    route.added_indeces = {"m1": [9], "m2": [9]}
    route.removed_indeces = {"m1": [1], "m2": [1]}
    route.set_common_core_mapping([(0, 0), (2, 2)])
    # the explicit mapping is authoritative
    assert route.added_indeces == {"m1": [], "m2": []}
    assert route.removed_indeces == {"m1": [], "m2": []}
    assert route.get_common_core_idx_mol1() == [0, 2]


def test_mapping_rejects_out_of_range_index():
    route = _make_route(n1=5, n2=5)
    with pytest.raises(ValueError, match="out of range"):
        route.set_common_core_mapping([(0, 0), (5, 1)])  # 5 >= n1


def test_mapping_rejects_duplicate_index():
    route = _make_route()
    with pytest.raises(ValueError, match="duplicate"):
        route.set_common_core_mapping([(0, 0), (0, 1)])  # 0 twice in mol1


def test_mapping_rejects_empty():
    route = _make_route()
    with pytest.raises(ValueError, match="empty"):
        route.set_common_core_mapping([])


def test_mapping_rejects_asfe():
    route = _make_route()
    route.asfe = True
    with pytest.raises(RuntimeError, match="ASFE"):
        route.set_common_core_mapping([(0, 0)])
