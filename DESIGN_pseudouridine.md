# Design: pseudouridine (Ψ ↔ U) and bond-migrating mutations in transformato

Status: **in progress** (branch `PSU_cc`). This document describes what
transformato needs in order to handle relative free-energy calculations between
uridine (U) and pseudouridine (Ψ), and more generally any mutation where a bond
*moves* between two atoms that both stay real.

### Implementation status

| Phase | What | State |
|---|---|---|
| 0 | Explicit common-core atom mapping (bypass MCS) | **done** — `ProposeMutationRoute.set_common_core_mapping`; tests in `tests/test_common_core_mapping.py` |
| 1a | Graceful bonded-term **breaking** (k→0 when no counterpart), gated by `allow_bonded_topology_change` | **done** — `_mutate_bonds/_angles/_torsions`; tests in `tests/test_bond_breaking.py` |
| 1b | Bonded-term **forming** (insert a bond present only in the other ligand) | **done for CHARMM** — `_form_missing_bonds/_angles/_torsions`; Amber raises `NotImplementedError`; tests in `tests/test_bond_forming.py` |
| 2 | Shared-endstate accounting / cycle closure | spec'd in §4 Phase 2 (architecture noted); test not written |
| 3 | Geometry & sampling safeguards | partial — schedule chosen (see 1b note); restraint not yet built |
| 4 | Validation on a nucleoside | not started — needs a Ψ/U test system; logic unit-tested, end-to-end not yet |

**Phase 1b implementation notes (2026-06-07).**
- Forming is gated by the same `allow_bonded_breaking` /
  `allow_bonded_topology_change` flag as breaking, so the MCS path is untouched
  (regression: `test_acetylacetone_tautomer_pair` still passes).
- **Idempotent insertion.** `write_state` mutates the *same* psf object across all
  λ-states, so forming terms are inserted find-or-create and tagged
  `_tf_forming`; the forward matching loops skip tagged terms. Re-applying across
  λ does not duplicate.
- **Schedule (deviation from the §5 "sequenced" plan, on purpose).** Forming
  bonds/angles use an **overlapping linear** ramp (`k = (1−λ)·k_cc2`), the mirror
  of the committed linear breaking (`k = λ·k_cc1`). At every λ the base is held by
  *some* glycosidic force constant, so it cannot dissociate — this is runnable
  **without** the restraint that a fully-sequenced (detached-at-midpoint) schedule
  would require. The midpoint is mildly doubly-tethered (strained 4-membered
  C1′–N1–C6–C5 ring) instead. Forming torsions keep the **sequenced** factor
  (`f = 1 − min(2λ, 1)`), matching the existing matched-torsion convention.
  Revisit once the Phase 3 restraint exists: if the strained midpoint hurts MBAR
  overlap, switch bonds/angles to sequenced + restraint.
- **Amber not yet supported** for forming: inserting a bond into an `AmberParm`
  needs `remake_parm` + exclusion-list rebuild (§4 Phase 1b step 5). Forming on an
  Amber topology raises `NotImplementedError`; use CHARMM for Ψ↔U for now.
- **Still unvalidated end-to-end:** the parmed insertion + CHARMM writer path is
  exercised by unit tests on hand-built structures, but not yet on a real Ψ/U
  system. Phase 4 must confirm the written `.psf`/`dummy_parameters.prm` and
  cycle closure.

**Approach decision (2026-06-07): Option A** (direct single-topology bond
migration), with the sequenced break/form λ schedule as the geometry mitigation;
Option B kept only as a fallback. Full rationale in §5.

**Key finding while implementing 1a (CHARMM path).** The CHARMM writer
(`state.py` `_write_*`, the `dummy_parameters.prm` block) emits modified bonded
parameters *only for bonds already present in `psf.view.bonds`* and keys them by
atom-type pair. Consequences:
- **Breaking** a bond needs no topology deletion: a `k=0` bond is energetically
  identical to "no bond," and both legs' common-core endstates then evaluate to
  the same energy. This is what 1a does. ✅
- **Forming** a bond (1b) genuinely requires inserting the bond (and its
  angles/torsions) into mol1's psf so the writer emits it, plus handling
  type-keyed parameter collisions and PSF connectivity. That is the remaining
  hard part and is unavoidable for Ψ↔U (at the common-core endstate the new
  glycosidic bond must be physically present at full force constant).

---

## 1. The chemistry

U and Ψ have almost the same atom set but differ in **one bond** (a *bond
migration*), plus one swapped hydrogen and a few atom types:

| | Uridine (U) | Pseudouridine (Ψ) |
|---|---|---|
| Glycosidic bond | **N1 — C1′** (N-glycoside) | **C5 — C1′** (C-glycoside) |
| C5 | carries **H5** | bonded to sugar, no H5 |
| N1 | substituted, no proton | carries **H1** (extra imino donor) |
| N3, C6, carbonyls O2/O4, sugar–phosphate backbone | common | common |

Aligning the rings, the *only* real differences are:

1. the glycosidic bond migrates from **N1→C1′** to **C5→C1′**;
2. one hydrogen swaps: **H5** (U only) ↔ **H1** (Ψ only);
3. atom types / partial charges change on N1, C5 (and slightly C2, C4, C6).

Everything else — the full uracil ring, both carbonyls, H3, H6, and the entire
ribose–phosphate backbone — is common.

> Ψ has **two** imino donors (N1–H and N3–H); U has only N3–H. The extra N1–H is
> the origin of Ψ's water-mediated stabilization (see §6).

## 2. Why transformato cannot do this today

Three constraints in the code, in increasing severity:

1. **MCS returns a single connected substructure and will not cross a mismatched
   bond.** `ProposeMutationRoute._find_mcs` (`transformato/mutate.py:1078`) runs
   `rdFMCS.FindMCS(..., completeRingsOnly=True, ringMatchesRingOnly=True)` on
   H-stripped mols and takes one `GetSubstructMatch`. Because the sugar↔base bond
   connects *different ring atoms* in U vs Ψ, the largest connected common piece
   is the **backbone only**, and the whole base is pushed into the dummy region.
   This is the observed "the whole base is not part of the common core."

2. **The common core must have identical bond topology between the two ligands.**
   This is the real blocker. `CommonCoreTransformation._mutate_bonds`
   (`transformato/mutate.py:1790`) iterates over every common-core bond of
   ligand 1 and *requires* a matching common-core bond in ligand 2; otherwise it
   does `raise RuntimeError("No corresponding bond in cc2 found")`
   (`transformato/mutate.py:1864`). The same one-to-one assumption holds for
   `_mutate_angles` and `_mutate_torsions`. A migrated glycosidic bond
   (C1′–N1 on one side, C1′–C5 on the other) has **no counterpart**, so it cannot
   be represented. Transformato can interpolate the *parameters* (type, charge,
   `k`, `req`) of an **existing** common-core bond between the two ligands, but it
   cannot **create or destroy** a bond on atoms that remain real.

3. **Dummifying the whole base is physically poor even when it runs.** Destroying
   and rebuilding an aromatic, conjugated, carbonyl-bearing ring — in a *flipped*
   orientation — is an enormous perturbation: weak phase-space overlap between
   λ-windows, slow convergence, and (critically for Ψ) it tears down the water
   structure that matters (§6).

**Conclusion.** The missing capability is **alchemical bond making/breaking on
real common-core atoms** (single-topology bond migration). It is *not* "change
one atom into another" — that already works inside the common core
(`_mutate_atoms`/`_mutate_bonds` interpolate types, charges and bonded
parameters between the two ligands).

## 3. The minimal path we want

Keep **backbone + the entire uracil ring** as a real common core. The whole
transformation then reduces to:

- **2 true dummy atoms total:** H5 (U-leg only) and H1 (Ψ-leg only) — small
  terminal dummies, the case transformato already handles well;
- **1 migrating bond:** C1′–N1 ⇄ C1′–C5, plus the angles/dihedrals riding on it;
- **type/charge morphing** on N1, C5, C2, C4, C6 — already supported.

The conceptual cost collapses from "rebuild a ring" to "swap a proton + slide one
bond."

## 4. Implementation plan (phased)

### Phase 0 — Atom mapping (bypass MCS)
MCS will never produce the ring↔ring map here. Add an explicit-mapping entry
point to `ProposeMutationRoute`: accept a user- or graph-supplied `cc1 ↔ cc2`
atom-index map (ring atoms by identity, backbone by identity) and skip
`_find_mcs`. Hooks already exist — `add_idx_to_common_core_of_mol1/2`,
`remove_idx_from_common_core_of_mol1/2`, `_set_common_core_parameters` — so wire a
"mapping provided" branch through `propose_common_core` / `finish_common_core`.
A ring graph-isomorphism (networkx, already a dependency) can auto-generate the
map for the U/Ψ ring.

### Phase 1a — Alchemical bonded-term *breaking* (done)
Generalize `_mutate_bonds` / `_mutate_angles` / `_mutate_torsions` so that a term
present in mol1 but absent in mol2 is **scaled to `k = 0`** instead of raising.
When `found == False`, hold `req`/`theteq` at the mol1 geometry and interpolate
`k(λ): full → 0`. Gated by `allow_bonded_breaking` (off by default → MCS path
unchanged). **Implemented**; tests in `tests/test_bond_breaking.py`.

A broken bond needs **no topology deletion**: a `k=0` bond contributes no energy,
so it is equivalent to "no bond" and the two legs' common-core endstates still
evaluate equal. (Confirmed against the CHARMM `dummy_parameters.prm` writer.)

### Phase 1b — Alchemical bonded-term *forming* (next, the hard half)

This is the asymmetric, harder direction: a bond present in **mol2** but **not**
in **mol1** must appear in mol1's psf with `k(λ): 0 → full`. Unlike breaking, the
writer only emits bonds that already exist in `psf.view.bonds`, so the bond (and
its angles/torsions) must be **inserted into the topology**. Concrete steps:

1. **Discover the forming terms.** After the existing mol1→mol2 matching loops,
   add a reverse pass: iterate `ligand2_psf.bonds` (then angles, then dihedrals)
   whose atoms are all in the common core, map their names back to mol1 via the
   inverse of `atom_names_mapping`, and collect any term whose mapped mol1 atoms
   are **not** already bonded/angled/dihedraled. Those are the forming terms.

2. **Insert into the parmed structure (mol1 psf).** For each forming term, create
   the parmed object between the corresponding mol1 atoms and append it:
   - bond: `pm.Bond(a_i, a_j, type=pm.BondType(k, req))`, append to `psf.bonds`
     and the type to `psf.bond_types`;
   - angle / dihedral analogously (`pm.Angle`, `pm.Dihedral` + their `*_types`).
   Set `req`/`theteq`/`phase`/`per` to the **mol2** target values; these are
   constant over λ. Only `k`/`phi_k` are scaled.

3. **Ramp the force constant on.** Give each inserted term a `mod_type` with
   `k(λ) = (1 − λ) · k_mol2` (0 at λ=1 = mol1 endstate, full at λ=0 = CC), using
   the **same sequenced schedule** as breaking so the two never overlap: break
   over λ∈[1.0, 0.5], form over λ∈[0.5, 0.0] (reuse the torsion
   `f = max(1 − (1−λ)·2, 0)` / `f = 1 − min(λ·2, 1)` factors). The base is then
   never doubly-tethered.

4. **Make the CHARMM writer emit the inserted term.** `state.py` writes a bonded
   term to `dummy_parameters.prm` only if one of its atoms `hasattr(initial_type)`
   (a dummy/changed-type atom) — keyed by **atom-type pair**. The junction atoms
   (N1, C5) change type during the transform, so they already get `initial_type`;
   verify the inserted C1′–C5 term is therefore written. **Watch for type-pair
   collisions:** params are keyed by `(typeA, typeB)`, so if the inserted bond's
   type pair coincides with an existing real bond's pair, the `already_seen`
   dedup will drop one. The junction atoms get unique `RRR*` types via
   `_modify_type_in_cc`, which should keep the pair unique — confirm on a real
   case.

5. **Amber path.** parmed `setBond/setAngle/addDihedral` *modify* existing terms;
   for a **new** bond, append the parmed objects (step 2) and call
   `psf.remake_parm()` so the prmtop arrays (and the **exclusion list**) are
   rebuilt before `write_parm`.

6. **Nonbonded exclusions (both engines, easy to miss).** Forming C1′–C5 creates
   new 1-2/1-3/1-4 neighbor relationships → new exclusions / 1-4 scaling between
   C1′ (and its sugar neighbors) and C5 (and its ring neighbors). Exclusions are
   binary, not λ-scaled, so decide the convention: simplest correct choice is to
   apply the **mol2 (CC) exclusion topology** for the whole forming window and let
   the soft `k`-ramp + restraints absorb the geometry. Validate energies at the
   λ=0 endstate equal a natively-built Ψ common core (cycle-closure test).

7. **Reverse-map helper.** Add an inverse of `_get_atom_mapping`
   (`cc2_name → cc1_name`) so steps 1–2 can translate mol2 terms onto mol1 atoms.

Gate all of this behind the same `allow_bonded_breaking` /
`allow_bonded_topology_change` flag (consider renaming to
`allow_bonded_topology_change` end-to-end since it now covers forming too).

### Phase 2 — Define the shared common-core endstate

**Architecture reminder (important for getting this right).** In transformato the
common-core endstate carries **mol2's** parameters: `generate_..._for_mol1` sheds
mol1's dummies **and** runs `_transform_common_core` (morph mol1 → mol2 params,
including bonds/angles/torsions via `CommonCoreTransformation`), while
`generate_..._for_mol2` only sheds mol2's dummies. So **the entire bond migration
happens on the mol1 leg** — it is not split across the two legs.

Take **mol1 = U, mol2 = Ψ** (so the common core uses Ψ's C1′–C5 connection):

- **U-leg (mol1):** turn off H5 (terminal dummy); then in the transform: **break
  C1′–N1** (Phase 1a, k→0) **and form C1′–C5** (Phase 1b, k 0→full), and morph
  N1/C5/C2/C4/C6 types and charges (existing machinery).
- **Ψ-leg (mol2):** turn off H1 (terminal dummy) only — Ψ is already at the CC.

At the endstate both legs present the same *interacting* common core: C1′–C5 at
full Ψ force constant, C1′–N1 at k=0 (energetically identical to absent), N1/C5
types = Ψ. H5 (U-leg) and H1 (Ψ-leg) are non-interacting dummies whose internal
terms cancel in the cycle, so ΔΔG = leg1 − leg2 stays exact.

> The direction matters only for code flow, not for the result: if instead
> mol1 = Ψ, mol2 = U, the migration (break C1′–C5, form C1′–N1) runs on the Ψ leg
> and the CC uses U's connection. Either assignment must work; 1b must handle both
> breaking and forming in the same transform.

This is the correctness-sensitive step — add a cycle-closure test (Phase 4).

### Phase 3 — Geometry & sampling safeguards
Bond migration moves the base relative to the sugar, so:
- soft-core / softened force constant on the forming–breaking bond and its angles
  to avoid clashes at intermediate λ;
- finer λ spacing across the junction window (the workflow already supports
  per-step λ counts);
- optional positional/orientational restraints on the base during the junction
  step, removed analytically.

### Phase 4 — Validation
Start in **vacuum with a Ψ/U nucleoside** (no chain); check thermodynamic-cycle
closure and MBAR overlap matrices before touching a solvated strand. Add a
regression test (analogous to `transformato/tests/test_amber_lj_mutation.py`)
asserting that bond-scaling gives `k = 0` at the right endpoint and that a
forming/breaking bond round-trips.

## 5. Options and recommendation

| Option | What | Cost | Risk |
|---|---|---|---|
| **A. Full bond make/break feature** (Phases 0–4) | General single-topology bond migration | High (core engine change) | Junction sampling geometry |
| **B. Indirect cycle via a common reference** | Mutate *both* U and Ψ to a shared intermediate (abasic/capped, or fully decoupled base once) so the bond never migrates directly | Medium-high (more simulations, no engine change) | More windows; reference choice |
| **C. Manual dual-topology, base stays real, morph N1/C5 + 1 bond** | Phase 0 + Phase 1 only, hand-built map | Lowest code | Same junction geometry risk, less automation |

### Decision (2026-06-07): go with A

**A is chosen.** C was the bootstrap (Phase 0 + 1a) and is now done; A is C
completed with the *forming* half (Phase 1b) so the migration is symmetric. B is
kept only as a fallback.

Rationale:

- **B does not actually avoid the hard thing.** Its only distinct variants are
  either (i) "decouple the whole base" — which *is* the poor-convergence path that
  motivated this work — or (ii) "break the bond, restrain the base, re-form it at
  C5" — which is A's machinery plus a restraint apparatus and roughly double the
  simulations. Same physics, more parts.
- **A is the smallest delta from where we are.** Phase 0 (mapping) and Phase 1a
  (breaking) exist; 1b is the mirror image, and `CommonCoreTransformation` already
  morphs types/charges. B would need a new reference-state + restraint +
  free-energy-correction layer transformato does not have.
- **A fits the engine as-is.** It yields one ordinary U↔Ψ mutation that drops into
  the rsfe/rbfe two-leg ΔΔG with no new analysis path.
- **A is reusable** for any bond-migrating mutation (other modified nucleotides,
  ring rearrangements); B is a one-off cycle design.
- **A's geometry risk is mitigated inside A** (see Phase 3): sequence the λ
  schedule — break C1′–N1 over the first half of λ, form C1′–C5 over the second —
  with the base lightly restrained during the swap. This is exactly the good part
  of B (never doubly-tethered) without B's separate reference state or doubled
  sampling, and it is idiomatic: `_mutate_torsions` already uses a half-off /
  half-on schedule.

**Fallback trigger.** If the Phase 4 nucleoside validation shows the junction
window will not converge (bad MBAR overlap across the break/form step that finer λ
+ restraints cannot fix), switch to **B in its break-then-restrain-then-form
form**, reusing the same insertion code from 1b.

## 6. Water-mediated stabilization of Ψ (address later)

Ψ's defining feature is the **extra N1–H** imino donor, which orders a bridging
water that often links to the backbone phosphate — the source of Ψ's
rigidifying/stabilizing effect. Two points, both of which favor the minimal path
above:

1. **In explicit solvent, MBAR already includes this for free — if it is
   sampled.** The free energy of the ordered/bridging water is part of ΔG as long
   as the windows are converged. The danger is purely sampling: water
   reorganization around a newly appearing donor is slow. Action items:
   (a) adequate per-λ sampling / consider enhanced sampling or longer
   equilibration around the junction; (b) **do not dummify the base** — the
   dummy-everything path destroys and rebuilds the hydration site nonphysically
   and will badly underestimate the effect. The minimal-perturbation path keeps
   the site intact throughout.

2. **To verify the effect is captured**, run hydration-site analysis comparing U
   vs Ψ: GIST or SSTMAP/WATsite on the trajectories, or 3D-RISM, to confirm a
   high-occupancy, long-residence water appears at N1–H/O2 in Ψ and is absent in
   U. If sampling is the bottleneck, options are GCMC/grand-canonical water
   insertion to guarantee occupancy of the bridging site, or a lightly restrained
   explicit bridging water corrected for analytically.

## 7. Suggested first concrete experiment

1. Build U and Ψ **nucleosides** (no phosphate chain) as CHARMM-GUI / amber
   inputs.
2. Hand-build the `cc1 ↔ cc2` map (ring + sugar identity; H5↔dummy, H1↔dummy).
3. Implement Phase 1 `k → 0` scaling behind a config flag so existing behavior is
   untouched.
4. Run the U-leg and Ψ-leg to the shared CC **in vacuum**, then in water.
5. Check: thermodynamic-cycle closure, MBAR overlap matrices, and that the H5/H1
   dummies and the migrating bond behave as intended.

---

### Code touch-points (for whoever implements this)

- `transformato/mutate.py:1078` — `_find_mcs` (Phase 0: explicit-mapping bypass)
- `transformato/mutate.py:445` — `_set_common_core_parameters` (Phase 0 wiring)
- `transformato/mutate.py:1790` — `_mutate_bonds` (Phase 1: `k → 0` scaling)
- `transformato/mutate.py:1864` — the `RuntimeError("No corresponding bond ...")`
  to replace with `k = 0` handling
- `transformato/mutate.py:1870` — `_mutate_angles` (Phase 1)
- `transformato/mutate.py:1953` — `_mutate_torsions` (Phase 1)
- `transformato/analysis.py:685` — `calculate_dG_to_common_core` (Phase 2:
  shared-endstate accounting / cycle closure)
- `transformato/tests/test_amber_lj_mutation.py` — pattern for the Phase 4 test
