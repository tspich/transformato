"""
File created in analogy to CHARMM-GUI (http://www.charmm-gui.org)
Last update: November, 2023
"""

from __future__ import print_function
import argparse
import sys
import os

from omm_readinputs import *
from omm_readparams import *
from omm_vfswitch import *

import openmm.unit as unit
from openmm.unit import *
from openmm import *
from openmm.app import *

parser = argparse.ArgumentParser()
parser.add_argument("-odcd", metavar="DCDFILE", dest="odcd")
parser.add_argument("-env", metavar="ENVIRONMENT", dest="env")
args = parser.parse_args()

# Resubmission after a time-out: a run that finished leaves <dcd>.done and is skipped,
# so a resubmitted job only redoes the run that was interrupted (from its start; a partial
# dcd without marker is overwritten). analyze.py warns about dcds without marker.
DONE_MARKER = f"{args.odcd}.done"
if os.path.isfile(DONE_MARKER):
    print(f"{args.odcd} already finished ({DONE_MARKER} exists) -- skipping")
    sys.exit(0)

# Short equilibration before production (not written to the trajectory): velocities at the
# simulation temperature, EQUIL_RESTRAINED_PS with the solute heavy atoms harmonically held at
# the minimized positions (removes clashes / lets water and ions relax around the RNA), then
# EQUIL_FREE_PS unrestrained. Set both to 0 to disable. The positional restraint is removed
# from the System before production, so the serialized _system.xml used for MBAR never
# contains it.
EQUIL_RESTRAINED_PS = 50.0
EQUIL_FREE_PS = 50.0
EQUIL_K = 1000.0  # kJ/mol/nm^2
_NON_SOLUTE = {"HOH", "WAT", "TIP3", "SOD", "CLA", "POT", "NA", "CL", "K", "NA+", "CL-", "K+", "MG"}

# Load parameters
env = args.env
print(f"Loading parameters in this environment {env}")


inputs = read_inputs(f"lig_in_{env}.inp")

if os.path.isfile(f"lig_in_{env}.parm7"):
    top = AmberPrmtopFile(f"lig_in_{env}.parm7")
    crd = AmberInpcrdFile(f"lig_in_{env}.rst7")
    fftype = "amber"
else:
    fftype = "charmm"
    params = read_params("toppar.str")
    top = CharmmPsfFile(f"lig_in_{env}.psf")
    crd = read_crd(f"lig_in_{env}.crd")
    top = gen_box(top, crd)

# Build system
if env == "waterbox" or env == "complex":
    nboptions = dict(
        nonbondedMethod=inputs.coulomb,
        nonbondedCutoff=inputs.r_off * unit.nanometers,
        constraints=inputs.cons,
        ewaldErrorTolerance=inputs.ewald_Tol,
    )
elif env == "vacuum":
    nboptions = dict(
        nonbondedMethod=NoCutoff,
        constraints=inputs.cons,
    )
print(f"Applying the following nonbonded options {nboptions}")

if inputs.vdw == "Switch" and env != "vacuum":
    print(f"Setting the vdw switching function to the defalut Openmm Switch")
    nboptions["switchDistance"] = inputs.r_on * unit.nanometers
if inputs.vdw == "LJPME" and env != "vacuum":
    print(f"Using LJPME for the vdw long range interactions")
    nboptions["nonbondedMethod"] = LJPME
if fftype == "amber":
    system = top.createSystem(**nboptions)
else:
    system = top.createSystem(params, **nboptions)


if inputs.vdw == "Force-switch" and fftype != "amber" and env != "vacuum":
    print(f"Setting the vdw switching function to: Force-switch")
    system = vfswitch(system, top, inputs)
if hasattr(inputs, "lj_lrc") and inputs.lj_lrc == "yes" and env != "vacuum":
    print(f"We will use LJ Long range correction (LRC)")
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            force.setUseDispersionCorrection(True)
        if (
            isinstance(force, CustomNonbondedForce)
            and force.getNumTabulatedFunctions() != 1
        ):
            force.setUseLongRangeCorrection(True)

if env != "vacuum":
    barostat = MonteCarloBarostat(inputs.p_ref * bar, inputs.temp * kelvin)
    system.addForce(barostat)

integrator = LangevinIntegrator(
    inputs.temp * kelvin, 1 / unit.picosecond, inputs.dt * unit.picoseconds
)

# Set platform
platform = Platform.getPlatformByName("CUDA")

# Optional A-form backbone-torsion restraint for the single-strand (waterbox)
# reference state. constrain_waterbox.py writes restraint_torsions.yaml (a list of
# atom-index quads + theta0 [rad] + K [kJ/mol]) into a waterbox state dir to hold the
# strand helical/stacked (the Turner NN reference is helical, not a floppy coil).
# Added to `system` BEFORE the Simulation is built, so it is present during sampling
# AND in the serialized _system.xml that analysis.py re-evaluates -> MBAR-consistent.
# A no-op unless the file is present (so it never affects other systems/users).
if env == "waterbox" and os.path.isfile("restraint_torsions.yaml"):
    import yaml as _yaml

    with open("restraint_torsions.yaml") as _fh:
        _rspec = _yaml.safe_load(_fh)
    _torsions = _rspec.get("torsions", [])
    _rforce = CustomTorsionForce("K*(1-cos(theta-theta0))")
    _rforce.addPerTorsionParameter("K")
    _rforce.addPerTorsionParameter("theta0")
    _rforce.setName("AformBackboneRestraint")
    for _t in _torsions:
        _i, _j, _k, _l = _t["idx"]
        _rforce.addTorsion(_i, _j, _k, _l, [float(_t["K"]), float(_t["theta0"])])
    system.addForce(_rforce)
    print(
        f"Applied A-form torsion restraint: {len(_torsions)} torsions "
        f"(K from restraint_torsions.yaml) to the {env} system"
    )


# Build simulation context
simulation = Simulation(top.topology, system, integrator, platform, prop)
simulation.context.setPositions(crd.positions)
if os.path.isfile(f"lig_in_{env}.irst"):
    with open(f"lig_in_{env}.irst", "r") as f:
        simulation.context.setState(XmlSerializer.deserialize(f.read()))


# Calculate initial system energy
print("\nInitial system energy")
print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

# Energy minimization
if inputs.mini_nstep > 0:
    print("\nEnergy minimization: %s steps" % inputs.mini_nstep)
    simulation.minimizeEnergy(maxIterations=inputs.mini_nstep)
    print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

# Generate initial velocities
if inputs.gen_vel == "yes":
    print("\nGenerate initial velocities")
    if inputs.gen_seed:
        simulation.context.setVelocitiesToTemperature(inputs.gen_temp, inputs.gen_seed)
    else:
        simulation.context.setVelocitiesToTemperature(inputs.gen_temp)

# Equilibration (see EQUIL_* above)
n_restr = int(round(EQUIL_RESTRAINED_PS / inputs.dt))
n_free = int(round(EQUIL_FREE_PS / inputs.dt))
if n_restr + n_free > 0:
    if inputs.gen_vel != "yes":
        simulation.context.setVelocitiesToTemperature(inputs.temp * kelvin)
    if n_restr > 0:
        ref = simulation.context.getState(getPositions=True).getPositions()
        posres = CustomExternalForce("0.5*k_eq*periodicdistance(x, y, z, x0, y0, z0)^2")
        posres.addGlobalParameter("k_eq", EQUIL_K)
        for p in ("x0", "y0", "z0"):
            posres.addPerParticleParameter(p)
        n_atoms = 0
        for atom in top.topology.atoms():
            if atom.residue.name.upper() in _NON_SOLUTE:
                continue
            if atom.element is not None and atom.element.symbol == "H":
                continue
            posres.addParticle(atom.index, ref[atom.index].value_in_unit(nanometers))
            n_atoms += 1
        posres_index = system.addForce(posres)
        simulation.context.reinitialize(preserveState=True)
        print(f"\nEquilibration: {EQUIL_RESTRAINED_PS} ps with {n_atoms} solute heavy atoms restrained")
        simulation.step(n_restr)
        system.removeForce(posres_index)
        simulation.context.reinitialize(preserveState=True)
    if n_free > 0:
        print(f"Equilibration: {EQUIL_FREE_PS} ps unrestrained")
        simulation.step(n_free)
    print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
    simulation.currentStep = 0

# Production
print("\nMD run: %s steps" % inputs.nstep)
simulation.reporters.append(DCDReporter(args.odcd, inputs.nstdcd))
simulation.reporters.append(
    StateDataReporter(
        sys.stdout,
        inputs.nstout,
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
        progress=True,
        remainingTime=True,
        speed=True,
        totalSteps=inputs.nstep,
        separator="\t",
    )
)

simulation.step(inputs.nstep)

# needed for later analysis
file_name = f"lig_in_{env}"
state = simulation.context.getState(getPositions=True, getVelocities=True)
with open(file_name + ".rst", "w") as f:
    f.write(XmlSerializer.serialize(state))
with open(file_name + "_integrator.xml", "w") as outfile:
    outfile.write(XmlSerializer.serialize(integrator))
with open(file_name + "_system.xml", "w") as outfile:
    outfile.write(XmlSerializer.serialize(system))

# mark this run as complete (written last, after every output file)
with open(DONE_MARKER, "w") as f:
    f.write(f"{inputs.nstep} steps\n")
