import openmm as mm
import numpy as np 
import os
import sys
from pdbfixer import PDBFixer
import openmm.app as app
from argparse import ArgumentParser
import openmm.unit as unit
T=280*unit.kelvin
dt=1.0*unit.femtoseconds
fric=1.0/unit.picosecond
import openmm
verlet=openmm.VerletIntegrator(dt)
langevin = openmm.LangevinMiddleIntegrator(T,fric,dt)
hoover=openmm.NoseHooverIntegrator(T,fric,dt)
brownian=openmm.BrownianIntegrator(T,fric,dt)
def get_args():
    parser=ArgumentParser()
    parser.add_argument('-i','--pdbin')
    parser.add_argument('-o','--outdir')
    return parser.parse_args()
def heat_system(simulation, T_start, T_end, n_steps, n_stages=20):
    """Gradually heat a system from T_start to T_end."""
    temps = np.linspace(T_start, T_end, n_stages + 1)
    steps_per_stage = n_steps // n_stages

    print(f"Heating from {T_start} K to {T_end} K in {n_stages} stages:")
    for i, T in enumerate(temps[1:], 1):
        # Set temperature in integrator
        simulation.integrator.setTemperature(T * unit.kelvin)
        simulation.context.setVelocitiesToTemperature(T * unit.kelvin)
        simulation.step(steps_per_stage)

        state = simulation.context.getState(getEnergy=True)
        e = state.getPotentialEnergy().in_units_of(unit.kilojoule_per_mole)
        print(f"  Stage {i:2d}: T = {T:6.1f} K, E_pot = {e}",flush=True)

def run_mm(pdbin,outdir):
    os.makedirs(outdir, exist_ok=True)
    fixer = PDBFixer(filename=pdbin)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.5)
    stem = os.path.splitext(os.path.basename(pdbin))[0]
    fixed_pdb = os.path.join(outdir, f"fixed_{stem}.pdb")
    with open(fixed_pdb, "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
    pdb=app.PDBFile(fixed_pdb)
    ff=app.ForceField('amber19-all.xml','amber19/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff, pH=7.0)
    modeller.addSolvent(ff,
    model="tip3p",
    padding=1.0*unit.nanometer,
    neutralize=True,
    ionicStrength=0.15*unit.molar,
    positiveIon="Na+",
    negativeIon="Cl-")
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0*unit.nanometer,
        rigidWater=True,constraints=app.HBonds,
        ewaldErrorTolerance=0.0005)
    restraint_force = openmm.CustomExternalForce('k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
    restraint_force.addGlobalParameter('k', 1000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    restraint_force.addPerParticleParameter('x0')
    restraint_force.addPerParticleParameter('y0')
    restraint_force.addPerParticleParameter('z0')
    positions=modeller.positions
    for atom in modeller.topology.atoms():
        if atom.residue.name not in {'HOH'}:
            x0 = positions[atom.index].x
            y0 = positions[atom.index].y
            z0 = positions[atom.index].z
            restraint_force.addParticle(atom.index, [x0, y0, z0])
    system.addForce(restraint_force)
    print("backbone restraint",flush=True)
    integrator = openmm.LangevinMiddleIntegrator(
        280*unit.kelvin,
        10.0/unit.picosecond,
        1.0*unit.femtoseconds)
    sim = app.Simulation(modeller.topology, system, integrator)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=200)
    from openmm.app import DCDReporter, StateDataReporter
    from sys import stdout
    sim.reporters.append(app.DCDReporter(os.path.join(outdir, "heating.dcd"), 10))
    sim.reporters.append(StateDataReporter(stdout, 10,step=True,temperature=True,potentialEnergy=True,separator='\t'))
    heat_system(sim, T_start=220, T_end=270, n_steps=2000, n_stages=20)
    state = sim.context.getState(getPositions=True)
    heated_pdb = os.path.join(outdir, f"heat_{stem}.pdb")
    with open(heated_pdb, "w") as f:
        app.PDBFile.writeFile(sim.topology, state.getPositions(), f)
    sim.context.setVelocitiesToTemperature(280*unit.kelvin)
    sim.reporters=[]
    sim.reporters.append(app.DCDReporter(os.path.join(outdir,"equilibrating.dcd"),100))
    sim.reporters.append(app.StateDataReporter(stdout,200,step=True,potentialEnergy=True,kineticEnergy=True,temperature=True,
                                               separator='\t'))
    sim.step(2000)
    sim.reporters=[]
    sim.reporters.append(app.StateDataReporter(os.path.join(outdir,'production.csv'),100,step=True,
                                                            potentialEnergy=True,kineticEnergy=True,
                                                            totalEnergy=True,temperature=True,
                                                            volume=True,density=True,remainingTime=True,speed=True,
                                                            totalSteps=1000,separator=','))
    sim.reporters.append(app.StateDataReporter(stdout,200,step=True,temperature=True,potentialEnergy=True,
                                               separator='\t'))
    prodcheck=os.path.join(outdir,f"prod_{stem}.chk")
    prodpdb=os.path.join(outdir,f"prod_{stem}.pdb")
    proddcd=os.path.join(outdir,f"prod_{stem}.dcd")
    sim.reporters.append(app.CheckpointReporter(prodcheck,500))
    sim.reporters.append(app.PDBReporter(prodpdb,200))
    sim.reporters.append(app.DCDReporter(proddcd, 200))
    sim.step(2000)
if __name__ == "__main__":
    args = get_args()
    run_mm(args.pdbin, args.outdir)
