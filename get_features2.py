import os
import csv
import numpy as np
import mdtraj as md
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--pdbin", required=True)
    parser.add_argument("--dcd", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pathochain", required=True)
    parser.add_argument("--plantchain", required=True)
    return parser.parse_args()


def calc_potential_e(system, positions):
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    platform = mm.Platform.getPlatformByName("CPU")
    context = mm.Context(system, integrator, platform)
    context.setPositions(positions)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    del context, integrator
    return energy


class ContactFeature:
    def __init__(self, pdbin, dcd, outdir, pathochain, plantchain):
        self.pdbin = pdbin
        self.dcd = dcd
        self.outdir = outdir
        self.pathochain = pathochain
        self.plantchain = plantchain
        os.makedirs(self.outdir, exist_ok=True)

    def GBSA(self, initframe=0, endframe=None, stride=1):
        traj = md.load(self.dcd, top=self.pdbin)

        if endframe is None:
            endframe = len(traj)

        traj = traj[initframe:endframe:stride]

        pathoidx = traj.topology.select(self.pathochain)
        plantidx = traj.topology.select(self.plantchain)
        bothidx = np.concatenate([pathoidx, plantidx])

        energy_split = {
            "Combo": traj.atom_slice(bothidx),
            "Plant": traj.atom_slice(plantidx),
            "Patho": traj.atom_slice(pathoidx),
        }

        results = {}

        for entity, subtraj in energy_split.items():
            ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
            top2mm = subtraj.topology.to_openmm()
            syst = ff.createSystem(
                top2mm,
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
            )

            energies = []
            for i in range(subtraj.n_frames):
                print(f"frame{i}",flush=True)
                positions = subtraj.openmm_positions(i)
                e = calc_potential_e(syst, positions)
                energies.append(e)

            results[entity] = {
                "mean": float(np.mean(energies)),
                "per_frame": np.array(energies, dtype=float),
            }

        ebind_per_frame = (
            results["Combo"]["per_frame"]
            - results["Plant"]["per_frame"]
            - results["Patho"]["per_frame"]
        )
        ebind_mean = float(np.mean(ebind_per_frame))

        return {
            "ebind_mean": ebind_mean,
            "ebind_per_frame": ebind_per_frame,
            "components": results,
        }


if __name__ == "__main__":
    args = get_args()

    confeats = ContactFeature(
        pdbin=args.pdbin,
        dcd=args.dcd,
        outdir=args.outdir,
        pathochain=args.pathochain,
        plantchain=args.plantchain,
    )

    res = confeats.GBSA()

    outfile = os.path.join(args.outdir, "gbsa.csv")
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "E_AB_kcalmol",
            "E_A_kcalmol",
            "E_B_kcalmol",
            "E_bind_kcalmol",
        ])

        E_AB = res["components"]["Combo"]["per_frame"]
        E_A = res["components"]["Plant"]["per_frame"]
        E_B = res["components"]["Patho"]["per_frame"]
        E_bind = res["ebind_per_frame"]

        for i in range(len(E_bind)):
            writer.writerow([i, E_AB[i], E_A[i], E_B[i], E_bind[i]])

    print(f"saved {outfile}")
    print(f"mean E_bind = {res['ebind_mean']:.4f} kcal/mol")
