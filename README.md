# CG-to-SAXS

This code can read LAMMPS dump trajectories of coarse grained simulations. It then generate SAXS intensity profiles using bead_weights dictionary, which is the count of electrons in each beads. The code rely on FFT which can use significant amount of RAM depending on the system size.