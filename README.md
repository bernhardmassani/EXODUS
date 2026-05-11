# EXODUS
Automated Sequential Unit Cell Fitting Routine for Large Powder-Diffraction Data Sets.

EXODUS (Equation-of-state X-ray Observation and Diffraction Unit-cell Solver) is an open-source Python package for the rapid sequential refinement of lattice parameters from large time-resolved X-ray diffraction datasets. EXODUS is designed for fast high-throughput analysis, where conventional Rietveld packages become a workflow bottleneck. The package implements a Pawley-style profile refinement seeded by a third-order Birch-Murnaghan equation of state, supports multi-phase systems with phase-dependent frame ranges, and provides a graphical user interface for interactive inspection. We validate EXODUS against GSAS-II, demonstrating agreement on lattice parameters within experimental uncertainties.

Two versions are currently avilable:
  1) Exodus v0.0.1 is a python script intended to create publication-quality figures in matplotlib
  2) Exodus v0.1.3 is the latest GUI version identical to v0.0.1. Data can saved as *.txt files and plotted in a dedicated plotting program (Origin, QTIplot, etc.)
