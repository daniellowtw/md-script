# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains a single Python script for running OpenMM molecular dynamics simulations with GCMC water equilibration. The script downloads PDB structures, prepares them with PDBFixer, solvates them, and runs MD simulations.

## Common Commands

### Installing dependencies and running the simulation
The project uses pixi for dependency management. Install dependencies with:
```bash
pixi install
```

Run the simulation:
```bash
pixi run run
```

Clean up output files:
```bash
pixi run clean
```

## Code Architecture

### Main Components

- **PDB Download & Preparation**: Downloads structures from RCSB PDB and uses PDBFixer to add missing atoms/residues and hydrogens
- **System Setup**: Uses AMBER14 force field with TIP3P water model, adds explicit solvent with 1.0 nm padding
- **Energy Minimization**: Langevin integrator-based minimization with platform fallback (CUDA → Reference)
- **GCMC Water Equilibration**: Custom Grand Canonical Monte Carlo implementation for optimizing water density before MD
- **MD Simulation**: Production molecular dynamics with trajectory and log output

### Key Configuration Points

- Default PDB ID: "1BNA" (B-DNA dodecamer) - changeable in `main()` function at line 241
- Simulation time: 10 ns (configurable at line 242)
- Timestep: 0.004 ps (line 243)
- Platform preference: CUDA with CPU fallback (line 244)
- GCMC steps: 500 (line 272)
- Output frequency: Every 5000 steps / 20 ps (line 281)

### Output Files

- `trajectory.pdb`: MD trajectory frames
- `simulation.log`: Energy, temperature, and system data
- `{PDB_ID}_prepared.pdb`: PDBFixer-processed structure
- Temporary `{PDB_ID}_raw.pdb` files are automatically cleaned up

### Platform Handling

The script implements automatic platform fallback: attempts CUDA first, then falls back to Reference platform if CUDA is unavailable. This is handled in the `minimize_energy()` function at lines 78-84.