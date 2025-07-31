#!/usr/bin/env python3
"""
OpenMM Molecular Dynamics Simulation Script
Downloads PDB, prepares with PDBFixer, minimizes, solvates, and runs 10 ns MD
"""

import os
import urllib.request
from openmm.app import *
from openmm import *
from openmm.unit import *
from pdbfixer import PDBFixer
import numpy as np
import random

def download_pdb(pdb_id, filename):
    """Download PDB file from RCSB PDB"""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"Downloading {pdb_id} from RCSB PDB...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

def prepare_protein(input_pdb, output_pdb):
    """Prepare protein using PDBFixer"""
    print("Preparing protein with PDBFixer...")
    
    fixer = PDBFixer(filename=input_pdb)
    
    # Find and add missing residues
    fixer.findMissingResidues()
    
    # Find and add missing atoms
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    # Add missing hydrogens
    fixer.addMissingHydrogens(7.0)  # pH 7.0
    
    # Remove heterogens (keep water if desired)
    fixer.removeHeterogens(keepWater=True)
    
    # Write prepared structure
    PDBFile.writeFile(fixer.topology, fixer.positions, open(output_pdb, 'w'))
    print(f"Prepared structure saved as {output_pdb}")
    
    return fixer.topology, fixer.positions

def setup_system(topology, positions):
    """Setup force field and system"""
    print("Setting up force field and system...")
    
    # Use AMBER14 force field with TIP3P water
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    
    # Add solvent (water box with 1.0 nm padding)
    modeller = Modeller(topology, positions)
    modeller.addSolvent(forcefield, model='tip3p', padding=1.0*nanometer)
    
    print(f"System contains {modeller.topology.getNumAtoms()} atoms after solvation")
    
    # Create system
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0*nanometer,
        constraints=HBonds
    )
    
    return system, modeller.topology, modeller.positions

def minimize_energy(system, topology, positions, platform_name='CPU'):
    """Energy minimization"""
    print("Performing energy minimization...")
    
    # Create integrator and simulation
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
    
    try:
        platform = Platform.getPlatformByName(platform_name)
        simulation = Simulation(topology, system, integrator, platform)
    except:
        print(f"Platform {platform_name} not available, using Reference platform")
        platform = Platform.getPlatformByName('Reference')
        simulation = Simulation(topology, system, integrator, platform)
    
    simulation.context.setPositions(positions)
    
    # Minimize energy
    print("Minimizing energy...")
    simulation.minimizeEnergy()
    
    # Get minimized positions
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    minimized_positions = state.getPositions()
    potential_energy = state.getPotentialEnergy()
    
    print(f"Potential energy after minimization: {potential_energy}")
    
    return simulation, minimized_positions

def gcmc_water_equilibration(simulation, topology, positions, gcmc_steps=1000, chemical_potential=-55.0):
    """GCMC water equilibration to optimize water density"""
    print("Starting GCMC water equilibration...")
    
    # Get water molecules
    water_molecules = []
    for residue in topology.residues():
        if residue.name == 'HOH':
            atoms = list(residue.atoms())
            if len(atoms) == 3:  # O, H, H
                water_molecules.append([atom.index for atom in atoms])
    
    print(f"Found {len(water_molecules)} water molecules for GCMC")
    
    # GCMC parameters
    kT = BOLTZMANN_CONSTANT_kB * 300 * kelvin  # Temperature
    volume = simulation.context.getState().getPeriodicBoxVectors()
    box_volume = volume[0][0] * volume[1][1] * volume[2][2]
    
    accepted_insertions = 0
    accepted_deletions = 0
    
    for step in range(gcmc_steps):
        if step % 100 == 0:
            print(f"GCMC step {step}/{gcmc_steps}")
        
        # Choose insertion or deletion randomly
        if random.random() < 0.5 and len(water_molecules) > 0:
            # Attempt deletion
            water_idx = random.randint(0, len(water_molecules) - 1)
            water_atoms = water_molecules[water_idx]
            
            # Get energy before deletion
            state_before = simulation.context.getState(getEnergy=True)
            energy_before = state_before.getPotentialEnergy()
            
            # Store original positions
            original_positions = simulation.context.getState(getPositions=True).getPositions()
            
            # Move water far away (pseudo-deletion)
            new_positions = list(original_positions)
            for atom_idx in water_atoms:
                new_positions[atom_idx] = Vec3(1000, 1000, 1000) * nanometer
            
            simulation.context.setPositions(new_positions)
            
            # Get energy after deletion
            state_after = simulation.context.getState(getEnergy=True)
            energy_after = state_after.getPotentialEnergy()
            
            # Calculate acceptance probability for deletion
            delta_E = energy_after - energy_before
            N = len(water_molecules)
            acceptance_prob = min(1.0, (box_volume / (N * 1.66e-3)) * np.exp((-delta_E + chemical_potential * kilojoule_per_mole) / kT))
            
            if random.random() < acceptance_prob:
                # Accept deletion
                water_molecules.pop(water_idx)
                accepted_deletions += 1
            else:
                # Reject deletion - restore positions
                simulation.context.setPositions(original_positions)
        
        else:
            # Attempt insertion
            # Generate random position in box
            box_vectors = simulation.context.getState().getPeriodicBoxVectors()
            rand_pos = Vec3(
                random.uniform(0, box_vectors[0][0]._value),
                random.uniform(0, box_vectors[1][1]._value), 
                random.uniform(0, box_vectors[2][2]._value)
            ) * nanometer
            
            # Get current positions
            current_positions = list(simulation.context.getState(getPositions=True).getPositions())
            
            # Find unused atom indices (those moved far away)
            unused_indices = []
            for i, pos in enumerate(current_positions):
                if pos[0]._value > 999:  # Far away position
                    unused_indices.append(i)
                    if len(unused_indices) >= 3:  # Need 3 atoms for water
                        break
            
            if len(unused_indices) >= 3:
                # Get energy before insertion
                state_before = simulation.context.getState(getEnergy=True)
                energy_before = state_before.getPotentialEnergy()
                
                # Insert water molecule
                water_atoms = unused_indices[:3]
                current_positions[water_atoms[0]] = rand_pos  # O
                current_positions[water_atoms[1]] = rand_pos + Vec3(0.1, 0, 0) * nanometer  # H
                current_positions[water_atoms[2]] = rand_pos + Vec3(-0.033, 0.094, 0) * nanometer  # H
                
                simulation.context.setPositions(current_positions)
                
                # Get energy after insertion
                state_after = simulation.context.getState(getEnergy=True)
                energy_after = state_after.getPotentialEnergy()
                
                # Calculate acceptance probability for insertion
                delta_E = energy_after - energy_before
                N = len(water_molecules)
                acceptance_prob = min(1.0, ((N + 1) * 1.66e-3 / box_volume) * np.exp((-delta_E - chemical_potential * kilojoule_per_mole) / kT))
                
                if random.random() < acceptance_prob:
                    # Accept insertion
                    water_molecules.append(water_atoms)
                    accepted_insertions += 1
                else:
                    # Reject insertion - move atoms back
                    for atom_idx in water_atoms:
                        current_positions[atom_idx] = Vec3(1000, 1000, 1000) * nanometer
                    simulation.context.setPositions(current_positions)
    
    print(f"GCMC completed: {accepted_insertions} insertions, {accepted_deletions} deletions accepted")
    print(f"Final number of water molecules: {len(water_molecules)}")
    
    return simulation.context.getState(getPositions=True).getPositions()

def run_md_simulation(simulation, steps, output_interval=1000):
    """Run molecular dynamics simulation"""
    print(f"Running {steps} steps of MD simulation...")
    
    # Add reporters
    simulation.reporters.append(PDBReporter('trajectory.pdb', output_interval))
    simulation.reporters.append(StateDataReporter(
        'simulation.log', output_interval,
        step=True, time=True, potentialEnergy=True, kineticEnergy=True,
        totalEnergy=True, temperature=True, volume=True, density=True,
        speed=True
    ))
    
    # Run simulation
    simulation.step(steps)
    print("Simulation completed!")

def main():
    # Configuration
    pdb_id = "1BNA"  # B-DNA dodecamer - change this to your desired PDB ID
    simulation_time = 10  # nanoseconds
    timestep = 0.004  # picoseconds
    platform_name = 'CUDA'  # Try CUDA first, fallback to CPU if not available
    
    # Calculate number of steps
    steps = int(simulation_time * nanoseconds / (timestep * picoseconds))
    print(f"Will run {steps} steps for {simulation_time} ns simulation")
    
    # File names
    raw_pdb = f"{pdb_id}_raw.pdb"
    prepared_pdb = f"{pdb_id}_prepared.pdb"
    
    try:
        # Step 1: Download PDB
        download_pdb(pdb_id, raw_pdb)
        
        # Step 2: Prepare protein
        topology, positions = prepare_protein(raw_pdb, prepared_pdb)
        
        # Step 3: Setup system with solvation
        system, solvated_topology, solvated_positions = setup_system(topology, positions)
        
        # Step 4: Energy minimization
        simulation, minimized_positions = minimize_energy(
            system, solvated_topology, solvated_positions, platform_name
        )
        
        # Step 5: GCMC water equilibration
        print("\nStarting GCMC water equilibration...")
        equilibrated_positions = gcmc_water_equilibration(
            simulation, solvated_topology, minimized_positions, gcmc_steps=500
        )
        simulation.context.setPositions(equilibrated_positions)
        
        # Additional minimization after GCMC
        print("Final energy minimization after GCMC...")
        simulation.minimizeEnergy()
        
        # Step 6: Run MD simulation
        run_md_simulation(simulation, steps, output_interval=5000)  # Save every 20 ps
        
        print("Molecular dynamics simulation completed successfully!")
        print("Output files:")
        print("- trajectory.pdb: MD trajectory")
        print("- simulation.log: Energy and system data")
        print(f"- {prepared_pdb}: Prepared structure")
        print("- GCMC water equilibration completed before MD")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        return False
    
    finally:
        # Clean up temporary files
        if os.path.exists(raw_pdb):
            os.remove(raw_pdb)
    
    return True

if __name__ == "__main__":
    main()