import argparse
from pdbfixer import PDBFixer
from openmm.app import PDBFile

def reconstruct_pdb(input_file, output_file):
    fixer = PDBFixer(filename=input_file)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    # fixer.addMissingHydrogens(7.0)
    # fixer.addSolvent(fixer.topology.getUnitCellDimensions())
    
    with open(output_file, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reconstruct PDB files.')
    parser.add_argument("input_file", help="Input PDB file")
    parser.add_argument("output_file", help="Output the reconstructed PDB file")
    args = parser.parse_args()
    reconstruct_pdb(args.input_file, args.output_file)