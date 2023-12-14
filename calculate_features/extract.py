import argparse

def extract_main_chain_coordinates(pdb_file_path, output_file_path):
    main_chain_atoms = ['N', 'CA', 'C']  # Main chain atoms in amino acids

    with open(pdb_file_path, 'r') as file, open(output_file_path, 'w') as output_file:
        # Write the header to the output file
        output_file.write('ATOM\tX\tY\tZ\tRESIDUE\n')

        for line in file:
            if line.startswith('ATOM'):  # Identify lines that contain atom information
                atom_name = line[12:16].strip()  # Extract atom name
                if atom_name in main_chain_atoms:
                    # Extracting X, Y, Z coordinates
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    # Extract residue number
                    res_num = int(line[22:26].strip())
                    # Write the coordinates and residue number to the output file
                    output_file.write(f'{atom_name}\t{x}\t{y}\t{z}\t{res_num}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract backbone from PDB files.')
    parser.add_argument('input_file', type=str, help='Input PDB file')
    parser.add_argument('output_file', type=str, help='Output file for the backbone structure')
    args = parser.parse_args()
    extract_main_chain_coordinates(args.input_file, args.output_file)