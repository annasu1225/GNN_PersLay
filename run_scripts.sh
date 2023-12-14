#!/bin/bash
# This script runs the Python scripts reconstruct.py, extract.py, distances.angles.py in calculate_features for each PDB structure.

# Initialize counter
counter=1
failed_pdb_ids=()

# Directory where the raw PDB files are stored
pdb_dir="PDBs"

# Directory where the output files will be stored
output_dir="data_files"

for pdb_file in $pdb_dir/*.pdb; do
    # Get the PDB ID from the file name
    pdb_id=$(basename $pdb_file .pdb)

    # Skip if the directory for this pdb_id already exists and is not empty
    if [ -d "$output_dir/$pdb_id" ] && [ "$(ls -A $output_dir/$pdb_id)" ]; then
        echo "Directory for pdb_id: $pdb_id already exists and is not empty, skipping..."
        continue
    fi

    echo "Processing pdb_id: $pdb_id, count: $counter"

    # Create a directory for this PDB ID
    mkdir -p $output_dir/$pdb_id

    if
        # Run the scripts and save the output files in the PDB ID directory
        python calculate_features/reconstruct.py $pdb_file $output_dir/$pdb_id/${pdb_id}_rec.pdb
        python calculate_features/extract.py $output_dir/$pdb_id/${pdb_id}_rec.pdb $output_dir/$pdb_id/${pdb_id}_rec_bb.txt
        python calculate_features/distances.angles.py $output_dir/$pdb_id/${pdb_id}_rec_bb.txt $output_dir/$pdb_id/${pdb_id}_bl.txt $output_dir/$pdb_id/${pdb_id}_ba.txt $output_dir/$pdb_id/${pdb_id}_da.txt
    then
        counter=$((counter+1))
    else
        # Record the PDB ID if any of the scripts failed
        echo "Failed to process pdb_id: $pdb_id"
        failed_pdb_ids+=($pdb_id)
    fi
done

# Print the list of failed PDB IDs at the end
if [ ${#failed_pdb_ids[@]} -ne 0 ]; then
    echo "Failed to process the following PDB IDs:"
    for pdb_id in "${failed_pdb_ids[@]}"; do
        echo $pdb_id
    done
else
    echo "All PDB files processed successfully."
fi