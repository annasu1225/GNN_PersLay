# This bash script implements the ph_functions.py script for each PDB structure.

# Directory containing the input files
input_dir="data_files"

# Directory to save the output files
output_dir="ph_files"

# For each directory in the input directory
for dir in $(ls -d ${input_dir}/*/)
do
    # Extract the id from the directory name
    id=$(basename ${dir})

    # Create the output directory if it doesn't exist
    mkdir -p "${output_dir}/${id}"

    # Construct the paths to the input and output files
    input_file="${dir}${id}_rec_bb.txt"
    output_file="${output_dir}/${id}/${id}_ph_vec.npy"

    # Call the Python script with the constructed file paths
    python ph_functions.py ${input_file} ${output_file} 
done