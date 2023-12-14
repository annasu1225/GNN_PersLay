'''
### This script takes in a reconstructed PDB backbone file and computes the persistence diagrams, vectors and landscape.
## You need to install the following packages:
pip install ripser
pip install gudhi 
'''

import numpy as np
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import gudhi.representations as gdr
import gudhi.tensorflow.perslay as gdtf
import os
import argparse

def get_persistence_diagrams(input_file):
    # Load the backbone atoms coordinates from the reconstructed backbone file
    if os.path.exists(input_file):
        data = np.loadtxt(input_file, skiprows=1, usecols=(1, 2, 3))
    else:
        raise FileNotFoundError(f"The file {input_file} does not exist")

    # Extract the id from the input_file
    id = os.path.basename(input_file).replace('_rec_bb.txt', '')

    # Compute persistent homology for the PDB structure
    diagrams = ripser(data, maxdim=1)['dgms']

    # Plot the persistence diagrams
    plt.figure()
    plot_diagrams(diagrams, show=False)
    plt.title(f'Persistence Diagrams for {id}')

    return diagrams, plt

def get_perslayer(diagrams):

    # Filter out infinite points
    diagrams = [np.array(diagram)[~np.isinf(np.array(diagram)[:, 1])] for diagram in diagrams[0]]
    # Rescale the coordinates of the points in the persistence diagrams in the unit square
    diagrams = gdr.DiagramScaler(use=True, scalers=[([0,1], MinMaxScaler())]).fit_transform(diagrams)

    # Convert into ragged tensors
    diagrams = tf.concat([
        tf.RaggedTensor.from_tensor(tf.constant(diagrams[0][None,:], dtype=tf.float32)),
        tf.RaggedTensor.from_tensor(tf.constant(diagrams[1][None,:], dtype=tf.float32))
    ], axis=0)

    with tf.GradientTape() as tape:
        rho = tf.identity 
        phi = gdtf.TentPerslayPhi(np.array(np.arange(-1.,2.,.001), dtype=np.float32))
        weight = gdtf.PowerPerslayWeight(1.,0.)
        perm_op = 'top3'
        
        perslay = gdtf.Perslay(phi=phi, weight=weight, perm_op=perm_op, rho=rho)
        vectors = perslay(diagrams)
    
    print('Gradient is ', tape.gradient(vectors, phi.samples))

    return vectors

def plot_pers_landscape(vectors, id):

    plt.figure()
    # vectors = np.reshape(vectors[0,:], [-1, 3])
    print(vectors.shape)
    vectors = np.reshape(vectors[0,:], [-1, 3])

    for k in range(2):
        plt.plot(vectors[:,k], linewidth=5., label='landscape ' + str(k))
    plt.legend()
    plt.title('Persistence landscape for ' + id)
    plt.show()
    return plt


# def main():
#     parser = argparse.ArgumentParser(description='Apply persistent homology to protein backbones.')
#     parser.add_argument('input_file', help='Protein backbone file')
#     parser.add_argument('output_file', help='PersLayer vectors file')
#     parser.add_argument('--diagrams_png', help='Persistence diagrams PNG file')
#     parser.add_argument('--landscape_png', help='Persistence landscape PNG file')

#     args = parser.parse_args()

#     # Call get_persistence_diagrams
#     diagrams = get_persistence_diagrams(args.input_file)
#     if args.diagrams_png:
#         plt.savefig(args.diagrams_png)
#     plt.close()

#     # Call get_perslayer
#     vectors = get_perslayer(diagrams)

#     # Call plot_pers_landscape
#     plot_pers_landscape(vectors, os.path.basename(args.input_file).replace('_rec_bb.txt', ''))
#     if args.landscape_png:
#         plt.savefig(args.landscape_png)
#     plt.close()
    
#     if args.output_file:
#         np.save(args.output_file, vectors)

# if __name__ == '__main__':
#     main()

'''
# Example usage:
# Input PDB ID
# Output: H0 and H1 Persistence diagrams, vectors, persistence landscape
'''
# input_file = '/Users/annasu/Documents/GitHub/profun/data_files/5oom/5oom_rec_bb.txt' 
# diagrams = get_persistence_diagrams(input_file)
# vectors = get_perslayer(diagrams)
# print(vectors)
vectors = np.load('/Users/annasu/Documents/GitHub/profun/ph_files/6y74/6y74_ph_vec.npy')
plot_pers_landscape(vectors, '6y74')