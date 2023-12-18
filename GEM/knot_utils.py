import numpy as np
from pahelix.datasets.inmemory_dataset import InMemoryDataset

from pahelix.utils.compound_tools import Compound3DKit

def load_atom_positions(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]  # Skip header line
        atom_positions = []
        for line in lines:
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            atom_positions.append([x, y, z])
    return np.array(atom_positions, dtype='float32')

def load_bond_lengths(file_path):
    edge_list = []
    bond_lengths = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        atom_indices = set()  # To keep track of unique atom indices for self-loops

        for line in lines:
            parts = line.split()
            atom1, atom2 = int(parts[4]), int(parts[6].replace(':', ''))  # Assuming indices are in these positions
            atom1 -= 1  # Convert to 0-indexing
            atom2 -= 1

            length = float(parts[7])  # Assuming bond length is in this position

            edge_list.append((atom1, atom2))
            bond_lengths.append(length)

            edge_list.append((atom2, atom1))  # Reverse edge
            bond_lengths.append(length)

            atom_indices.update([atom1, atom2])

        # Add self-loops
        for atom in atom_indices:
            edge_list.append((atom, atom))
            bond_lengths.append(0.0)  # Default value for self-loop bond length

    return edge_list, bond_lengths

def get_superedge_angles(edges, atom_poses, dir_type='HT'):
    """get superedge angles"""
    def _get_vec(atom_poses, edge):
        return atom_poses[edge[1]] - atom_poses[edge[0]]
    def _get_angle(vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
        vec2 = vec2 / (norm2 + 1e-5)
        angle = np.arccos(np.dot(vec1, vec2))
        return angle

    E = len(edges)
    edge_indices = np.arange(E)
    super_edges = []
    bond_angles = []
    bond_angle_dirs = []
    for tar_edge_i in range(E):
        tar_edge = edges[tar_edge_i]
        if dir_type == 'HT':
            src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
        elif dir_type == 'HH':
            src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
        else:
            raise ValueError(dir_type)
        for src_edge_i in src_edge_indices:
            if src_edge_i == tar_edge_i:
                continue
            src_edge = edges[src_edge_i]
            src_vec = _get_vec(atom_poses, src_edge)
            tar_vec = _get_vec(atom_poses, tar_edge)
            super_edges.append([src_edge_i, tar_edge_i])
            angle = _get_angle(src_vec, tar_vec)
            bond_angles.append(angle)
            bond_angle_dirs.append(src_edge[1] == tar_edge[0])  # H -> H or H -> T

    if len(super_edges) == 0:
        super_edges = np.zeros([0, 2], 'int64')
        bond_angles = np.zeros([0,], 'float32')
    else:
        super_edges = np.array(super_edges, 'int64')
        bond_angles = np.array(bond_angles, 'float32')
    return super_edges, bond_angles, bond_angle_dirs


# def get_knot_data(data):



# def get_knot_data(pos_path, bl_path, ba_path, edges):
#     atoms = load_atom_positions(pos_path)
#     edge_list, bond_lengths = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos'])

#     super_edges, bond_angles, bond_angle_dirs = get_superedge_angles(edges, atoms)

#     data = {}
#     data['edges'] = edges
#     data['atom_pos'] = atoms
#     data['bond_length'] = np.array(bond_lengths, dtype='float32')
#     data['BondAngleGraph_edges'] = super_edges
#     data['bond_angle'] = bond_angles

#     return data

def get_knot_data(pos_path, bl_path, ba_path):
    atoms = load_atom_positions(pos_path)
    edge_list, bond_lengths = load_bond_lengths(bl_path)
    edges = np.array(edge_list, dtype=np.int64)

    super_edges, bond_angles, bond_angle_dirs = get_superedge_angles(edges, atoms)

    data = {}
    data['edges'] = edges
    data['atom_pos'] = atoms
    data['bond_length'] = np.array(bond_lengths, dtype='float32')
    data['BondAngleGraph_edges'] = super_edges
    data['bond_angle'] = bond_angles

    return data