import math
import os
import argparse

class Atom:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def read_coordinates(file_path):
    atoms = []
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line (first line
        for line in file:
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            atoms.append(Atom(x, y, z))
    return atoms

def vector_between_points(p1, p2):
    return (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)

def norm(v):
    return math.sqrt(dot_product(v, v))

def dot_product(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))

def cross_product(v1, v2):
    return (v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0])

def dist(atoms, A, B):
    v = vector_between_points(atoms[A-1], atoms[B-1])
    return norm(v)

def ang(atoms, A, B, C):
    v1 = vector_between_points(atoms[A-1], atoms[B-1])
    v2 = vector_between_points(atoms[C-1], atoms[B-1])
    return math.acos(dot_product(v1, v2) / (norm(v1) * norm(v2))) * 180 / math.pi

def dihe(atoms, A, B, C, D):
    v1 = vector_between_points(atoms[A-1], atoms[B-1])
    v2 = vector_between_points(atoms[B-1], atoms[C-1])
    v3 = vector_between_points(atoms[C-1], atoms[D-1])

    normal1 = cross_product(v1, v2)
    normal2 = cross_product(v2, v3)

    angle = math.acos(dot_product(normal1, normal2) / (norm(normal1) * norm(normal2))) * 180 / math.pi

    if dot_product(cross_product(normal1, normal2), v2) < 0:
        angle = -angle

    return angle

def calculate_distances_angles(input_file, output_lengths, output_angles, output_dihedrals):
    atoms = read_coordinates(input_file)

    with open(output_lengths, 'w') as f_lengths, \
         open(output_angles, 'w') as f_angles, \
         open(output_dihedrals, 'w') as f_dihedrals:
        for i in range(1, len(atoms)):
            if i < len(atoms):
                f_lengths.write(f"Bond length between atoms {i} and {i+1}: {dist(atoms, i, i+1):.4f} Å\n")
            if i < len(atoms) - 1:
                f_angles.write(f"Bond angle defined by atoms {i}, {i+1}, and {i+2}: {ang(atoms, i, i+1, i+2):.2f} degrees\n")
            if i < len(atoms) - 2:
                f_dihedrals.write(f"Dihedral angle defined by atoms {i}, {i+1}, {i+2}, and {i+3}: {dihe(atoms, i, i+1, i+2, i+3):.2f} degrees\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate distances and angles.')
    parser.add_argument("input_file", help="Input file")
    parser.add_argument("output_lengths", help="Output file for bond lengths")
    parser.add_argument("output_angles", help="Output file for bond angles")
    parser.add_argument("output_dihedrals", help="Output file for dihedral angles")
    args = parser.parse_args()
    calculate_distances_angles(args.input_file, args.output_lengths, args.output_angles, args.output_dihedrals)