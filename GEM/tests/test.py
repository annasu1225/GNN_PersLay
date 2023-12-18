#jga32@r208u35n01.mccleary
from src.featurizer import DownstreamTransformFnPDB
import json
import time
from rdkit.Chem import AllChem
from knot_utils import get_knot_data
from rdkit import Chem


from pahelix.utils.compound_tools import mol_to_graph_data, Compound3DKit
import numpy as np




with open('/gpfs/gibbs/pi/gerstein/as4272/KnotFun/datasets_for_geo/knotprot.json') as f:
    raw = json.load(f)

sample = raw['6yma']
print(sample)

start_time = time.time()

# pos_path = sample['atoms_path']
# bl_path = sample['bond_length_path']
# ba_path = sample['bond_angle_path']
# data = get_knot_data(pos_path, bl_path, ba_path)

mol = AllChem.MolFromPDBFile(sample['pdb_path'])

atoms = Compound3DKit.get_atom_poses(mol, mol.GetConformer(0))

data = mol_to_graph_data(mol)

data['atom_pos'] = np.array(atoms, 'float32')
data['bond_length'] = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos'])
BondAngleGraph_edges, bond_angles, bond_angle_dirs = \
        Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'])



print(data.keys())
end_time = time.time()
execution_time = end_time - start_time

print("Execution time:", execution_time, "seconds")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


with open('new.json', 'w') as f:
    sorted_out = dict(sorted(data.items()))
    json.dump(sorted_out, f, indent=4, cls=NumpyEncoder)

#Old way
# a = DownstreamTransformFnPDB()
# old = a(sample)

# with open('old.json', 'w') as f:
#     sorted_old = dict(sorted(old.items()))
#     json.dump(sorted_old, f, indent=4, cls=NumpyEncoder)

old = json.load(open('old.json', 'r'))

for key in dict(sorted(data.items())):
    print(key, data[key].shape, np.array(old[key]).shape, data[key].shape==np.array(old[key]).shape)




# # Time was 43.58756399154663 seconds