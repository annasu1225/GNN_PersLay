import json
from rdkit.Chem import AllChem

count = 0
with open('/gpfs/gibbs/pi/gerstein/as4272/KnotFun/datasets_for_geo/knotprot.json', 'r') as f:
    oracle = json.load(f)
    
    data_list = []

    for id_, item in oracle.items():
        print(count)

        mol = AllChem.MolFromPDBFile(item['pdb_path'], sanitize=True)
        if mol is None:
            print(f"Skipping {id_}")
            continue
        else:
            data_list.append(id_)
    
        count += 1

print(len(data_list))

with open('knotprot_ids.csv', 'w') as f:
    f.write('\n'.join(data_list))