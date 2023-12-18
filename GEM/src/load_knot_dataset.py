import os
from os.path import join, exists
import pandas as pd
import numpy as np

import json

from pahelix.datasets.inmemory_dataset import InMemoryDataset


__all__ = ['get_default_knot_task_names', 'load_knot_dataset']


def get_default_knot_task_names():
    """Get that default knot task names."""
    return ['Class']


def get_knot_dataset(data_path, task_names=None):

    input_df = pd.read_json(data_path, orient='index')

    paths = input_df['pdb_path']
    labels = input_df['labels']

    # convert 0 to -1
    labels = labels.replace(0, -1)

    data_list = []
    for i in range(len(paths)):
        data = {}
        data['pdb_path'] = paths[i]        
        data['label'] = labels.values[i]
        data_list.append(data)
    
    dataset = InMemoryDataset(data_list)
    return dataset

# if __name__ == '__main__':
#     print(get_knot_dataset('/gpfs/gibbs/pi/gerstein/as4272/KnotFun/datasets_for_geo/knotprot.json'))