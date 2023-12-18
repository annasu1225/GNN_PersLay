#!/usr/bin/python                                                                                                                                  
#-*-coding:utf-8-*- 
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Finetune:to do some downstream task
"""

import os
from os.path import join, exists, basename
import argparse
import numpy as np

import paddle
import paddle.nn as nn
import pgl

from pahelix.utils.compound_tools import mol_to_graph_data, Compound3DKit
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', "0D Tensor cannot be used as 'Tensor.numpy()[0]'")

import json

from rdkit import Chem

from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config
from pahelix.datasets.inmemory_dataset import InMemoryDataset

from rdkit.Chem import AllChem
from pahelix.utils.compound_tools import mol_to_graph_data

from knot_utils import get_knot_data

from src.model import DownstreamModel
from src.featurizer import DownstreamTransformFn, DownstreamTransformFnPDB, DownstreamCollateFn
from src.utils import get_dataset, create_splitter, get_downstream_task_names, \
        calc_rocauc_score, exempt_parameters, calc_rocauc_score_multi, custom_multi_label_roc_auc
from src.load_knot_dataset import get_knot_dataset    


def train(args, model, train_dataset, collate_fn, criterion, encoder_opt, head_opt):
    """
    Define the train function 
    Args:
        args,model,train_dataset,collate_fn,criterion,encoder_opt,head_opt;
    Returns:
        the average of the list loss
    """
    data_gen = train_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True,
            collate_fn=collate_fn)
    list_loss = []
    model.train()
    for atom_bond_graphs, bond_angle_graphs, valids, labels in data_gen:
        if len(labels) < args.batch_size * 0.5:
            continue
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        valids = paddle.to_tensor(valids, 'float32')
        preds = model(atom_bond_graphs, bond_angle_graphs)
        loss = criterion(preds, labels)
        loss = paddle.sum(loss * valids) / paddle.sum(valids)
        loss.backward()
        encoder_opt.step()
        head_opt.step()
        encoder_opt.clear_grad()
        head_opt.clear_grad()
        list_loss.append(loss.numpy())
    return np.mean(list_loss)

import numpy as np
import paddle

def evaluate(args, model, test_dataset, collate_fn):
    data_gen = test_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=False,
            collate_fn=collate_fn)

    total_correct = 0
    total_labels = 0

    # Initialize arrays to track correct predictions and total predictions per label
    correct_per_label = None
    total_per_label = None

    model.eval()
    for atom_bond_graphs, bond_angle_graphs, valids, labels in data_gen:
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        valids = paddle.to_tensor(valids, 'float32')
        preds = model(atom_bond_graphs, bond_angle_graphs)

        predicted = preds.numpy() > 0.5  # or any threshold you want
        true = labels.numpy()

        # Initialize arrays if they haven't been initialized yet
        if correct_per_label is None:
            correct_per_label = np.zeros(predicted.shape[1])
            total_per_label = np.zeros(predicted.shape[1])

        # Update correct and total counts
        total_correct += np.sum(predicted == true)
        total_labels += predicted.shape[0] * predicted.shape[1]

        # Update per-label counts
        correct_per_label += np.sum(predicted == true, axis=0)
        total_per_label += predicted.shape[0]

    # Calculate overall accuracy
    accuracy = total_correct / total_labels

    # Calculate per-label accuracy
    per_label_accuracy = correct_per_label / total_per_label

    return accuracy, per_label_accuracy





# def evaluate(args, model, test_dataset, collate_fn):
#     """
#     Define the evaluate function
#     In the dataset, a proportion of labels are blank. So we use a `valid` tensor 
#     to help eliminate these blank labels in both training and evaluation phase.
#     """
#     data_gen = test_dataset.get_data_loader(
#             batch_size=args.batch_size, 
#             num_workers=args.num_workers, 
#             shuffle=False,
#             collate_fn=collate_fn)
#     total_pred = []
#     total_label = []
#     total_valid = []
#     model.eval()
#     for atom_bond_graphs, bond_angle_graphs, valids, labels in data_gen:
#         atom_bond_graphs = atom_bond_graphs.tensor()
#         bond_angle_graphs = bond_angle_graphs.tensor()
#         labels = paddle.to_tensor(labels, 'float32')
#         valids = paddle.to_tensor(valids, 'float32')
#         preds = model(atom_bond_graphs, bond_angle_graphs)
#         total_pred.append(preds.numpy())
#         total_valid.append(valids.numpy())
#         total_label.append(labels.numpy())
#     total_pred = np.concatenate(total_pred, 0)
#     total_label = np.concatenate(total_label, 0)
#     total_valid = np.concatenate(total_valid, 0)
#     #print('pred', total_pred, 'label', total_label, 'valid', total_valid)
#     #return multi_label_roc_auc(total_label, total_pred, total_valid)
#     #return calc_rocauc_score(total_label, total_pred, total_valid)
#     return custom_multi_label_roc_auc(total_label, total_pred)


def get_pos_neg_ratio(dataset):
    """tbd"""
    labels = np.array([data['label'] for data in dataset])
    return np.mean(labels == 1), np.mean(labels == 0)


def main(args):
    """
    Call the configuration function of the model, build the model and load data, then start training.
    model_config:
        a json file  with the hyperparameters,such as dropout rate ,learning rate,num tasks and so on;
    num_tasks:
        it means the number of task that each dataset contains, it's related to the dataset;
    """
    
    compound_encoder_config = load_json_config(args.compound_encoder_config)
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate

    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        model_config['dropout_rate'] = args.dropout_rate
    task_names = ['Class']
    model_config['task_type'] = 'class'
    model_config['num_tasks'] = 34 # number of molecular function classes

    ### build model
    compound_encoder = GeoGNNModel(compound_encoder_config)
    model = DownstreamModel(model_config, compound_encoder)
    criterion = nn.CrossEntropyLoss(reduction='none', use_softmax=False, soft_label=True)
    encoder_params = compound_encoder.parameters()
    head_params = exempt_parameters(model.parameters(), encoder_params)
    encoder_opt = paddle.optimizer.Adam(args.encoder_lr, parameters=encoder_params)
    head_opt = paddle.optimizer.Adam(args.head_lr, parameters=head_params)
    print('Total param num: %s' % (len(model.parameters())))
    print('Encoder param num: %s' % (len(encoder_params)))
    print('Head param num: %s' % (len(head_params)))
    for i, param in enumerate(model.named_parameters()):
        print(i, param[0], param[1].name)

    if not args.init_model is None and not args.init_model == "":
        compound_encoder.set_state_dict(paddle.load(args.init_model))
        print('Load state_dict from %s' % args.init_model)

    ### load data
    # featurizer:
    #     Gen features according to the raw data and return the graph data.
    #     Collate features about the graph data and return the feed dictionary.
    # splitter:
    #     split type of the dataset:random,scaffold,random with scaffold. Here is randomsplit.
    #     `ScaffoldSplitter` will firstly order the compounds according to Bemis-Murcko scaffold, 
    #     then take the first `frac_train` proportion as the train set, the next `frac_valid` proportion as the valid set 
    #     and the rest as the test set. `ScaffoldSplitter` can better evaluate the generalization ability of the model on 
    #     out-of-distribution samples. Note that other splitters like `RandomSplitter`, `RandomScaffoldSplitter` 
    #     and `IndexSplitter` is also available."
    
    print("i exist p2")
    
    if args.task == 'data':
        with open(args.data_path, 'r') as f:
            oracle = json.load(f)
        
        data_list = []

        count = 0
        for id_, item in oracle.items():
            count+=1
            # if count >= 50:
            #     break
            
            print(count)

            mol = AllChem.MolFromPDBFile(item['pdb_path'], sanitize=True)
            if mol is None:
                print(f"Skipping {id_}")
                continue

            atoms = Compound3DKit.get_atom_poses(mol, mol.GetConformer(0))

            data = mol_to_graph_data(mol)

            data['atom_pos'] = np.array(atoms, 'float32')
            data['bond_length'] = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos'])
            BondAngleGraph_edges, bond_angles, bond_angle_dirs = Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'])
            data['BondAngleGraph_edges'] = BondAngleGraph_edges
            data['bond_angle'] = np.array(bond_angles, 'float32')
            data['label'] = np.array(item['labels'])
            


            # pos_path = item['atoms_path']
            # bl_path = item['bond_length_path']
            # ba_path = item['bond_angle_path']
            # data = get_knot_data(pos_path, bl_path, ba_path)

            # mol = AllChem.MolFromPDBFile(item['pdb_path'])
            # if mol is None:
            #     print(f"Skipping {id_}")
            #     continue
            # mol_data = mol_to_graph_data(mol)

            # del mol_data['edges']

            # data.update(mol_data)

            # print('Edges shape', data['edges'].shape, 'Bond length shape', data['bond_length'].shape, 'Bond angle shape', data['bond_angle'].shape)

            # data['label'] = np.array(item['labels'])

            # data['smiles'] = Chem.MolToSmiles(mol)

            data_list.append(data)

        dataset = InMemoryDataset(data_list=data_list)
        dataset.save_data(args.cached_data_path)
        return

    if args.cached_data_path is None or args.cached_data_path == "":
        print("Pleace preprocess first using task=data.")
        return
    else:
        print('Read preprocessed data...')
        dataset = InMemoryDataset(npz_data_path=args.cached_data_path)
        # This was a temp fix so I didn't have to reprocess the dataset
        # for elem in dataset.data_list:
        #     elem['label'] = elem['labels']
        #     del elem['labels']

    print(f'My datalist length is {len(dataset.data_list)}')

    splitter = create_splitter(args.split_type)
    train_dataset, valid_dataset, test_dataset = splitter.split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    print("Train/Valid/Test num: %s/%s/%s" % (
            len(train_dataset), len(valid_dataset), len(test_dataset)))
    print('Train pos/neg ratio %s/%s' % get_pos_neg_ratio(train_dataset))
    print('Valid pos/neg ratio %s/%s' % get_pos_neg_ratio(valid_dataset))
    print('Test pos/neg ratio %s/%s' % get_pos_neg_ratio(test_dataset))

    ### start train
    # Load the train function and calculate the train loss in each epoch.
    # Here we set the epoch is in range of max epoch,you can change it if you want. 

    # Then we will calculate the train loss ,valid auc,test auc and print them.
    # Finally we save it to the model according to the dataset.

    collate_fn = DownstreamCollateFn(
        atom_names=compound_encoder_config['atom_names'], 
        bond_names=compound_encoder_config['bond_names'],
        bond_float_names=compound_encoder_config['bond_float_names'],
        bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
        task_type='class')
    ### Start train
    best_val_accuracy = -1
    best_test_accuracy = -1
    best_epoch = -1

    for epoch_id in range(args.max_epoch):
        train_loss = train(args, model, train_dataset, collate_fn, criterion, encoder_opt, head_opt)

        # Evaluate model on validation set
        val_accuracy, per_label_val_accuracy = evaluate(args, model, valid_dataset, collate_fn)

        # Evaluate on test set and update best model if current model is better
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch_id
            best_test_accuracy, best_per_label_test_accuracy = evaluate(args, model, test_dataset, collate_fn)

        # Logging
        print(f"Epoch: {epoch_id}, Train Loss: {train_loss}, Validation Accuracy: {val_accuracy}, Per Label Validation Accuracy {np.mean(per_label_val_accuracy)}")

        # Save model checkpoints
        paddle.save(compound_encoder.state_dict(), 
                    f'{args.model_dir}/epoch{epoch_id}/compound_encoder.pdparams')
        paddle.save(model.state_dict(), 
                    f'{args.model_dir}/epoch{epoch_id}/model.pdparams')

    # Log best epoch and its test accuracy
    print(f"FINAL Best Epoch: {best_epoch}, Best Validation Accuracy: {best_val_accuracy}, Test Accuracy: {best_test_accuracy}")


    # list_val_auc, list_test_auc = [], []
    # collate_fn = DownstreamCollateFn(
    #         atom_names=compound_encoder_config['atom_names'], 
    #         bond_names=compound_encoder_config['bond_names'],
    #         bond_float_names=compound_encoder_config['bond_float_names'],
    #         bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
    #         task_type='class')
    # for epoch_id in range(args.max_epoch):
    #     train_loss = train(args, model, train_dataset, collate_fn, criterion, encoder_opt, head_opt)
    #     val_auc = evaluate(args, model, valid_dataset, collate_fn)
    #     test_auc = evaluate(args, model, test_dataset, collate_fn)

    #     list_val_auc.append(val_auc)
    #     list_test_auc.append(test_auc)
    #     test_auc_by_eval = list_test_auc[np.argmax(list_val_auc)]
    #     print("epoch:%s train/loss:%s" % (epoch_id, train_loss))
    #     print("epoch:%s val/auc:%s" % (epoch_id, val_auc))
    #     print("epoch:%s test/auc:%s" % (epoch_id, test_auc))
    #     print("epoch:%s test/auc_by_eval:%s" % (epoch_id, test_auc_by_eval))
    #     paddle.save(compound_encoder.state_dict(), 
    #             '%s/epoch%d/compound_encoder.pdparams' % (args.model_dir, epoch_id))
    #     paddle.save(model.state_dict(), 
    #             '%s/epoch%d/model.pdparams' % (args.model_dir, epoch_id))

    # # After training, calculate F1 score
    # total_tp, total_fp, total_fn, total_tn = evaluate(args, model, test_dataset, collate_fn)
    
    # # Calculate F1 score for each class
    # f1_scores = []
    # for i in range(34):
    #     precision = total_tp[i] / (total_tp[i] + total_fp[i] + 1e-8)
    #     recall = total_tp[i] / (total_tp[i] + total_fn[i] + 1e-8)
    #     f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    #     f1_scores.append(f1_score)

    # # Save F1 scores to a file
    # with open('f1_scores.txt', 'w') as file:
    #     for i, score in enumerate(f1_scores):
    #         file.write(f"Class {i}: F1 Score = {score}\n")
    # outs = {
    #     'model_config': basename(args.model_config).replace('.json', ''),
    #     'metric': '',
    #     'dataset': args.dataset_name, 
    #     'split_type': args.split_type, 
    #     'batch_size': args.batch_size,
    #     'dropout_rate': args.dropout_rate,
    #     'encoder_lr': args.encoder_lr,
    #     'head_lr': args.head_lr,
    #     'exp_id': args.exp_id,
    # }
    # offset = 20
    # best_epoch_id = np.argmax(list_val_auc[offset:]) + offset
    # for metric, value in [
    #         ('test_auc', list_test_auc[best_epoch_id]),
    #         ('max_valid_auc', np.max(list_val_auc)),
    #         ('max_test_auc', np.max(list_test_auc))]:
    #     outs['metric'] = metric
    #     print('\t'.join(['FINAL'] + ["%s:%s" % (k, outs[k]) for k in outs] + [str(value)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=['train', 'data'], default='train')

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--dataset_name", 
            choices=['bace', 'bbbp', 'clintox', 'hiv', 
                'muv', 'sider', 'tox21', 'toxcast', 'knot'])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--cached_data_path", type=str, default=None)
    parser.add_argument("--split_type", 
            choices=['random', 'scaffold', 'random_scaffold', 'index'])

    parser.add_argument("--compound_encoder_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--encoder_lr", type=float, default=0.001)
    parser.add_argument("--head_lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--exp_id", type=int, help='used for identification only')
    args = parser.parse_args()
    
    main(args)
