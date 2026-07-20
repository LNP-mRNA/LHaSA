#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_data_LANCE.py

COMET 论文数据预处理脚本。
将 JSON 格式的 LNP 数据转换为 LMDB 格式，供 Uni-Mol 模型训练使用。

功能：
  1. 用 RDKit 生成脂质分子的 3D 构象（10个3D + 1个2D）
  2. 按多种策略划分 Train/Valid/Test（随机、Top/Bottom heldout、K-Fold）
  3. 支持训练集子采样、多标签拆分
  4. 输出 mol.lmdb + train.lmdb / valid.lmdb / test.lmdb / infer.lmdb

原文件：preprocess_data_LANCE.ipynb
改写为 Python 脚本，功能和数据路径保持一致。

用法：
  python preprocess_data_LANCE.py --fig fig2a
  python preprocess_data_LANCE.py --fig fig3dii
  python preprocess_data_LANCE.py --fig all
"""

import os
import sys
import argparse
import pickle
import lmdb
import json
import random
import shutil
import copy
import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore')

# ======================== 全局随机种子 ========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ======================== 分子坐标生成 ========================

def smi2_2Dcoords(smi):
    """生成 2D 坐标"""
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    assert len(mol.GetAtoms()) == len(coordinates),         "2D coordinates shape is not align with {}".format(smi)
    return coordinates


def smi2_3Dcoords(smi, cnt):
    """生成 cnt 个 3D 构象"""
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list = []
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    coordinates = smi2_2Dcoords(smi)
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    coordinates = smi2_2Dcoords(smi)
        except:
            coordinates = smi2_2Dcoords(smi)

        assert len(mol.GetAtoms()) == len(coordinates),             "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list


def inner_smi2coords(content, pickle_output=False):
    """处理单个 SMILES，生成 3D+2D 坐标"""
    smi = content
    cnt = 10  # 10 个 3D + 1 个 2D
    mol = Chem.MolFromSmiles(smi)
    if len(mol.GetAtoms()) > 400:
        coordinate_list = [smi2_2Dcoords(smi)] * (cnt + 1)
        print("atom num >400, use 2D coords", smi)
    else:
        coordinate_list = smi2_3Dcoords(smi, cnt)
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

    output = {
        'atoms': atoms,
        'coordinates': coordinate_list,
        'mol': mol,
        'smi': smi
    }

    if pickle_output:
        return pickle.dumps(output, protocol=-1)
    else:
        return output


def smi2coords_onlymol(content):
    """包装函数，带异常处理"""
    try:
        return inner_smi2coords(content, pickle_output=True)
    except:
        print("failed smiles: {}".format(content))
        return None


# ======================== LNP 数据处理 ========================

def inner_lnp2data(smi2mol_id, content, pickle_output=True):
    """将单条 LNP JSON 转换为 LMDB 格式"""
    components_list = content['components']
    if "labels" in content:
        raw_labels = content['labels']
    else:
        raw_labels = {}
    dataset_name = content['dataset_name']
    lnp_id = content['lnp_id']

    # 可选属性（NP_ratio, actual_ilrna_wt_ratio, volumetric_ratio）
    np_props = {}
    if "NP_ratio" in content:
        np_props['NP_ratio'] = content['NP_ratio']
    if "actual_ilrna_wt_ratio" in content:
        np_props['actual_ilrna_wt_ratio'] = content['actual_ilrna_wt_ratio']
    if "volumetric_ratio" in content:
        np_props['volumetric_ratio'] = content['volumetric_ratio']

    labels = raw_labels
    output_components_list = []
    mol_ids = []
    percents = []
    component_types = []

    for component in components_list:
        component_output = copy.deepcopy(component)
        mol_id = smi2mol_id[component['smi']]
        component_output['mol_id'] = mol_id
        output_components_list.append(component_output)
        mol_ids.append(mol_id)
        percents.append(component['percent'])
        component_types.append(component['component_type'])

    output = {
        'mol_id': mol_ids,
        'percent': percents,
        'component_type': component_types,
        'target': labels,
        'dataset_name': dataset_name,
        'components': output_components_list,
        'lnp_id': lnp_id,
        **np_props
    }

    if pickle_output:
        return pickle.dumps(output, protocol=-1)
    else:
        return output


def lnp2data(smi2mol_id, content):
    """包装函数，带异常处理"""
    try:
        return inner_lnp2data(smi2mol_id, content)
    except:
        print("failed lnp: {}".format(content.get('lnp_id', 'unknown')))
        return None


# ======================== 主处理函数 ========================

def write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
    inpath='./',
    outpath='processed_data_dirs/',
    nthreads=16,
    test_ratio=0.1,
    kfold_valid=None,
    valid_ratio=None,
    top_heldout_ratio=0,
    bottom_heldout_ratio=0,
    target_label_name="in_house_lnp_DC24_luc",
    random_train_subsample_ratio=None,
    train_subsample_sample_ids=None,
    subsample_target_label=None,
    split_multilabel_train_sample=False,
    labels_to_split_into_subfolders=None,
    train_lnp_ids=None,
    valid_lnp_ids=None,
    test_lnp_ids=None,
    debug=False,
    shuffle=True,
):
    """
    主函数：读取 JSON，生成 mol.lmdb 和 train/valid/test/infer.lmdb
    """
    # 检查固定划分参数
    if all(v is not None for v in [train_lnp_ids, valid_lnp_ids, test_lnp_ids]):
        fixed_train_valid_test_split = True
    else:
        fixed_train_valid_test_split = False
    if not (all(v is None for v in [train_lnp_ids, valid_lnp_ids, test_lnp_ids]) or fixed_train_valid_test_split):
        raise ValueError("train_lnp_ids, valid_lnp_ids, test_lnp_ids must be all None or all not None")
    print("fixed_train_valid_test_split: ", fixed_train_valid_test_split)

    # Top/Bottom heldout 计算
    top_bottom_ratio = top_heldout_ratio + bottom_heldout_ratio
    if top_bottom_ratio > test_ratio:
        raise ValueError("top_heldout_ratio + bottom_heldout_ratio > test_ratio")
    else:
        total_random_test_ratio = test_ratio - top_bottom_ratio
        if top_bottom_ratio < 1.0:
            remaining_random_test_ratio = total_random_test_ratio / (1 - top_bottom_ratio)
        else:
            remaining_random_test_ratio = 0.0

    # 读取 JSON
    with open(os.path.join(inpath), 'r') as openfile:
        json_obj = json.load(openfile)

    dataset_name_list = []
    dataset_dict = {}
    json_list = []

    for lnp_id in json_obj:
        lnp_dict = json_obj[lnp_id]
        lnp_dict['lnp_id'] = lnp_id

        if 'dataset_name' in lnp_dict:
            lnp_dataset_name = lnp_dict['dataset_name']
            if lnp_dataset_name not in dataset_name_list:
                dataset_name_list.append(lnp_dataset_name)
                dataset_dict[lnp_dataset_name] = []
            dataset_dict[lnp_dataset_name].append(lnp_dict)

        # 将 mol 转换为 percent（代码中原逻辑）
        np_components = lnp_dict['components']
        for c_id, component in enumerate(np_components):
            lnp_dict['components'][c_id]['percent'] = lnp_dict['components'][c_id]['mol']

        json_list.append(lnp_dict)

    sz = len(json_list)
    print("sz: ", sz)

    # ===================== 生成 mol.lmdb =====================
    smi_list = []
    for np_obj in json_list:
        np_components = np_obj['components']
        for component in np_components:
            smi = component['smi']
            if smi not in smi_list:
                smi_list.append(smi)

    mol_filename = "mol.lmdb"
    os.makedirs(outpath, exist_ok=True)
    mol_output_name = os.path.join(outpath, mol_filename)
    try:
        os.remove(mol_output_name)
    except:
        pass

    env_new = lmdb.open(
        mol_output_name,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)
    with Pool(nthreads) as pool:
        i = 0
        for inner_output in tqdm(pool.imap(smi2coords_onlymol, smi_list), total=len(smi_list), desc="mol.lmdb"):
            if inner_output is not None:
                txn_write.put(f'{i}'.encode("ascii"), inner_output)
                i += 1
        print('{} process {} lines'.format(mol_filename, i))
        txn_write.commit()
        env_new.close()

    print("finished processing mol.lmdb")

    # 构建 smi2mol_id 映射
    smi2mol_id = {}
    mol_id2smi = {}
    for mol_id, smi in enumerate(smi_list):
        smi2mol_id[smi] = mol_id
        mol_id2smi[mol_id] = smi

    lnp2data_w_smi2mol_id = partial(lnp2data, smi2mol_id)

    # ===================== 辅助函数 =====================

    def get_train_test_split_with_heldout_topbottom(
        dataset_json_list, target_label_name,
        bottom_heldout_ratio, top_heldout_ratio, remaining_random_test_ratio
    ):
        lnp_label_values = []
        dataset_sz = len(dataset_json_list)
        for lnp_obj in dataset_json_list:
            lnp_label_dict = lnp_obj['labels']
            if len(list(lnp_label_dict.keys())) > 1 and target_label_name not in list(lnp_label_dict.keys()):
                raise ValueError("target_label_name not in label names")
            elif len(list(lnp_label_dict.keys())) == 1:
                target_label_name = list(lnp_label_dict.keys())[0]
            lnp_label_value = lnp_label_dict[target_label_name]
            lnp_label_values.append(lnp_label_value)

        # 按标签值排序
        dataset_json_list = [x for _, x in sorted(zip(lnp_label_values, dataset_json_list), key=lambda y: y[0])]

        dataset_json_list_wo_heldout = dataset_json_list[
            int(dataset_sz * bottom_heldout_ratio):int(dataset_sz * (1 - top_heldout_ratio))
        ]
        top_heldout_set = dataset_json_list[int(dataset_sz * (1 - top_heldout_ratio)):]
        bottom_heldout_set = dataset_json_list[:int(dataset_sz * bottom_heldout_ratio)]

        if remaining_random_test_ratio > 0:
            if shuffle:
                np.random.shuffle(dataset_json_list_wo_heldout)
            wo_heldout_dataset_sz = len(dataset_json_list_wo_heldout)
            random_test = dataset_json_list_wo_heldout[:int(wo_heldout_dataset_sz * remaining_random_test_ratio)]
            train_valid = dataset_json_list_wo_heldout[int(wo_heldout_dataset_sz * remaining_random_test_ratio):]
        else:
            random_test, train_valid = [], dataset_json_list_wo_heldout

        test = random_test + top_heldout_set + bottom_heldout_set
        return train_valid, test

    def subsample_train(train_set, random_train_subsample_ratio=None,
                        train_subsample_sample_ids=None, subsample_target_label=None):
        if train_subsample_sample_ids is not None:
            sampled_indices = []
            for i, train_sample in enumerate(train_set):
                if train_sample["lnp_id"] in train_subsample_sample_ids:
                    sampled_indices.append(i)

            if subsample_target_label is not None:
                new_train_set = []
                for i, train_sample in enumerate(train_set):
                    if i in sampled_indices:
                        new_train_set.append(train_sample)
                    else:
                        train_sample['labels'].pop(subsample_target_label, None)
                        if len(train_sample['labels'].keys()) > 0:
                            new_train_set.append(train_sample)
                train_set = new_train_set

        if random_train_subsample_ratio is not None:
            train_size = len(train_set)
            sampled_indices = random.sample(range(train_size), int(train_size * random_train_subsample_ratio))
            if subsample_target_label is not None:
                new_train_set = []
                for i, train_sample in enumerate(train_set):
                    if i in sampled_indices:
                        new_train_set.append(train_sample)
                    else:
                        train_sample['labels'].pop(subsample_target_label, None)
                        if len(train_sample['labels'].keys()) > 0:
                            new_train_set.append(train_sample)
            else:
                new_train_set = [train_set[i] for i in sampled_indices]
            train_set = new_train_set

        return train_set

    def make_data_lmdb(train, valid, test, infer, dataset_outpath, debug=False):
        """生成 train/valid/test/infer 四个 LMDB 文件"""
        for name, content_list in [('train.lmdb', train),
                                    ('valid.lmdb', valid),
                                    ('test.lmdb', test),
                                    ('infer.lmdb', infer)]:
            os.makedirs(dataset_outpath, exist_ok=True)

            if debug:
                output_json_name = os.path.join(dataset_outpath, name.replace(".lmdb", ".json"))
                with open(output_json_name, "w") as outfile:
                    json.dump(content_list, outfile, indent=4)

            output_name = os.path.join(dataset_outpath, name)
            try:
                os.remove(output_name)
            except:
                pass

            env_new = lmdb.open(
                output_name,
                subdir=False,
                readonly=False,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
                map_size=int(100e9),
            )
            txn_write = env_new.begin(write=True)
            with Pool(nthreads) as pool:
                i = 0
                for inner_output in tqdm(pool.imap(lnp2data_w_smi2mol_id, content_list), total=len(content_list), desc=name):
                    if inner_output is not None:
                        txn_write.put(f'{i}'.encode("ascii"), inner_output)
                        i += 1
                print('{} process {} lines'.format(name, i))
                txn_write.commit()
                env_new.close()

    def filter_target_label(data_list, label_name):
        new_data_list = []
        for data_sample in data_list:
            if label_name in data_sample['labels'].keys():
                if len(data_sample['labels'].keys()) > 1:
                    new_data_sample = copy.deepcopy(data_sample)
                    new_data_sample['labels'] = {label_name: data_sample['labels'][label_name]}
                    new_data_list.append(new_data_sample)
                else:
                    new_data_list.append(data_sample)
        return new_data_list

    # ===================== 对每个数据集进行处理 =====================
    all_output_train_ids = {}
    all_output_valid_ids = {}
    all_output_test_ids = {}

    for dataset_name in dataset_dict:
        dataset_json_list = dataset_dict[dataset_name]

        if not fixed_train_valid_test_split:
            if shuffle:
                np.random.shuffle(dataset_json_list)
        dataset_sz = len(dataset_json_list)

        if kfold_valid is None:
            # 普通训练/验证/测试划分
            dataset_outpath = os.path.join(outpath, dataset_name)
            if valid_ratio is None:
                valid_ratio = test_ratio

            print("fixed_train_valid_test_split: ", fixed_train_valid_test_split)
            if fixed_train_valid_test_split:
                dataset_json_lnp_id2obj = {}
                for lnp_obj in dataset_json_list:
                    dataset_json_lnp_id2obj[lnp_obj['lnp_id']] = lnp_obj
                train, valid, test = [], [], []
                for split, split_lnp_ids in [(train, train_lnp_ids), (valid, valid_lnp_ids), (test, test_lnp_ids)]:
                    for lnp_id in split_lnp_ids:
                        lnp_obj_to_add = dataset_json_lnp_id2obj[lnp_id]
                        split.append(lnp_obj_to_add)
                print("fixed_train_valid_test_split, train len: ", len(train))
                print("fixed_train_valid_test_split, valid len: ", len(valid))
                print("fixed_train_valid_test_split, test len: ", len(test))
            else:
                if top_bottom_ratio > 0:
                    train_valid, test = get_train_test_split_with_heldout_topbottom(
                        dataset_json_list, target_label_name,
                        bottom_heldout_ratio, top_heldout_ratio, remaining_random_test_ratio
                    )
                    train = train_valid[:int(dataset_sz * (1 - test_ratio - valid_ratio))]
                    valid = train_valid[int(dataset_sz * (1 - test_ratio - valid_ratio)):]
                else:
                    train = dataset_json_list[:int(dataset_sz * (1 - test_ratio - valid_ratio))]
                    valid = dataset_json_list[int(dataset_sz * (1 - test_ratio - valid_ratio)):int(dataset_sz * (1 - test_ratio))]
                    test = dataset_json_list[int(dataset_sz * (1 - test_ratio)):]

            train = subsample_train(train, random_train_subsample_ratio, train_subsample_sample_ids, subsample_target_label)

            if split_multilabel_train_sample:
                new_train = []
                for train_sample in train:
                    if len(train_sample['labels'].keys()) > 1:
                        for label_name in train_sample['labels'].keys():
                            new_train_sample = copy.deepcopy(train_sample)
                            new_train_sample['labels'] = {label_name: train_sample['labels'][label_name]}
                            new_train.append(new_train_sample)
                    else:
                        new_train.append(train_sample)
                train = new_train

            if type(labels_to_split_into_subfolders) == list:
                for label_name in labels_to_split_into_subfolders:
                    label_dataset_name = label_name
                    label_dataset_outpath = os.path.join(outpath, label_dataset_name)
                    label_train = filter_target_label(train, label_name)
                    label_valid = filter_target_label(valid, label_name)
                    label_test = filter_target_label(test, label_name)
                    infer = filter_target_label(dataset_json_list, label_name)
                    make_data_lmdb(label_train, label_valid, label_test, infer, label_dataset_outpath, debug)
            else:
                infer = dataset_json_list
                make_data_lmdb(train, valid, test, infer, dataset_outpath, debug)

            output_train_lnp_ids = [lnp_obj['lnp_id'] for lnp_obj in train]
            output_valid_lnp_ids = [lnp_obj['lnp_id'] for lnp_obj in valid]
            output_test_lnp_ids = [lnp_obj['lnp_id'] for lnp_obj in test]

            all_output_train_ids[dataset_name] = output_train_lnp_ids
            all_output_valid_ids[dataset_name] = output_valid_lnp_ids
            all_output_test_ids[dataset_name] = output_test_lnp_ids

        else:
            # K-Fold 交叉验证
            fold_dir_paths = []

            if top_bottom_ratio > 0:
                train_valid, test = get_train_test_split_with_heldout_topbottom(
                    dataset_json_list, target_label_name,
                    bottom_heldout_ratio, top_heldout_ratio, remaining_random_test_ratio
                )
            else:
                train_valid = dataset_json_list[:int(dataset_sz * (1 - test_ratio))]
                test = dataset_json_list[int(dataset_sz * (1 - test_ratio)):]

            output_test_lnp_ids = [lnp_obj['lnp_id'] for lnp_obj in test]

            kf_train = KFold(n_splits=kfold_valid, shuffle=shuffle, random_state=SEED)
            kfold_output_train_lnp_ids = {}
            kfold_output_valid_lnp_ids = {}

            for i_valid_fold, (train_index, valid_index) in enumerate(kf_train.split(train_valid)):
                fold_subdir_name = "fold_V" + str(i_valid_fold)
                fold_subdir_outpath = os.path.join(outpath, fold_subdir_name)
                os.makedirs(fold_subdir_outpath, exist_ok=True)

                print("i_valid_fold: ", i_valid_fold)
                print("train_index len: ", len(train_index))
                print("valid_index len: ", len(valid_index))
                train = list(np.array(train_valid)[train_index])
                valid = list(np.array(train_valid)[valid_index])

                if fold_subdir_outpath not in fold_dir_paths:
                    fold_dir_paths.append(fold_subdir_outpath)

                dataset_outpath = os.path.join(fold_subdir_outpath, dataset_name)
                train = subsample_train(train, random_train_subsample_ratio, train_subsample_sample_ids, subsample_target_label)

                if split_multilabel_train_sample:
                    new_train = []
                    for train_sample in train:
                        if len(train_sample['labels'].keys()) > 1:
                            for label_name in train_sample['labels'].keys():
                                new_train_sample = copy.deepcopy(train_sample)
                                new_train_sample['labels'] = {label_name: train_sample['labels'][label_name]}
                                new_train.append(new_train_sample)
                        else:
                            new_train.append(train_sample)
                    train = new_train

                if type(labels_to_split_into_subfolders) == list:
                    for label_name in labels_to_split_into_subfolders:
                        label_dataset_name = label_name
                        label_dataset_outpath = os.path.join(fold_subdir_outpath, label_dataset_name)
                        label_train = filter_target_label(train, label_name)
                        label_valid = filter_target_label(valid, label_name)
                        label_test = filter_target_label(test, label_name)
                        infer = filter_target_label(dataset_json_list, label_name)
                        print("label_dataset_outpath: ", label_dataset_outpath)
                        make_data_lmdb(label_train, label_valid, label_test, infer, label_dataset_outpath, debug)
                else:
                    infer = dataset_json_list
                    make_data_lmdb(train, valid, test, infer, dataset_outpath, debug)

                fold_output_train_lnp_ids = [lnp_obj['lnp_id'] for lnp_obj in train]
                kfold_output_train_lnp_ids[fold_subdir_name] = fold_output_train_lnp_ids
                fold_output_valid_lnp_ids = [lnp_obj['lnp_id'] for lnp_obj in valid]
                kfold_output_valid_lnp_ids[fold_subdir_name] = fold_output_valid_lnp_ids

            # 将 mol.lmdb 复制到每个 fold 目录
            for fold_dir_path in fold_dir_paths:
                shutil.copy(mol_output_name, fold_dir_path)

            all_output_train_ids[dataset_name] = kfold_output_train_lnp_ids
            all_output_valid_ids[dataset_name] = kfold_output_valid_lnp_ids
            all_output_test_ids[dataset_name] = output_test_lnp_ids

    return all_output_train_ids, all_output_valid_ids, all_output_test_ids


# ======================== 各 Figure 配置 ========================

DATA_DIR = "data_json"
OUT_DIR = "processed_data_dirs"

# 输入文件路径
JSON_DC24_ONLY = os.path.join(DATA_DIR, "in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_onlyDC24_3028.json")
JSON_DUAL_LABEL = os.path.join(DATA_DIR, "in_house_lnp_data_overall_lance_without_pbae+caco2_2024-04-16_npratios_foldcaco2label_3028.json")
JSON_WITH_PBAE = os.path.join(DATA_DIR, "in_house_lnp_data_overall_new_full_with_pbae_NPratios_updated_09222023_npratios.json")

# 拆分后的单标签数据集（用于E1实验的单任务独立训练）
JSON_B16F10_ONLY = os.path.join(DATA_DIR, "in_house_lnp_data_overall_lance_without_pbae+caco2_2024-04-16_npratios_foldcaco2label_3028_B16F10.json")
JSON_DC24_ONLY_SPLIT = os.path.join(DATA_DIR, "in_house_lnp_data_overall_lance_without_pbae+caco2_2024-04-16_npratios_foldcaco2label_3028_DC24.json")


def run_fig2a():
    """Fig 2a: 随机划分 20% 测试集，8 折交叉验证，仅 DC24 标签"""
    print("\n========== Running Fig 2a ==========")
    return write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_DC24_ONLY,
        outpath=os.path.join(OUT_DIR, "OS_demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_onlyDC24_09252023gen_fig2a"),
        nthreads=8,
        kfold_valid=8,
        test_ratio=0.2,
        valid_ratio=0.1,
        debug=True,
    )


def run_fig2bi():
    """Fig 2b(i): Top 10% + Bottom 10% 作为测试集，仅 DC24 标签"""
    print("\n========== Running Fig 2b(i) ==========")
    return write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_DC24_ONLY,
        outpath=os.path.join(OUT_DIR, "OS_demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_onlyDC24_09252023gen_fig2bi"),
        nthreads=8,
        kfold_valid=8,
        test_ratio=0.2,
        top_heldout_ratio=0.1,
        bottom_heldout_ratio=0.1,
        target_label_name="in_house_lnp_DC24_luc",
        debug=True,
    )


def run_fig2bii():
    """Fig 2b(ii): Top 10% + 随机 10% 作为测试集，仅 DC24 标签"""
    print("\n========== Running Fig 2b(ii) ==========")
    return write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_DC24_ONLY,
        outpath=os.path.join(OUT_DIR, "OS_demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_onlyDC24_09252023gen_fig2bii"),
        nthreads=8,
        kfold_valid=8,
        test_ratio=0.2,
        top_heldout_ratio=0.1,
        target_label_name="in_house_lnp_DC24_luc",
        debug=True,
    )


def run_fig3di():
    """Fig 3d(i): 随机划分，双标签（B16F10 + DC24）"""
    print("\n========== Running Fig 3d(i) ==========")
    return write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_DUAL_LABEL,
        outpath=os.path.join(OUT_DIR, "OS_demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fig3di"),
        nthreads=8,
        kfold_valid=8,
        test_ratio=0.2,
        valid_ratio=0.1,
        debug=True,
    )


def run_fig3dii():
    """Fig 3d(ii): Top 10% + 随机 10% 作为测试集，双标签"""
    print("\n========== Running Fig 3d(ii) ==========")
    return write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_DUAL_LABEL,
        outpath=os.path.join(OUT_DIR, "OS_demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fig3dii"),
        nthreads=8,
        kfold_valid=8,
        test_ratio=0.2,
        top_heldout_ratio=0.1,
        target_label_name="in_house_lnp_DC24_luc",
        debug=True,
    )


def run_fig3di_b16f10():
    """Fig 3d(i) B16F10: 拆分数据集，仅 B16F10 单标签，随机划分（无 heldout）"""
    print("\n========== Running Fig 3d(i) B16F10 (单标签) ==========")
    return write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_B16F10_ONLY,
        outpath=os.path.join(OUT_DIR, "OS_demo_in_house_lnp_data_overall_lance_B16F10_only_09252023gen_fig3di"),
        nthreads=8,
        kfold_valid=8,
        test_ratio=0.2,
        valid_ratio=0.1,
        debug=True,
    )


def run_fig3dii_b16f10():
    """Fig 3d(ii) B16F10: 拆分数据集，仅 B16F10 单标签，Top 10% + 随机 10% 测试集"""
    print("\n========== Running Fig 3d(ii) B16F10 (单标签) ==========")
    return write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_B16F10_ONLY,
        outpath=os.path.join(OUT_DIR, "OS_demo_in_house_lnp_data_overall_lance_B16F10_only_09252023gen_fig3dii"),
        nthreads=8,
        kfold_valid=8,
        test_ratio=0.2,
        top_heldout_ratio=0.1,
        target_label_name="in_house_lnp_B16F10_luc",
        debug=True,
    )


def run_fig3di_dc24():
    """Fig 3d(i) DC24: 拆分数据集，仅 DC24 单标签，随机划分（无 heldout）"""
    print("\n========== Running Fig 3d(i) DC24 (单标签) ==========")
    return write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_DC24_ONLY_SPLIT,
        outpath=os.path.join(OUT_DIR, "OS_demo_in_house_lnp_data_overall_lance_DC24_only_09252023gen_fig3di"),
        nthreads=8,
        kfold_valid=8,
        test_ratio=0.2,
        valid_ratio=0.1,
        debug=True,
    )


def run_fig3dii_dc24():
    """Fig 3d(ii) DC24: 拆分数据集，仅 DC24 单标签，Top 10% + 随机 10% 测试集"""
    print("\n========== Running Fig 3d(ii) DC24 (单标签) ==========")
    return write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_DC24_ONLY_SPLIT,
        outpath=os.path.join(OUT_DIR, "OS_demo_in_house_lnp_data_overall_lance_DC24_only_09252023gen_fig3dii"),
        nthreads=8,
        kfold_valid=8,
        test_ratio=0.2,
        top_heldout_ratio=0.1,
        target_label_name="in_house_lnp_DC24_luc",
        debug=True,
    )


def run_fig4b():
    """Fig 4b: 含 PBAE 的随机划分"""
    print("\n========== Running Fig 4b ==========")
    return write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_WITH_PBAE,
        outpath=os.path.join(OUT_DIR, "OS_demo_in_house_lnp_data_overall_new_full_with_pbae_NPratios_updated_09222023_npratios_09252023gen_fig4b"),
        nthreads=8,
        kfold_valid=8,
        test_ratio=0.2,
        debug=True,
    )


def run_fig4bii():
    """Fig 4b(ii): 含 PBAE 的 Top 10% + 随机 10% 划分"""
    print("\n========== Running Fig 4b(ii) ==========")
    return write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_WITH_PBAE,
        outpath=os.path.join(OUT_DIR, "OS_demo_in_house_lnp_data_overall_new_full_with_pbae_NPratios_updated_09222023_npratios_09252023gen_fig4bii"),
        nthreads=8,
        kfold_valid=8,
        test_ratio=0.2,
        top_heldout_ratio=0.1,
        target_label_name="in_house_lnp_DC24_luc",
        debug=True,
    )


def run_fig4c():
    """Fig 4c: 集成模型训练，5 折，无测试集（test_ratio=0）"""
    print("\n========== Running Fig 4c ==========")
    return write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_WITH_PBAE,
        outpath=os.path.join(OUT_DIR, "OS_demo_in_house_lnp_data_overall_new_full_with_pbae_NPratios_updated_09222023_npratios_09252023gen_fig4cDeployTrain"),
        nthreads=8,
        kfold_valid=5,
        test_ratio=0,
        debug=True,
    )


def run_fig2a_extra(version=1):
    """Fig 2a 额外版本 V1/V2（不同随机种子划分）"""
    print(f"\n========== Running Fig 2a V{version} ==========")
    return write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_DC24_ONLY,
        outpath=os.path.join(OUT_DIR, f"OS_demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_onlyDC24_09252023gen_fig2a_V{version}"),
        nthreads=8,
        kfold_valid=8,
        test_ratio=0.2,
        debug=True,
    )


def run_fig3di_extra(version=1):
    """Fig 3di 额外版本 V1/V2"""
    print(f"\n========== Running Fig 3di V{version} ==========")
    return write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_DUAL_LABEL,
        outpath=os.path.join(OUT_DIR, f"OS_demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fig3di_V{version}"),
        nthreads=8,
        kfold_valid=8,
        test_ratio=0.2,
        debug=True,
    )


def run_fig3aii(train_percent=100):
    """Fig 3a(ii): 多任务学习，训练集子采样不同比例"""
    print(f"\n========== Running Fig 3a(ii) {train_percent}% ==========")
    # 先读取 Fig 3dii V0 的划分
    input_folder = os.path.join(OUT_DIR, "demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fig3dii/fold_V0")

    data_split_lnp_dict = {}
    if os.path.exists(input_folder):
        for f in os.listdir(input_folder):
            folder_path = os.path.join(input_folder, f)
            if os.path.isdir(folder_path):
                for f2 in os.listdir(folder_path):
                    if f2.endswith(".json"):
                        json_file_path = os.path.join(folder_path, f2)
                        data_split = f2.split(".json")[0]
                        data_split_lnp_dict[data_split] = []
                        with open(json_file_path, "r") as fp:
                            dataset = json.load(fp)
                        for lnp in dataset:
                            data_split_lnp_dict[data_split].append(lnp['lnp_id'])

    train_ratio = train_percent / 100.0

    # 版本1：不拆分标签
    write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_DUAL_LABEL,
        outpath=os.path.join(OUT_DIR, f"OS_demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fold_V0_fig3aii{train_percent}_matchingfig3dii"),
        nthreads=8,
        test_ratio=0.2,
        random_train_subsample_ratio=train_ratio,
        subsample_target_label="in_house_lnp_B16F10_luc",
        train_lnp_ids=data_split_lnp_dict.get("train"),
        valid_lnp_ids=data_split_lnp_dict.get("valid"),
        test_lnp_ids=data_split_lnp_dict.get("test"),
        debug=True,
    )

    # 版本2：拆分标签
    write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_DUAL_LABEL,
        outpath=os.path.join(OUT_DIR, f"OS_demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fold_V0_fig3aii{train_percent}_splitlabels_matchingfig3dii"),
        nthreads=8,
        test_ratio=0.2,
        random_train_subsample_ratio=train_ratio,
        subsample_target_label="in_house_lnp_B16F10_luc",
        split_multilabel_train_sample=True,
        train_lnp_ids=data_split_lnp_dict.get("train"),
        valid_lnp_ids=data_split_lnp_dict.get("valid"),
        test_lnp_ids=data_split_lnp_dict.get("test"),
        debug=True,
    )


def run_pbae_bootstrap(pct=5):
    """PBAE 自举实验：训练集包含不同比例的 PBAE 样本"""
    print(f"\n========== Running PBAE Bootstrap {pct}% ==========")
    # 读取 Fig 4bii fold_V0 的划分
    test_path = os.path.join(OUT_DIR, "demo_in_house_lnp_data_overall_new_full_with_pbae_NPratios_updated_09222023_npratios_09252023gen_fig4bii/fold_V0/in_house_lnp/test_lnp_ids.json")
    valid_path = os.path.join(OUT_DIR, "demo_in_house_lnp_data_overall_new_full_with_pbae_NPratios_updated_09222023_npratios_09252023gen_fig4bii/fold_V0/in_house_lnp/valid_lnp_ids.json")
    train_path = os.path.join(OUT_DIR, f"demo_in_house_lnp_data_overall_new_full_with_pbae_NPratios_updated_09222023_npratios_09252023gen_fig4bii/fold_V0/in_house_lnp/train_lnp_ids_PBAELNPtrain{pct}pct.json")

    with open(test_path, 'r') as fp:
        test_lnp_ids = json.load(fp)
    with open(valid_path, 'r') as fp:
        valid_lnp_ids = json.load(fp)
    with open(train_path, 'r') as fp:
        train_lnp_ids = json.load(fp)

    print(f"train_lnp_ids len: {len(train_lnp_ids)}")

    return write_lmdb_concatdataset_w_topbottomheldout_trainsubsampling(
        inpath=JSON_WITH_PBAE,
        outpath=os.path.join(OUT_DIR, f"OS_demo_in_house_lnp_data_overall_new_full_with_pbae_NPratios_updated_09222023_npratios_09252023gen_fig4biiPBAELNPtrain{pct}pct_fold_V0"),
        nthreads=8,
        train_lnp_ids=train_lnp_ids,
        valid_lnp_ids=valid_lnp_ids,
        test_lnp_ids=test_lnp_ids,
        debug=True,
    )


# ======================== 主入口 ========================

FIGURE_MAP = {
    'fig2a': run_fig2a,
    'fig2bi': run_fig2bi,
    'fig2bii': run_fig2bii,
    'fig3di': run_fig3di,
    'fig3di_b16f10': run_fig3di_b16f10,       # 拆分数据集：B16F10单标签，fig3di随机划分
    'fig3di_dc24': run_fig3di_dc24,             # 拆分数据集：DC24单标签，fig3di随机划分
    'fig3dii': run_fig3dii,
    'fig3dii_b16f10': run_fig3dii_b16f10,     # 拆分数据集：B16F10单标签，fig3dii heldout
    'fig3dii_dc24': run_fig3dii_dc24,           # 拆分数据集：DC24单标签，fig3dii heldout
    'fig4b': run_fig4b,
    'fig4bii': run_fig4bii,
    'fig4c': run_fig4c,
    'fig2a_v1': lambda: run_fig2a_extra(1),
    'fig2a_v2': lambda: run_fig2a_extra(2),
    'fig3di_v1': lambda: run_fig3di_extra(1),
    'fig3di_v2': lambda: run_fig3di_extra(2),
}


def main():
    parser = argparse.ArgumentParser(description="COMET 数据预处理脚本")
    parser.add_argument(
        '--fig', type=str, default='fig2a',
        help='要执行的 Figure 配置。可选：fig2a, fig2bi, fig2bii, fig3di, fig3dii, fig3dii_b16f10, fig3dii_dc24, fig4b, fig4bii, fig4c, fig2a_v1, fig2a_v2, fig3di_v1, fig3di_v2, all'
    )
    parser.add_argument(
        '--nthreads', type=int, default=8,
        help='并行进程数（默认 8）'
    )
    parser.add_argument(
        '--fig3aii', action='store_true',
        help='执行 Fig 3a(ii) 的所有训练比例（5,10,15,25,35,50,75,100）'
    )
    parser.add_argument(
        '--pbae_bootstrap', action='store_true',
        help='执行 PBAE 自举实验（5, 25, 50, 75）'
    )
    args = parser.parse_args()

    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.fig3aii:
        # Fig 3a(ii) 需要 Fig 3dii 的划分结果，先检查
        for p in [100, 0, 5, 10, 15, 25, 35, 50, 75]:
            run_fig3aii(p)
        return

    if args.pbae_bootstrap:
        for p in [5, 25, 50, 75]:
            run_pbae_bootstrap(p)
        return

    if args.fig == 'all':
        # 执行所有主要 Figure
        for fig_name in ['fig2a', 'fig2bi', 'fig2bii', 'fig3di', 'fig3dii', 'fig4b', 'fig4bii', 'fig4c']:
            print(f"\n{'='*60}")
            print(f"Running {fig_name}")
            print('='*60)
            FIGURE_MAP[fig_name]()
    elif args.fig in FIGURE_MAP:
        FIGURE_MAP[args.fig]()
    else:
        print(f"错误：未知的 Figure 名称 '{args.fig}'")
        print(f"可选值：{list(FIGURE_MAP.keys()) + ['all']}")
        sys.exit(1)


if __name__ == '__main__':
    main()
