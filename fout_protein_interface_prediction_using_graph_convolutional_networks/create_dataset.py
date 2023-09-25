"""
@author: mguarin0
desc:
"""

import argparse
import os
import cPickle
from collections import Counter
import numpy as np
import traceback

def get_args():
    """
    desc: get cli arguments
    returns:
    args: dictionary of cli arguments
    """
    parser = argparse.ArgumentParser(description="this script calculates edge features and hood indices for receptor protein")
    parser.add_argument("--full_path_master_pickle", default="pl_data/master_records.cpkl", help="full path to dataset pickle file", type=str)
    parser.add_argument("--train_split", default=0.7, help="percent of data to be used for training", type=float)
    parser.add_argument("--validation_split", default=0.2, help="percent of data to be used for validation", type=float)
    parser.add_argument("--test_split", default=0.1, help="percent of data to be used for testing", type=float)
    args = parser.parse_args()
    return args
# end

if __name__=="__main__":

    args = get_args()

    datadir_path = "/".join(args.full_path_master_pickle.split("/")[:-1])
    master = cPickle.load(open(args.full_path_master_pickle))
    
    total = len(master[1])
    train_split = int(total * args.train_split)
    val_split = int(total * args.validation_split)
    test_split = int(total * args.test_split)

    print("training splits...")
    train_indices = []
    for train_idx in range(train_split):
        print("++++complex id: {}".format(master[1][train_idx]["complex_code"]))
        sorted_labels = np.asarray(list(reversed(sorted(master[1][train_idx]["label"], key=lambda x: x[2]))), dtype=np.int64)
        row_idx_neg_label, _ = np.where(sorted_labels==-1)
        row_idx_pos_label = np.asarray(list(range(row_idx_neg_label[0])))
      
        try:
            pos_rebalanced = sorted_labels[row_idx_pos_label]
            neg_rebalanced = sorted_labels[row_idx_neg_label[:len(row_idx_pos_label)*10]]
            balanced_labels = np.vstack((pos_rebalanced, neg_rebalanced))

            label_count_org = Counter(master[1][train_idx]["label"][:,2])
            new_label_count = Counter(balanced_labels[:,2])

            print("label_count_org: {}".format(label_count_org))
            print("new_label_count: {}".format(new_label_count))
            if new_label_count[1] == 0:
                continue
            else:
                train_indices.append(train_idx)
                master[1][train_idx]["label"] = balanced_labels 
        except Exception as er:
            ex_str = traceback.format_exc()
            pass

    print("validation splits...")
    val_indices = []
    for val_idx in range(train_split, (train_split+val_split)):
        print("++++complex id: {}".format(master[1][val_idx]["complex_code"]))
        sorted_labels = np.asarray(list(reversed(sorted(master[1][val_idx]["label"], key=lambda x: x[2]))), dtype=np.int64)
        label_count_org = Counter(master[1][val_idx]["label"][:,2])
        print("label_count_org: {}".format(label_count_org))
        if label_count_org[1] == 0:
            continue
        else:
            val_indices.append(val_idx)
            master[1][val_idx]["label"] = sorted_labels

    print("testing splits...")
    test_indices = []
    for test_idx in range((train_split+val_split), (train_split+val_split+test_split)):
        print("++++complex id: {}".format(master[1][test_idx]["complex_code"]))
        sorted_labels = np.asarray(list(reversed(sorted(master[1][test_idx]["label"], key=lambda x: x[2]))), dtype=np.int64)
        label_count_org = Counter(master[1][test_idx]["label"][:,2])
        print("label_count_org: {}".format(label_count_org))
        if label_count_org[1] == 0:
            continue
        else:
            test_indices.append(test_idx)
            master[1][test_idx]["label"] = sorted_labels

    with open(os.path.join(datadir_path, "cpi_train_balanced.cpkl"), "wb") as train_file:
        cPickle.dump(([master[0][i] for i in train_indices],
                      [master[1][i] for i in train_indices]), train_file)

    with open(os.path.join(datadir_path, "cpi_val.cpkl"), "wb") as val_file:
        cPickle.dump(([master[0][i] for i in val_indices],
                      [master[1][i] for i in val_indices]), val_file)

    with open(os.path.join(datadir_path, "cpi_test.cpkl"), "wb") as test_file:
        cPickle.dump(([master[0][i] for i in test_indices],
                      [master[1][i] for i in test_indices]), test_file)

