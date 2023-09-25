import os
import pandas as pd
import cPickle
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_score, recall_score

def get_pickle_data(path_to_pickle):
    return cPickle.load(open(path_to_pickle))
# end

def fetch_predictions(persist_dir, replica_of_interest, run_type_flag):
    complex_data = {}
    for cpkl_file in os.listdir(persist_dir):
        if int(cpkl_file.split("_")[1]) == replica_of_interest:
            if cpkl_file.split("_")[0] == run_type_flag:
                complex_code = cpkl_file.split("_")[2].split(".")[0]
                cpkl_data = get_pickle_data(os.path.join(persist_dir, cpkl_file))
                l_idxs = [row[0] for row in cpkl_data]
                r_idxs = [row[1] for row in cpkl_data]
                y_trues = list(map(lambda x: 1 if x==1 else 0, [row[2] for row in cpkl_data]))
                y_preds = [row[3][0] for row in cpkl_data]
                d = {"l_idxs": l_idxs, "r_idxs": r_idxs, "y_trues": y_trues, "y_preds": y_preds}            
                complex_data[complex_code] = {"df": pd.DataFrame(data=d)}
    return complex_data
# end

def calc_basic_performance_metrics(complex_data):
    for complex_code in complex_data.keys():

        fpr, tpr, threshold = roc_curve(complex_data[complex_code]["df"]["y_trues"].tolist(),
                                      complex_data[complex_code]["df"]["y_preds"].tolist())
        complex_data[complex_code]["fpr"] = fpr
        complex_data[complex_code]["tpr"] = tpr
        complex_data[complex_code]["threshold"] = threshold
        complex_data[complex_code]["auc"] = auc(fpr, tpr)
        complex_data[complex_code]["average_precision_score"] = average_precision_score(complex_data[complex_code]["df"]["y_trues"].tolist(),
                                                                                      complex_data[complex_code]["df"]["y_preds"].tolist())

    return complex_data
# end

def calc_precision_recall(y_trues, y_preds):

    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    precision = precision_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    
    return (precision, recall)
# end

def aggregate_predictions_all_complexes(complex_data):

    all_y_trues = []
    all_y_preds = []

    for complex_code in complex_data.keys():
        all_y_trues.extend(complex_data[complex_code]["df"]["y_trues"].tolist())
        all_y_preds.extend(complex_data[complex_code]["df"]["y_preds"].tolist())
    return (all_y_trues, all_y_preds)
# end
