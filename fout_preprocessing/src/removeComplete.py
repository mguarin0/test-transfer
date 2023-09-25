import argparse
import os

__author__ = "Michael Guarino"
__version__ = "0.1"

def get_args():
    """
    desc: get cli arguments
    returns:
    args: dictionary of cli arguments
    """
    parser = argparse.ArgumentParser(description="this script removes files from the dataset that have already have a pickle file representing a complete calculation")
    parser.add_argument("full_path_data_dir", help="full path to directory containing dataset", type=str)
    parser.add_argument("full_path_output", help="full path to output directory containing pickle files", type=str)
    args = parser.parse_args()
    return args
# end

if __name__=="__main__":

    args = get_args()

    all_pickle_complex_codes = [pfile.split("_")[-1].split(".")[0] for pfile in os.listdir(args.full_path_output) if pfile.split("_")[-1].split(".")[-1]=="p"]

    for pdbfile in os.listdir(args.full_path_data_dir):
        pdb_complex_code = pdbfile.split("_")[0]
        if pdb_complex_code in all_pickle_complex_codes:
            os.remove(os.path.join(args.full_path_data_dir, pdbfile))
