import argparse
import os
import sys
import shutil

__author__ = "Michael Guarino"
__version__ = "0.1"

def get_args():
    """
    desc: get cli arguments
    returns:
    args: dictionary of cli arguments
    """
    parser = argparse.ArgumentParser(description="this script calculates edge features and hood indices for receptor protein")
    parser.add_argument("full_path_data_dir", help="full path to directory containing dataset", type=str)
    parser.add_argument("full_path_dst_data_dir", help="full path to destination directory", type=str)
    args = parser.parse_args()
    return args
# end

#Generate the file paths to traverse
def getfiles(path):
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for name in files:
                yield os.path.join(root, name)
    else:
        yield path
# end

def _mkdir(new_path):
    if not os.path.exists(new_path):
        os.mkdir(new_path)
# end

if __name__=="__main__":

    args = get_args()
    _mkdir(args.full_path_dst_data_dir)
    for dir in ["protein_pdb",
                "pocket_pdb",
                "ligand_mol2",
                "ligand_sdf"]: _mkdir(os.path.join(args.full_path_dst_data_dir, dir))

    for f in getfiles(args.full_path_data_dir):
        full_filename = f.split("/")[-1]
        filename, ext = full_filename.split(".")
        filename_class = filename.split("_")[-1]
        sub_dir = "{}_{}".format(filename_class, ext)
        if  in ["protein_pdb", "pocket_pdb", "ligand_mol2", "ligand_sdf"]:
            if os.path.isfile(os.path.join(args.full_path_dst_data_dir, sub_dir, full_filename)):
                full_filename = f.replace(args.full_path_data_dir,"",1).replace("/","_")
            shutil.copy(f, os.path.join(args.full_path_dst_data_dir, sub_dir, full_filename))


