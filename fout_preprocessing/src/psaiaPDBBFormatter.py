"""
@author: mguarin0
desc: this script formats pdbb protein data files in order to be
      to be taken as input by the psaia tool. pdbb protein files
      (those with extension .pdb) have a few lines of metadata 
      included at the beginning of the file that cause the psaia
      tool to throw errors. This script removes those metadata
      lines and nests the pdbb protein files in directories
      `path_to_data_dst_dir` of size `file_batch_size`. psaia
      tool will not accept full dataset as input at one time thus
      the need to batch the data. This script then tars the
      `path_to_data_dst_dir` and copies it to s3 `s3_dst_bucket`.
      It assumes that you have performed `aws configure`.
"""

import argparse
import os
from multiprocessing import Pool
from subprocess import Popen
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
    parser.add_argument("path_to_data_src_dir", help="source data dir", type=str)
    parser.add_argument("path_to_data_dst_dir", help="destination data dir", type=str)
    parser.add_argument("s3_dst_bucket", help="s3 bucket to move to", type=str)
    parser.add_argument("file_batch_size", default = 400, help="number of files to be included in one run of psaia tool", type=int)
    parser.add_argument("remove_first_n_lines", help="remove first n lines of file", type=int)
    args = parser.parse_args()
    return args
# end

def remove_first_n_lines(args):
    """
    desc:
    args:
    returns:
    """
    path_filename, n_lines = args
    with open(path_filename, "r") as file:
        lines = file.readlines()

    os.remove(path_filename)

    with open(path_filename, "w") as file:
        file.write("".join(lines[n_lines:]))
# end

if __name__ == "__main__":
    args = get_args()

    if not os.path.exists(args.path_to_data_dst_dir):
        os.mkdir(args.path_to_data_dst_dir)

    filebuckets = [os.listdir(args.path_to_data_src_dir)[idx:idx+args.file_batch_size] for idx in range(0, len(os.listdir(args.path_to_data_src_dir)), args.file_batch_size)]
    for i, filebucket in enumerate(filebuckets):
        current_dir = os.path.join(args.path_to_data_dst_dir, "part_{}".format(i))
        os.mkdir(current_dir) 
        for file in filebucket:
            shutil.copy(os.path.join(args.path_to_data_src_dir, file), os.path.join(current_dir, file))

    pool = Pool(os.cpu_count())
    for dir in os.listdir(args.path_to_data_dst_dir):
        if os.path.isdir(os.path.join(args.path_to_data_dst_dir, dir)):
            pool.map(remove_first_n_lines, [[os.path.join(args.path_to_data_dst_dir, dir, file), args.remove_first_n_lines] for file in os.listdir(os.path.join(args.path_to_data_dst_dir, dir)) if file.split("/")[-1].split(".")[1]=="pdb"])
    pool.close()
    pool.join()

    Popen(["tar", "-cvf", "{}.tar.gz".format(args.path_to_data_dst_dir), args.path_to_data_dst_dir]).wait()
    Popen(["aws", "s3", "cp", "{}.tar.gz".format(args.path_to_data_dst_dir), "s3://{}".format(args.s3_dst_bucket)]).wait()
