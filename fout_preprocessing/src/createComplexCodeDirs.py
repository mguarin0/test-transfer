"""
@author: mguarin0
desc: copies feature calculations into subdirectories by complex code
      this is the first step in creating a features for ingestion
      by machine learning algorithm
"""

import argparse
import os
import logging
from collections import defaultdict
from shutil import copyfile

from utils import *

__author__ = "Michael Guarino"
__version__ = "0.1"

def get_args():
  """
  desc: get cli arguments
  returns:
  args: dictionary of cli arguments
  """
  parser = argparse.ArgumentParser(description="this script merges pickle files "
                                               "containing calculated features into "
                                               "directories by complex")
  parser.add_argument("root_feature_dir_path",
                      help="full path to directory "
                           "containing calculated features",
                      type=str)
  parser.add_argument("output_dir_path",
                      help="full path to output directory",
                      type=str)
  args = parser.parse_args()
  return args
# end

def _get_dir_features(feature_dir):
  """
  desc: given a path to a directory this function will organize
        picklefile paths by complex code
  args:
    feature_dir: path to a directory containing calculated features 
  returns:
    tuple where first element is dictionary where the complex code
    is the key and the value is a path to the pickle file second element
    is an int representing the number of pickle files
  """
  complex_code_picklefile_grps = [(str(picklefile.split("_")[-1].split(".")[0]),
                                   os.path.join(feature_dir, picklefile))
                                     for picklefile in list(filter(
                                       lambda x: x.split(".")[-1]=="p",
                                         os.listdir(feature_dir)))]
  if len(complex_code_picklefile_grps) == 0:
    return None
  else:
    picklefiles_by_complex_code = defaultdict(list)
    for complex_code, picklefile in complex_code_picklefile_grps:
      picklefiles_by_complex_code[complex_code].append(picklefile)
    return (picklefiles_by_complex_code,
            len(picklefiles_by_complex_code[list(picklefiles_by_complex_code.keys())[0]]))
# end

def _get_all_features(feature_dirs):
  """
  desc: given path to feature directories this function will assemble
        pickle files by complex code across all feature directories
  args:
    feature_dirs: path to a directory containing all feature directories
  returns:
    all_picklefiles_by_complex_code: dictionary where the complex code is
                                     the key and the value is a path to all
                                     pickle files across all feature directories
  """
  picklefiles_by_dir = list(filter(None, map(_get_dir_features, feature_dirs)))
  num_expected_features = sum(list(map(lambda x: x[1], picklefiles_by_dir)))

  all_picklefiles_by_complex_code = defaultdict(list)
  for picklefiles_by_complex_code, _ in picklefiles_by_dir:
    for complex_code, picklefile in picklefiles_by_complex_code.items():
      all_picklefiles_by_complex_code[complex_code].extend(picklefile)

  all_picklefiles_by_complex_code = {complex_code: picklefiles
                                       for complex_code, picklefiles in all_picklefiles_by_complex_code.items()
                                         if len(picklefiles)==num_expected_features}
  return all_picklefiles_by_complex_code 
# end

def merge_main():
  """
  desc: given cli args will merge all feature calculations by complex code directories
  """
  args = get_args()
  root_feature_dir_contents = list(map(lambda x: os.path.join(args.root_feature_dir_path, x),
                                       os.listdir(args.root_feature_dir_path)))

  root_feature_dir_dirs = list(filter(os.path.isdir, root_feature_dir_contents))
  all_picklefiles_by_complex_code = _get_all_features(feature_dirs=root_feature_dir_dirs)

  if not os.path.exists(args.output_dir_path): os.mkdir(args.output_dir_path)

  for complex_code, picklefiles in all_picklefiles_by_complex_code.items():
    complex_code_dir = os.path.join(args.output_dir_path, complex_code)
    if not os.path.exists(complex_code_dir): os.mkdir(complex_code_dir)
    for picklefile in picklefiles:
      copyfile(picklefile,
               os.path.join(complex_code_dir,
                            picklefile.split("/")[-1]))
# end

if __name__ == "__main__":
  merge_main()
