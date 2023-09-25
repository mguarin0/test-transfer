"""
@author: mguarin0
desc:
"""

import argparse
import os
from logging.handlers import QueueHandler
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
import logging
import traceback
from timeit import default_timer as timer

from utils import *

__author__ = "Michael Guarino"
__version__ = "0.1"

def get_args():
  """
  desc: get cli arguments
  returns:
    args: dictionary of cli arguments
  """
  parser = argparse.ArgumentParser(description="this script creates a pickle file "
                                               "representing all features for a given complex")
  parser.add_argument("root_complex_code_dir_path",
                      help="full path to directory containing all "
                           "features organized by complex code",
                      type=str)
  args = parser.parse_args()
  return args
# end

def _dir_reader(complex_code_dir_path):
  """
  desc:
  args:
  returns:
  """
  
  pickle_file_paths = list(filter(
                           lambda x: x.split(".")[-1]=="p",
                           map(lambda file: os.path.join(complex_code_dir_path,
                                                         file),
                               os.listdir(complex_code_dir_path))))
  complex_data = {"_".join(picklefile.split("/")[-1].split(".")[0].split("_")[:-1]):from_pickle(picklefile)
                  for picklefile in pickle_file_paths}
  logging.info("|| read complex_code: {}".format(complex_code_dir_path))
  return (complex_code_dir_path,
          complex_data)
# end

def _recode_labels(x):
  """
  desc: this function recodes labels to [1,-1]
  args:
    x: label
  returns:
    recoded label bound to [1,-1]
  """
  if x == 1: return 1
  else: return -1
# end

def _format(complex_data):
  """
  desc:
  args:
  returns:
  """
  r_vertex = defaultdict(list)
  for k in complex_data[1]:

    if k in ["r_stride",
             "r_protrusion_idx",
             "r_residueDepth_avg_dst",
             "r_residueDepth_ac",
             "r_hsaac"]:
      # assemble protein vertex features
      for r_k in complex_data[1][k]:
        if k == "r_hsaac":
          r_vertex[r_k].extend(complex_data[1][k][r_k][0])
          r_vertex[r_k].extend(complex_data[1][k][r_k][1])
        else:
          if isinstance(complex_data[1][k][r_k], list):
            r_vertex[r_k].extend(complex_data[1][k][r_k])
          else:
            r_vertex[r_k].append(complex_data[1][k][r_k])

    elif k in ["l_molecule_features"]:
      """
      l_molecule_features contains a tuple of
      (atom_features, hood_index, edge_features, coordinates)
      """
      # persist for good measure
      for l_i, l_v in enumerate(complex_data[1][k]):
        if l_i == 0:
          l_vertex = l_v
        elif l_i == 1:
          l_hood_indices = l_v
        elif l_i == 2:
          l_edge = l_v

    elif k in ["r_edge"]:
      r_edge = complex_data[1][k]

    elif k in ["r_hood_indices"]:
      r_hood_indices = complex_data[1][k]

    elif k in ["labels"]:
      label = complex_data[1][k]

  assert(list(set(r_vertex.keys()) ^ set(r_hood_indices.keys()))==[]
          and list(set(r_vertex.keys()) ^ set(r_edge.keys()))==[]
          and list(set(r_hood_indices.keys()) ^ set(r_edge.keys()))==[])

  # persist for good measure
  to_pickle(r_vertex,
            os.path.join(complex_data[0], "r_vertex.p"))
  to_pickle(l_vertex,
            os.path.join(complex_data[0], "l_vertex.p"))
  to_pickle(l_hood_indices,
            os.path.join(complex_data[0], "l_hood_indices.p"))
  to_pickle(l_edge,
            os.path.join(complex_data[0], "l_edge.p"))
  idx_key_map = {composite_protein_key:idx
                 for idx, composite_protein_key in enumerate(r_vertex.keys())}

  # recoding
  r_edge = sorted([(idx_key_map[k],v) for k, v in r_edge.items()], key=lambda x: x[0])
  r_edge = np.float64(np.asarray([row[1] for row in r_edge])) # make sure you recast it

  r_vertex = sorted([(idx_key_map[k],v) for k, v in r_vertex.items()], key=lambda x: x[0])
  r_vertex = np.float64(np.asarray([row[1]for row in r_vertex])) # make sure you recast it

  r_hood_indices_recoded = []
  for k, v in r_hood_indices.items():
    r_hood_indices_recoded_sub = []
    for v_sub in v:
      r_hood_indices_recoded_sub.append([idx_key_map[v_sub[0]]])
    r_hood_indices_recoded.append((idx_key_map[k], r_hood_indices_recoded_sub))
  r_hood_indices = sorted(r_hood_indices_recoded, key=lambda x: x[0])
  r_hood_indices = np.int64(np.asarray([row[1] for row in r_hood_indices]))

  l_edge = np.float64(l_edge)
  l_vertex = np.float64(l_vertex)
  l_hood_indices = np.int64(l_hood_indices)
  l_hood_indices = np.expand_dims(l_hood_indices,
                                  axis=2)
  label = np.int64(np.asarray([[k[0],
                                idx_key_map[k[1]],
                                _recode_labels(v)] for k, v in label.items()]))
  logging.info("r_edge.shape: {}".format(r_edge.shape))
  logging.info("r_vertex.shape: {}".format(r_vertex.shape))
  logging.info("r_hood_indices.shape: {}".format(r_hood_indices.shape))
  logging.info("l_edge.shape: {}".format(l_edge.shape)) 
  logging.info("l_vertex.shape: {}".format(l_vertex.shape))
  logging.info("l_hood_indices.shape: {}".format(l_hood_indices.shape)) 
  logging.info("label: {}".format(label)) 

  complex_code = complex_data[0].split("/")[-1]
  record = {
          "r_vertex":r_vertex,
          "r_hood_indices":r_hood_indices,
          "r_edge":r_edge,
          "l_vertex":l_vertex,
          "l_hood_indices":l_hood_indices,
          "l_edge":l_edge,
          "complex_code":complex_code,
          "label":label
         }

  to_pickle(record,
            os.path.join(complex_data[0], "record.p"))
  return (complex_code, record)
# end

def worker_init(queue):
    """
    desc:
    args:
    returns:
    """
    # all records from worker processes go to qh and then into q
    queue_handler = QueueHandler(queue)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(queue_handler)
# end

def main():
  """
  desc:
  args:
  returns:
  notes:
    check issue#18 for description of of molecular features
  """
  args = get_args()
  run_type = "merge"
  paths = prjPaths(run_type=run_type)
  logger, queue_listener, queue = logger_init(path=paths.LOGS_DIR,
                                              filename=run_type)
  logging.info("process_id: {} "
               "check how many cores this process uses with "
               "`ps -o pid,psr,comm -p <pid>`".format(os.getpid()))

  complex_code_dir_paths = list(filter(
                                os.path.isdir,
                                map(lambda complex_code_dir_path:
                                    os.path.join(args.root_complex_code_dir_path,
                                                 complex_code_dir_path),
                                    os.listdir(args.root_complex_code_dir_path))))
  master = []
  complex_codes = []
  records = []
  pool = Pool(os.cpu_count())
  for complex_data in pool.map(_dir_reader, complex_code_dir_paths):
    try:
      complex_code, record = _format(complex_data)
      complex_codes.append(complex_code)
      records.append(record)

    except Exception as err:
      logging.error("|| filename: {} "
                    "|| error thrown: {}".format(complex_data[0],
                                                 traceback.format_exc()))
      pass
  pool.close()
  pool.join()
  master.append(complex_codes)
  master.append(records)
  to_pickle(master,
            os.path.join(args.root_complex_code_dir_path, "master_records.p"),
            protocol_type=2)
# end

if __name__ == "__main__":
  main()
