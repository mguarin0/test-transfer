"""
@author: mguarin0
desc: this script is used for calculating Relative Accessible Surface Area via stride
      and writes out to a pickle file
"""

from timeit import default_timer as timer
import os
import subprocess
import numpy as np
import traceback
import logging
import re
from utils import *

__author__ = "Michael Guarino"
__version__ = "0.1"
__all__ = [
        "relative_accessible_surface_area_main"
]

def _run_cmd(stride_cmd):
    """
    desc:
    args:
    returns:
    """
    popen = subprocess.Popen(stride_cmd, stdout=subprocess.PIPE)
    return popen
# end

def _separate_chain(flat_ress, idx_chain_letter):
    """
    desc:
    args:
    returns:
    """
    idx_splits = []
    for i, res in enumerate(flat_ress):
        chain_letter = res[idx_chain_letter] 
        if i==0:
            prev_chain_letter = ""
        if chain_letter != prev_chain_letter:
            idx_splits.append(i)
            prev_chain_letter = chain_letter
    idx_splits.append(len(flat_ress))
    return [flat_ress[idx_splits[i-1]:idx_splits[i]]
                 for i in range(len(idx_splits)) if i!=0]
# end

def relative_accessible_surface_area_main(kwargs):
    """
    desc:
    args:
    returns:
    """
    start_time = timer()
    try:
        assert(all(k in ["run_type", "stride_path", "ent_filename"]
                        for k in kwargs.keys())), "incorrect list of args"
        assert(kwargs["run_type"]=="r_stride"), "run type does not match argument passed"
        assert(os.path.exists(kwargs["stride_path"])), "stride path does not exist"
        assert(os.path.exists(kwargs["ent_filename"])), "ent filename does not exist"


        stride_cmd = [kwargs["stride_path"], kwargs["ent_filename"]]

        paths = prjPaths(run_type=kwargs["run_type"])
        filename, ext = kwargs["ent_filename"].split("/")[-1].split(".")
        if ext == "ent":
            complex_code = filename[3:7]
        elif ext == "bio1" or ext == "pdb":
            complex_code = filename[:4]

        popen = _run_cmd(stride_cmd) 
        if ext in ["ent", "bio1"]:
            exit("this script will not work properly with bio1 or ent files at this time")
            indices_rsaa_start = 64; indices_rsaa_end = 70
            output = popen.stdout.read().decode("utf-8").split("\n")
            arsa = [float(ent[indices_rsaa_start:indices_rsaa_end])/max_accessilbe_surface_area_by_aa[ent[5:8].lower()]["Theoretical"]
                                  for ent in output if ent[:3]=="ASG"]
        elif ext == "pdb":
            chunk_size = 10; idx_residue_name=1; idx_chain_letter=2; idx_residue_seq_num=3; idx_arsa=9;
            output = " ".join(list(filter(None, [line for line in popen.stdout.readline().decode("utf-8").split(" ")]))) # split stride tool output on space and filter out white spaces and join on whitespace
            #search = "\-\-\-\-\-\-\-\-\-\-\-\-\-\-\- Detailed secondary structure assignment\-\-\-\-\-\-\-\-\-\-\-\-\- REM REM \|\-\-\-Residue\-\-\-\| \|\-\-Structure\-\-\| \|\-Phi\-\| \|\-Psi\-\| \|\-Area\-\|"
            search = "\|\-Area\-\|"
            m = re.search(search, output) # search for detailed secondary structure assignment label in stride output
            subset_output = output[m.end():] # subset stride output to narrow down to detailed secondary structure assignment
            subset_output_list = list(filter(None, subset_output.split(" "))) # convert to list and filter out empty string
            subset_output_lists = [subset_output_list[i:i+chunk_size]
                                       for i in range(len(subset_output_list))[::chunk_size]] # split list into entry
            # filter out non detailed secondary structure assignment information
            subset_output_lists = list(filter(lambda x: x[0]=="ASG", subset_output_lists))
            subset_output_lists_by_chain = _separate_chain(flat_ress=subset_output_lists,
                                                           idx_chain_letter=idx_chain_letter) 
            arsa = {}
            for chain in subset_output_lists_by_chain:
                for i_res, residue in enumerate(chain):
                    chain_id = "{}_{}".format(str(residue[idx_chain_letter]),
                                              complex_code)
                    res_id = "{}_{}_{}".format(*list(map(str, [residue[idx_residue_seq_num],
                                                               i_res,
                                                               residue[idx_residue_name]])))
                    if (chain_id, res_id) in arsa.keys():
                        logging.error("|| complex_code: {} "
                                      "|| arsa already contains key: {}".format(complex_code,
                                                                                   list(map(str, (chain_id, res_id)))))
                    arsa[(chain_id, res_id)] = float(residue[idx_arsa])

        to_pickle(obj_to_pickle=arsa,
                  path=os.path.join(paths.RUN_TYPE_OUT_DIR,
                                    "r_stride_{}.p".format(complex_code)))
        end_time = timer()
        run_time = end_time-start_time
        logging.info("|| complex_code: {} "
                     "|| stride shape: ({}, 1)".format(complex_code, len(arsa.keys())))
        logging.info("|| complex_code: {} "
                     "|| run time (in seconds): {}".format(complex_code, run_time))
        return_msg = {complex_code: run_time}
    except Exception as err:
        logging.error("|| filename: {} "
                      "|| error thrown: {}".format(kwargs["ent_filename"],
                                                   traceback.format_exc()))
        end_time = timer()
        run_time = end_time-start_time
        return_msg = {kwargs["ent_filename"]: run_time}
        pass
    finally:
        return return_msg
# end
