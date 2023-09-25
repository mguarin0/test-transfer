"""
@author: mguarin0
desc:
"""

import os
from timeit import default_timer as timer
import logging
import traceback
import numpy as np
from utils import *

__author__ = "Michael Guarino"
__version__ = "0.1"
__all__ = [
    "protrusion_idx_hydrophobicity_main"
]

def _tbl_file_reader(path_to_tbl_file, complex_code):
    """
    desc:
    args:
    returns:
    """
    assert(os.path.isfile(path_to_tbl_file)), "file does not exist"
    start_idx_data = 7 # first 7 lines are all meta info
    with open(path_to_tbl_file, "r") as file:
        psaia_records = []
        for i, line in enumerate(file.readlines()[start_idx_data:]):
            if i == 0:
                headers = list(map(lambda x: x.replace(" ", "_"), list(filter(None, map(str.strip, line.split("|"))))))
            else:
                current_line = list(filter(None, map(str.strip, line.split(" "))))
                if len(headers)!=len(current_line):
                    logging.error("|| complex_code: {} "
                                  "|| len(headers): {} "
                                  "|| len(current_line): {} "
                                  "|| line number: {}".format(complex_code,
                                                              len(headers),
                                                              len(current_line),
                                                              i+7))
                    logging.error("|| complex_code: {} "
                                  "|| headers: {}".format(complex_code,
                                                          headers))
                    logging.error("|| complex_code: {} "
                                  "|| current_line: {}".format(complex_code,
                                                               current_line))
                psaia_records.append(dict(zip(headers, current_line)))
    return psaia_records
# end

def _create_final_format(psaia_output_records, complex_code):
    """
    desc:
    args:
    returns:
    """
    psaia = {}
    for i_res, res in enumerate(psaia_output_records):
        chain_id = "{}_{}".format(res["chain_id"],
                                  complex_code)
        res_id = "{}_{}_{}".format(res["res_id"],
                                   str(i_res),
                                   res["res_name"])
        if (chain_id, res_id) in psaia.keys():
            logging.error("|| complex_code: {} "
                          "|| dst_agl already contains key: {}".format(complex_code,
                                                                       list(map(str, (chain_id, res_id)))))
        psaia[(chain_id, res_id)] = [res["average_CX"], 
                                     res["s_avg_CX"],
                                     res["s-ch_avg_CX"],
                                     res["s-ch_s_avg_CX"],
                                     res["max_CX"],
                                     res["min_CX"],
                                     res["Hydrophobicity"]]
    return psaia
# end

def protrusion_idx_hydrophobicity_main(kwargs):
    """
    desc:
    args:
    returns:
    """

    start_time = timer()
    try:
        assert(all(k in ["run_type", "tbl_filename"]
                        for k in kwargs.keys())), "incorrect list of args"
    
        assert(kwargs["run_type"]=="r_protrusion_idx"), "run type does not match argument passed"
        assert(os.path.exists(kwargs["tbl_filename"])), "tbl filename does not exist"
    
        filename, ext = kwargs["tbl_filename"].split("/")[-1].split(".")
        assert(ext=="tbl"), "file extension must be .tbl"
        complex_code = filename[:4]
    
        logging.info("|| complex_code: {} "
                     "|| tbl_filename: {}".format(complex_code,
                                                  kwargs["tbl_filename"]))
    
        psaia_output_records = _tbl_file_reader(path_to_tbl_file=kwargs["tbl_filename"],
                                                complex_code=complex_code)
        final_format_protrusion_idxs = _create_final_format(psaia_output_records=psaia_output_records,
                                                            complex_code=complex_code) 
    
        paths = prjPaths(run_type=kwargs["run_type"])
        to_pickle(obj_to_pickle=final_format_protrusion_idxs,
                  path=os.path.join(paths.RUN_TYPE_OUT_DIR,
                                    "r_protrusion_idx_{}.p".format(complex_code)))
        end_time = timer()
        run_time = end_time-start_time
        logging.info("|| complex_code: {} "
                     "|| protrusion_idx and hydrophobicity shape: ({}, {})".format(complex_code,
                                                                                   len(final_format_protrusion_idxs),
                                                                                   len(final_format_protrusion_idxs[list(final_format_protrusion_idxs.keys())[0]])))
        logging.info("|| complex_code: {} "
                     "|| run time (in seconds): {}".format(complex_code,
                                                           run_time))
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
