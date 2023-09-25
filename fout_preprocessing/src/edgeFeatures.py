"""
@author: mguarin0
desc: calculates edge features and hood indices for receptor protein
      and writes out to a pickle file
"""

from Bio.PDB import *
import os
from scipy.spatial import distance
import numpy as np
from operator import itemgetter
from math import pi
from timeit import default_timer as timer
import traceback
import logging

from utils import *

__author__ = "Michael Guarino"
__version__ = "0.1"
__all__ = [
        "edge_features_main"
]

"""Average Atomic Distance Calculation"""
def _cal_avg_atomic_position(all_atoms_coords):
    """
    desc:
    args:
    returns:
    """
    x = 0
    y = 0
    z = 0
    for elm in all_atoms_coords:
        x += elm["x"]
        y += elm["y"]
        z += elm["z"]
    return (x/len(all_atoms_coords),
            y/len(all_atoms_coords),
            z/len(all_atoms_coords))
# end

"""Euclidean Distance btw Residues"""
def _euclidean_distance(res_id_c, res_id_o): return distance.euclidean(res_id_c, res_id_o)

"""Apply Gaussian like function to Euclidean Distance"""
def _fout_gaussian(x, std_dev): return np.exp(-x**2/std_dev**2)

"""Calculate Norm"""
def _norm(xo, xc, xca):
    """
    desc:
    args:
    returns:
    """
    oc_v = xo-xc
    cac_v = xca-xc
    cp = np.cross(oc_v, cac_v) 
    l2_norm = np.linalg.norm(cp)
    return cp/l2_norm
# end

"""Calculate Angle"""
def _arc_cos(ni, nj, normalize=False):
    """
    desc:
    args:
    returns:
    """
    num = np.dot(ni, nj)
    dem = np.linalg.norm(ni)*np.linalg.norm(nj)
    if normalize:
        agl = np.arccos((num/dem)/(2*pi))
    else:
        agl = np.arccos(num/dem)
    return agl
# end

"""Distance to other Residues calculation"""
def _cal_dst_agl_to_other_residues(dst_agl, closest_n, std_dev):
    """
    desc:
    args:
    returns:
    """
    all_residue_ec_dst_agl = {}
    all_residue_hood_indices = {}
    for i_k, c in dst_agl.items():
        chain_id, res_id = i_k
        residue_ec_dst_agl = []
        ni = _norm(xo=c["O"],
                   xc=c["C"],
                   xca=c["CA"])
        for j_k, o in dst_agl.items():
            nj = _norm(xo=o["O"],
                       xc=o["C"],
                       xca=o["CA"])
            residue_ec_dst_agl.append([_euclidean_distance(c["avg_atomic_dst"],
                                                           o["avg_atomic_dst"]),
                                       _arc_cos(ni=ni, nj=nj),
                                       j_k])
        residue_ec_dst_agl.sort(key=itemgetter(0)) # sort in place

        if (chain_id, res_id) in all_residue_ec_dst_agl.keys():
            logging.error("|| complex_code: {} "
                          "|| all_residue_ec_dst_agl already contains key: {}".format(complex_code,
                                                                                      list(map(str, (chain_id, res_id)))))
        if (chain_id, res_id) in all_residue_hood_indices.keys():
            logging.error("|| complex_code: {} "
                          "|| all_residue_hood_indices already contains key: {}".format(complex_code,
                                                                                        list(map(str, (chain_id, res_id)))))
        all_residue_ec_dst_agl[(chain_id, res_id)] = [[_fout_gaussian(dst, std_dev), agl]
                                                      for dst, agl, j_k in residue_ec_dst_agl[1:closest_n+1]]
        all_residue_hood_indices[(chain_id, res_id)] = [[j_k]
                                                        for _0, _1, j_k in residue_ec_dst_agl[1:closest_n+1]]
    return (all_residue_ec_dst_agl, all_residue_hood_indices)
# end

"""Gather all Feature Info """
def edge_features_main(kwargs):
    """
    desc:
    args:
    returns:
    """
    start_time = timer()
    try:
        assert(all(k in ["run_type", "ent_filename", "closest_n", "std_dev", "norm_agl"]
                        for k in kwargs.keys())), "incorrect list of args"
        assert(kwargs["run_type"]=="r_edge"), "run type does not match argument passed"
        assert(os.path.exists(kwargs["ent_filename"])), "ent filename does not exist"

        paths = prjPaths(run_type=kwargs["run_type"])

        filename, ext = kwargs["ent_filename"].split("/")[-1].split(".")
        if ext == "ent":
            complex_code = filename[3:7]
        elif ext in ["bio1", "pdb"]:
            complex_code = filename[:4]

        structure = load_structure(complex_code, kwargs["ent_filename"])
        ### assemble info for every residue
        dst_agl = {}
        for model in structure:
            for chain in model:
                chain_id = "{2}_{0}".format(*list(map(str, chain.full_id)))
                for i_res, residue in enumerate(chain.get_residues()):
                    if residue.id[0] is not " ":
                        continue
                    res_id = "{}_{}_{}".format(residue.id[1], i_res, residue.get_resname())
                    all_atoms_coords = []
                    for atom in residue:
                        all_atoms_coords.append({"x": atom.coord[0],
                                                 "y": atom.coord[1],
                                                 "z": atom.coord[2]})

                    if (chain_id, res_id) in dst_agl.keys():
                        logging.error("|| complex_code: {} "
                                      "|| dst_agl already contains key: {}".format(complex_code,
                                                                                   list(map(str, (chain_id, res_id)))))
                    dst_agl[(chain_id, res_id)] = {"O": residue["O"].coord,
                                                   "C": residue["C"].coord,
                                                   "CA": residue["CA"].coord,
                                                   "avg_atomic_dst": _cal_avg_atomic_position(all_atoms_coords)}
        all_res_dst_agl, all_res_hood_indices = _cal_dst_agl_to_other_residues(dst_agl=dst_agl,
                                                                               closest_n=kwargs["closest_n"],
                                                                               std_dev=kwargs["std_dev"])
        to_pickle(obj_to_pickle=all_res_dst_agl,
                  path=os.path.join(paths.RUN_TYPE_OUT_DIR,
                                    "r_edge_{}.p".format(complex_code)))
        to_pickle(obj_to_pickle=all_res_hood_indices,
                  path=os.path.join(paths.RUN_TYPE_OUT_DIR,
                                    "r_hood_indices_{}.p".format(complex_code)))
        end_time = timer()
        run_time = end_time-start_time
        logging.info("|| complex_code: {} "
                     "|| r_edge shape: ({}, {}, {}) "
                     "|| r_hood_indices shape: ({}, {}, {})".format(complex_code,
                                                              len(all_res_dst_agl),
                                                              len(all_res_dst_agl[list(all_res_dst_agl.keys())[0]]),
                                                              len(all_res_dst_agl[list(all_res_dst_agl.keys())[0]][0]),
                                                              len(all_res_hood_indices),
                                                              len(all_res_hood_indices[list(all_res_hood_indices.keys())[0]]),
                                                              len(all_res_hood_indices[list(all_res_hood_indices.keys())[0]][0])))
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
