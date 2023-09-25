"""
@author: Daniel Burkat

"""
from math import pi
import numpy as np

from Bio.PDB import *
from Bio.PDB.AbstractPropertyMap import AbstractPropertyMap
from Bio.PDB.Polypeptide import CaPPBuilder, is_aa
from Bio.PDB.vectors import rotaxis

from Bio.PDB.HSExposure import HSExposureCB
import csv

from timeit import default_timer as timer
import logging
import traceback
from utils import *
import os

from tqdm import tqdm

__author__ = "Daniel Burkat"
__version__ = "0.1"

__all__ = [
        "half_sphere_amino_acid_comp_main"
]


class HSAAC(AbstractPropertyMap):
    """Class to calculate HSAAC

    The calculation results are stored in the HSAAC object
    """

    def __init__(self, model, radius, offset, structure_id):
        """
        Based on the _AbstractHSExposure() class in Bio.PDB.HSExposure.py
        """
        assert(offset >= 0)
        ppb = CaPPBuilder()
        ppl = ppb.build_peptides(model)
        hsaac_map = {}
        hsaac_list = []
        hsaac_keys = []
        for idx, pp1 in enumerate(tqdm(ppl, "evaluating complex: {}".format(structure_id))):
            for i in range(0, len(pp1)):
                r2 = pp1[i]
                result = self._get_cb(r2)
                if result is None:
                    # Missing atoms
                    continue
                pcb, angle = result
                hsc_u = self._get_aa_dict()
                hsc_d = self._get_aa_dict()
                ca2 = r2['CA'].get_vector()
                for pp2 in ppl:
                    for j in range(0, len(pp2)):
                        if pp1 is pp2 and abs(i-j) <= offset:
                            # neighboring residues in the chain are ignored
                            continue
                        ro = pp2[j]
                        if not is_aa(ro) or not ro.has_id('CA'):
                            continue
                        cao = ro['CA'].get_vector()
                        d = (cao - ca2)
                        if self._in_radius_condition(r2, ro, radius):
                            if d.angle(pcb) < (pi / 2):
                                hsc_u[ro.get_resname()] += 1
                            else:
                                hsc_d[ro.get_resname()] += 1
                # assemble a chain id and res id
                chain_id = "{2}_{0}".format(*list(map(str, r2.get_parent().full_id)))
                res_id = "{}_{}_{}".format(r2.id[1], i, r2.get_resname())
                
                hsc_u_list = self._hsc_to_norm_list(hsc_u)
                hsc_d_list = self._hsc_to_norm_list(hsc_d)

                if (chain_id, res_id) in hsaac_map.keys():
                    logging.error("|| complex_code: {} "
                                  "|| hsaac_map already contains key {}".format(structure_id,
                                                                                list(map(str, (chain_id, res_id)))))

                hsaac_map[(chain_id, res_id)] = (hsc_u_list, hsc_d_list, angle)
                hsaac_list.append((res_id, hsc_u_list, hsc_d_list, angle))
                hsaac_keys.append((chain_id, res_id))
        AbstractPropertyMap.__init__(self, hsaac_map, hsaac_keys, hsaac_list)
    # end

    
    def _in_radius_condition(self, ra, rb, radius):
        for atm_i in ra:
            for atm_j in rb:
                d = atm_j.get_vector() - atm_i.get_vector()
                if d.norm() < radius:
                    return True
        return False

    
    def _get_cb(self, r2):
        """Calculate CB-CA vector. (as Thomas Hamelryck did it in HSExposure.py)
        args:
        r2 -- a residue
        
        return:
        the CB - CA vector. None if couldn't calculate it
        """
        
        if r2.get_resname() == 'GLY':
            return self._get_gly_cb_vector(r2), 0.0
        else:
            if r2.has_id('CB') and r2.has_id('CA'):
                vcb = r2['CB'].get_vector()
                vca = r2['CA'].get_vector()
                return (vcb - vca), 0.0
            return None
    # end


    def _get_gly_cb_vector(self, residue):
        """Return a pseudo CB vecotr for Gly residues. 
        (as Thomas Hamelryck did it in HSExposure.py)
        
        args:
        residue -- a residue

        return:
        the pseudo CB vector
        """

        try:
            n_v = residue["N"].get_vector()
            c_v = residue["C"].get_vector()
            ca_v = residue["CA"].get_vector()
        except Exception:
            return None
        # center at origin
        n_v = n_v - ca_v
        c_v = c_v - ca_v
        rot = rotaxis(-pi * 120.0/180.0, c_v)
        cb_at_origin_v = n_v.left_multiply(rot)
        # move back to ca position
        cb_v = cb_at_origin_v + ca_v
        # Note, did din't add the thing for PyMol visualization as in the original code
        return cb_at_origin_v
    # end

    
    def _get_aa_dict(self):
        """Function that returns an initialized dict to count up the aa's
        """
        hsc = {
            "ALA":0, "ARG":0, "ASN":0, "ASP":0, "CYS":0, "GLU":0, "GLN":0, 
            "GLY":0, "HIS":0, "ILE":0, "LEU":0, "LYS":0, "MET":0, "PHE":0,
            "PRO":0, "SER":0, "THR":0, "TRP":0, "TYR":0, "VAL":0
        }
        return hsc


    def _hsc_to_norm_list(self, hsc_dict):
        """Take the dictionary and get it into a list format"""
        hsc_list = []
        aas = [
            "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
            "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"
        ]

        for aa in aas:
            hsc_list.append(hsc_dict[aa])
        
        hsc_a = np.array(hsc_list)
        tot = hsc_a.sum()
        if tot != 0:
            hsc_b = hsc_a / tot
            hsc_list = hsc_b.tolist()

        return hsc_list



def hsaac_on_fp(abs_fp_entry, structure_id, chains=None):
    """
    desc: calculate the HSAAC as described by A. Fout
    
    args:
    abs_fp_entry -- Needs to be to full absolute path to the pdb like entry file 
    structure_id -- add the 4 letter code for the entry
    chains -- To restrict to only certain chains, then supply the a list of chain id (strings)

    return: an object that hold the properties.
    
    """
    
    #Assume that we are given the absolute file path to an entry
    
    

    parser = PDBParser(PERMISSIVE=1)
    struct = parser.get_structure(structure_id, abs_fp_entry)
    
    # Get the first model
    m_list = struct.get_list()
    
    """The code below will check if there are more than one model. However, comented it out for now""" 
    # TODO determine if we need to evaluate multiple models...if there are multiple models in a pdbbind protein file
    #if len(m_list) == 1:
        # Get the first model
    #    m = m_list[0]
    #elif len(m_lisr) == 0:
    #    logger.info("Model length ZERO for structure_id: {}".format(structure_id))
        #Nothing to do so return
    #    return None
    #else:
    #    logger.info("Model length GREATER than 1 for structure_id: {} | length: {}".format(structure_id, len(m_list) ))
        # For now pick the first model. 
    #    m = m_list[0]

    #Select the first model
    m = m_list[0]

    # Keep only the chains in the list Chains
    if chains is not None:
        c = list(m.child_dict.keys())
        c_to_detach = [x for x in c if x not in chains]
        for each in c_to_detach:
            m.detach_child(each)

    hsc = HSAAC(m, radius=8.0, offset=0, structure_id=structure_id)
    
    return hsc
# end


"""Below was used to develop the function"""
def half_sphere_amino_acid_comp_main(kwargs):
    start_time = timer()
    try:
        assert(all(k in ["run_type", "ent_filename"]
                        for k in kwargs.keys())), "incorrect list of args"
        assert(kwargs["run_type"]=="r_hsaac"), "run type does not match argument passed"
    
    
        paths = prjPaths(run_type=kwargs["run_type"])
        filename, ext = kwargs["ent_filename"].split("/")[-1].split(".")
        if ext == "ent":
            complex_code = filename[3:7]
        elif ext == "bio1" or ext == "pdb":
            complex_code = filename[:4]
    
        output = hsaac_on_fp(kwargs["ent_filename"], complex_code) 
        to_pickle(obj_to_pickle=output.property_dict,
                  path=os.path.join(paths.RUN_TYPE_OUT_DIR,
                                    "r_hsaac_{}.p".format(complex_code)))
        end_time = timer()
        run_time = end_time-start_time
        logging.info("|| complex_code: {} "
                     "|| hsaac shape: ({},{},{})".format(complex_code,
                                                         len(output.property_dict),
                                                         len(output.property_dict[list(output.property_dict.keys())[0]]),
                                                         len(output.property_dict[list(output.property_dict.keys())[0]][0])))
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
