"""
@author: Michael Guarino
desc: this script is used to support download.py
"""

from Bio.PDB import *
import os
import datetime
import logging
import pickle
import multiprocessing
from logging.handlers import QueueHandler, QueueListener
import numpy as np

__author__ = "Michael Guarino"
__version__ = "0.1"
__all__ = [
    "prjPaths",
    "logger_init",
    "to_pickle",
    "from_pickle",
    "load_structure",
    "max_accessilbe_surface_area_by_aa",
    "remove_from_fs",
    "one_letter_amino_acid_codes"
]

class prjPaths:
  def __init__(self, run_type):
    """
    desc: create object contraining project paths
    args:
    returns:
    """

    assert(run_type in ["r_edge",
                        "r_stride",
                        "r_hsaac",
                        "r_residueDepth",
                        "r_pssm",
                        "r_protrusion_idx",
                        "l_molecule_features",
                        "interaction_labels",
                        "merge"]), "invalid run type"

    self.SRC_DIR = os.path.abspath(os.path.curdir)
    self.ROOT_DIR = "/".join(self.SRC_DIR.split("/")[:-1])
    self.LIB_DIR = os.path.join(self.ROOT_DIR, "lib")
    self.OUT_DIR = os.path.join(self.ROOT_DIR, "output")
    self.STRIDE_DIR = os.path.join(self.ROOT_DIR, "stride")
    self.LOGS_DIR = os.path.join(self.LIB_DIR, "logs")
    self.RUN_TYPE_OUT_DIR = os.path.join(self.OUT_DIR, run_type)

    pth_exists_else_mk = lambda path: os.mkdir(path) if not os.path.exists(path) else None

    pth_exists_else_mk(self.OUT_DIR)
    pth_exists_else_mk(self.LOGS_DIR)
    pth_exists_else_mk(self.RUN_TYPE_OUT_DIR)

  # end
# end

def logger_init(path, filename):
    """
    desc: create logger 
    args:
    returns:
    """
    queue = multiprocessing.Queue()
    # handler for all log records
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(process)s - %(message)s"))
  
    # queue_listener gets records from the queue and sends them to the handler
    queue_listener = QueueListener(queue, handler)
    queue_listener.start()
  
    # currentTime = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logFileName = os.path.join(path, "{}.log".format(filename))
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=logFileName,
                        filemode='w')
  
    logger = logging.getLogger()
  
    logger.setLevel(logging.INFO)
    # add the handler to the logger so records from this process are handled
    logger.addHandler(handler)
  
    return logger, queue_listener, queue
# end

def to_pickle(obj_to_pickle, path, protocol_type=None):
    """
    desc:
    args:
    returns:
    """
    if protocol_type==2:
        pickle.dump(obj_to_pickle, open(path, "wb"), protocol=protocol_type)
    else:
        # TODO assert that extension is .p
        pickle.dump(obj_to_pickle, open(path, "wb"))
# end

def from_pickle(path_to_pickle):
   """
   desc:
   args:
   returns:
   """
   assert(os.path.exists(path_to_pickle)), "path to pickle file {} does not exist".format(path_to_pickle)
   return pickle.load(open(path_to_pickle, "rb"))
# end

def load_structure(complex_code, ent_filename):
    """
    desc:
    args:
    returns:
    """
    parser=PDBParser()
    structure=parser.get_structure(complex_code, ent_filename)
    return structure
# end

def remove_from_fs(path_to_entity):
    """
    desc:
    args:
    returns:
    """
    if os.path.isfile(path_to_entity):
        os.remove(path_to_entity)
    elif os.path.isdir(path_to_entity):
        shutil.rmtree(path_to_entity) 
    assert(not os.path.exists(path_to_entity)), "file system entity could not be removed"
# end

max_accessilbe_surface_area_by_aa = {
    "ala":{"Theoretical":129,"Empirical":121,"Miller":113,"Rose":118.1},
    "arg":{"Theoretical":274,"Empirical":265,"Miller":241,"Rose":256},
    "asn":{"Theoretical":195,"Empirical":187,"Miller":158,"Rose":165.5},
    "asp":{"Theoretical":193,"Empirical":187,"Miller":151,"Rose":158.7},
    "cys":{"Theoretical":167,"Empirical":148,"Miller":140,"Rose":146.1},
    "glu":{"Theoretical":223,"Empirical":214,"Miller":183,"Rose":186.2},
    "gln":{"Theoretical":225,"Empirical":214,"Miller":189,"Rose":193.2},
    "gly":{"Theoretical":104,"Empirical":97,"Miller":85,"Rose":88.1},
    "his":{"Theoretical":224,"Empirical":216,"Miller":194,"Rose":202.5},
    "ile":{"Theoretical":197,"Empirical":195,"Miller":182,"Rose":181},
    "leu":{"Theoretical":201,"Empirical":191,"Miller":180,"Rose":193.1},
    "lys":{"Theoretical":236,"Empirical":230,"Miller":211,"Rose":225.8},
    "met":{"Theoretical":224,"Empirical":203,"Miller":204,"Rose":203.4},
    "phe":{"Theoretical":240,"Empirical":228,"Miller":218,"Rose":222.8},
    "pro":{"Theoretical":159,"Empirical":154,"Miller":143,"Rose":146.8},
    "ser":{"Theoretical":155,"Empirical":143,"Miller":122,"Rose":129.8},
    "thr":{"Theoretical":172,"Empirical":163,"Miller":146,"Rose":152.5},
    "trp":{"Theoretical":285,"Empirical":264,"Miller":259,"Rose":266.3},
    "tyr":{"Theoretical":263,"Empirical":255,"Miller":229,"Rose":236.8},
    "val":{"Theoretical":174,"Empirical":165,"Miller":160,"Rose":164.5}
}

one_letter_amino_acid_codes = {
    "ALA": "A", # Alanine
    "ARG": "R", # Arginine 
    "ASN": "N", # Asparagine 
    "ASP": "D", # Aspartic Acid 
    "CYS": "C", # Cysteine 
    "GLU": "E", # Glutamic Acid
    "GLN": "Q", # Glutamine 
    "GLY": "G", # Glycine 
    "HIS": "H", # Histidine 
    "ILE": "I", # Isoleucine 
    "LEU": "L", # Leucine 
    "LYS": "K", # Lysine 
    "MET": "M", # Methionine 
    "PHE": "F", # Phenylalanine 
    "PRO": "P", # Proline 
    "SER": "S", # Serine 
    "THR": "T", # Threonine 
    "TRP": "W", # Tryptophan 
    "TYR": "Y", # Tyrosine 
    "VAL": "V",  # Valine 
}
