"""
created on Nov 7th 2018

@author: danielburkat
"""

from rdkit import Chem
from rdkit.Chem import rdPartialCharges
import os
import numpy as np
import pickle
from timeit import default_timer as timer
import logging
import traceback
from utils import *

__author__ = "Daniel Burkat"
__version__ = "0.1"
__all__ = [
        "molecule_features_main"        
]


"""Function that calculates a single atom's features"""
def _atom_features(atom):

    atom_feats = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe',
        'As','Al','I','B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se',
        'Ti','Zn','H','Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr',
        'Pt','Hg','Pb','Unknown']
        ) + one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,10]) + \
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6]) + \
        [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
        one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]) + [atom.GetIsAromatic()*1] + \
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4]) + \
        [atom.GetProp("_GasteigerCharge")]


    return np.array(atom_feats)


"""Helper function to calculae the bond_features"""
def _bond_features(bond):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats)



"""Calculates all the edge features
Makes use of the helper funciton bond_features()
"""
def _get_edge_features(mol, hood_len):
    N = mol.GetNumAtoms()
    
    atoms = mol.GetAtoms()
    coords = np.zeros((N,3))
    for atom in atoms:
        idx = atom.GetIdx()
        pos = mol.GetConformer(0).GetAtomPosition(idx)
        coords[idx,:] = [pos.x, pos.y, pos.z]
    dist = np.sqrt(np.sum(np.square( np.stack([coords]*N, axis=1) - np.stack([coords]*N, axis=0)), axis=2))
    
    # for each atom get the 'hood_len' closest atoms
    hood = np.zeros((N, hood_len), dtype=int)
    for i in range(N):
        closest = np.argsort(dist[i])
        hood[i,:] = closest[1:hood_len+1].copy()
    
    # For each edge get the bond features
    NUM_BOND_FEATURES = 7
    edge_feats = np.zeros((N, hood_len, NUM_BOND_FEATURES))
    for i in range(N):
        closest = hood[i,:]
        for pos, j in enumerate(closest):
            edge_feats[i,pos,0] = dist[i,j]
            bt = mol.GetBondBetweenAtoms(i,int(j))
            if bt is not None:
                bond_feats = _bond_features(bt)
                edge_feats[i,pos,1:] = bond_feats.copy()

    return hood, edge_feats, coords



def molecule_features_main(kwargs):
        
    # This variable determines how many atoms are considered in the neighbourhood
    HOOD_LEN = 10 

    start_time = timer()
    try:

        assert(all(k in ["run_type", "filepath"] for k in kwargs.keys())), "incorrect list of args"
        assert(kwargs["run_type"]=="l_molecule_features"), "run type does not match argument passed"
        assert(os.path.exists(kwargs["filepath"])), "ent filename does not exist"

        start_time = timer()

        paths = prjPaths(run_type=kwargs["run_type"])

        # Get the complex code and the extention
        filename, ext = kwargs["filepath"].split("/")[-1].split(".")
        complex_code = filename[:4]

        if ext == "sdf":
            """Use the block below with sdf files.
            NOTE: This code has been tested on .mol2 files.
            Need to confirm a few thing on how it works with sdf before using it."""
            # make sure there is only one molecule in the file
            #suppl = Chem.SDMolSupplier(filepath)
            #if sum(1 for _ in suppl) != 1:
                #TODO: Log this error.
            #    return None
            # IF it gets to this point then we know there is only one mol
            #suppl = Chem.SDMolSupplier(filepath)
            #mol = next(suppl)
            
            # For now pass this since we don't use sdf yet.
            pass
        elif ext == "mol2":
            """Use the block below for mol2 files"""
            #TODO: Ensure that we catch when the mol2 file can't be loaded properly
            # I observed that it does that for ~ 600 files.
            # Also need to ensure that the min number of atoms is greater of 11
            mol = Chem.MolFromMol2File(kwargs["filepath"])



        rdPartialCharges.ComputeGasteigerCharges(mol)
        
        # Calculate the atoms features
        atoms = mol.GetAtoms()        
        a_feat = {}

        # The below might seem wierd, but this is done to ensure we are as certain as possible
        # that each row of atom_feats correspond to the GetIdx() value of an atom
        for a in atoms:
            #build a dict with these features where index is the GetIdx value
            a_feat[a.GetIdx()] = _atom_features(a)
        atom_feats = np.stack([a_feat[i] for i in range(len(atoms))])

        # Calculate the edge features and additional info
        # NOTE: the variable hood_len is set at the beggining of the funciton. It was
        # chosen to be 10. 
        hood_index, edge_feats, coords = _get_edge_features(mol=mol, hood_len=HOOD_LEN) 
        
        # Pack it into a tuple and persist it to file
        feats = (atom_feats, hood_index, edge_feats, coords)
        
        to_pickle(obj_to_pickle=feats, path=os.path.join(paths.RUN_TYPE_OUT_DIR,
            'l_molecule_features_{}.p'.format(complex_code)))

        end_time = timer()
        run_time = end_time - start_time
        logging.info("|| complex_code: {} "
                "|| atom_feats shape: ({}, {}) "
                "|| hood_index: ({}, {}) "
                "|| edge_feats: ({}, {}, {}) "
                "|| coords: ({}, {})".format(complex_code,
                                        atom_feats.shape[0],
                                        atom_feats.shape[1],
                                        hood_index.shape[0],
                                        hood_index.shape[1],
                                        edge_feats.shape[0],
                                        edge_feats.shape[1],
                                        edge_feats.shape[2],
                                        coords.shape[0],
                                        coords.shape[1]))
                

        logging.info("|| complex_code: {} || run time: {} s".format(complex_code, run_time))

        return_msg = {complex_code: run_time}

    except Exception as err:
        logging.error("|| filename: {} "
                    "|| error thrown: {}".format(kwargs['filepath'], traceback.format_exc()))
        end_time = timer()
        run_time = end_time-start_time
        return_msg = {kwargs["filepath"]: run_time}
        pass
    finally:
        return return_msg
# end



""" one_of_k_endocoding() was taken from neuralfingerprint/util.py """
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set {1}:".format(x, allowable_set))
    #return list(map(lambda s: x == s, allowable_set))
    return list(map(lambda s: 1 if x == s else 0, allowable_set))

""" one_of_k_encoding_unk() was taken from neuralfinderprint/util.py """
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    #return list(map(lambda s: x == s, allowable_set))
    return list(map(lambda s: 1 if x == s else 0, allowable_set))




#if __name__ == "__main__":
#    """Example of how you would run it on one '*_ligand.mol2' file"""
#    data_fp = "/home/mg/pdbb/pdbbind_v2018_refined/refined-set"
#    pdb_entry =  "4ufl"
#    mol2_ext = "_ligand.mol2"
#    mol2_fp = os.path.join(data_fp, pdb_entry, pdb_entry+mol2_ext)

    """ 'feats' is a tuple of 4 numpy arrays.
    can unpack the tuple in the following way:
    atom_feats, hood, edge_feats, coords = feats
    """
    
#    kw = {
#        "run_type":"molecule_features",
#        "filepath":mol2_fp
#    }
#    rt = molecule_features_main(kw)
#    print(rt)

    #with open('mol_feats_{}.pkl'.format(pdb_entry), 'wb') as f:
    #    pickle.dump(feats,f)
    




