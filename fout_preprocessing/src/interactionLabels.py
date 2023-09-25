"""
created on Nov 14th 2018

@author: danielburkat
"""

from rdkit import Chem
from Bio.PDB.PDBParser import PDBParser
import os
import numpy as np
import pandas as pd
import pickle
from timeit import default_timer as timer
import logging
import traceback
from utils import *


__all__ = [
        "interaction_labels_main"        
]


def _get_atom_coords(mol):
    N = mol.GetNumAtoms()

    atoms = mol.GetAtoms()
    coords = np.zeros((N,3))
    for atom in atoms:
        idx = atom.GetIdx()
        pos = mol.GetConformer(0).GetAtomPosition(idx)
        coords[idx,:] = [pos.x, pos.y, pos.z]
    return coords


def interaction_labels_main(kwargs):

    INTERACTION_DISTANCE = 6

    start_time = timer()
    try:
        
        assert(all(k in ["run_type", "protein_filepath", "ligand_file_mol2"] 
            for k in kwargs.keys())), "incorrect list of args"
        assert(kwargs["run_type"]=="interaction_labels"), "run type does not match arguments passed"
        assert(os.path.exists(kwargs["protein_filepath"])), "ent filename does not exist"

        # build the absolute filepaths to the other files of the complex code
        # the protein pdb file
        protein_fp = kwargs['protein_filepath']
        filename = protein_fp.split("/")[-1]
        complex_code = filename[:4]

        # from that get the folder that is one up
        dataset_directory = os.path.join(*protein_fp.split("/")[:-2])
        if dataset_directory[:4] == 'home':
            dataset_directory = os.sep + dataset_directory
        # make the pocket file path
        pocket_fp = os.path.join(dataset_directory, "pocket_pdb", complex_code+"_pocket.pdb") 
        # make the ligand filepath
        ligand_file_mol2 = kwargs['ligand_file_mol2']
        if ligand_file_mol2:
            ligand_ext = 'mol2'
        else:
            ligand_ext = 'sdf'

        ligand_fp = os.path.join(dataset_directory, "ligand_"+ligand_ext, complex_code+"_ligand."+ligand_ext)

        assert(os.path.exists(pocket_fp)), "*_pocket.pdb filename does not exists"
        assert(os.path.exists(ligand_fp)), "ligand filename does not exists"


        paths = prjPaths(run_type=kwargs["run_type"])
        
        # Load the molecule
        if ligand_ext == 'sdf':
            # Not using sdf now so pass
            pass
        elif ligand_ext == 'mol2':
            mol = Chem.MolFromMol2File(ligand_fp)
        # Get the atoms coordinates in the molecule
        coords = _get_atom_coords(mol)

            
        # Load the residues of the protein
        # consider only the first model
        parser = PDBParser()
        struct = parser.get_structure(complex_code, protein_fp)
        m = struct.get_list()[0]
    
        # First build a dictionary with all the possible combinations of residue to atom
        # initialized to 0 (not interacting)
        labels = {}
        res_count = 0
        for chain in m:
            chain_id = "{2}_{0}".format(*list(map(str, chain.full_id)))
            for i_res, residue in enumerate(chain.get_residues()):
                if residue.id[0] is not " ":
                    continue
                res_id = "{}_{}_{}".format(residue.id[1], i_res, residue.get_resname())
                res_count += 1
                for coord_idx, _ in enumerate(coords):
                    labels[(coord_idx, (chain_id, res_id))] = 0
       

        # Now visit those residues that are in the pocket
        struc = parser.get_structure(complex_code, pocket_fp)
        m = struct.get_list()[0]
        pocket_atom_coords = []
        pocket_atom_id = []
        for chain in m:
            chain_id = "{2}_{0}".format(*list(map(str, chain.full_id)))
            for i_res, residue in enumerate(chain.get_residues()):
                if residue.id[0] is not " ":
                    continue
                res_id = "{}_{}_{}".format(residue.id[1], i_res, residue.get_resname())
                # put all the atoms into one list
                for atom in residue:
                    pocket_atom_coords.append(atom.get_vector().get_array())
                    pocket_atom_id.append((chain_id, res_id))
        
        
        # Figure out if there is interaction
        interact_count = 0

        p_coords = np.stack(pocket_atom_coords, axis=0)
        for coord_idx, _ in enumerate(coords):
            dist = np.sqrt(np.sum(np.square(p_coords - coords[coord_idx,:]), axis=1))
            mask = dist < INTERACTION_DISTANCE
            interact_idx = np.argwhere(mask)    
            interact_residue = set()
            for each in interact_idx:
                interact_residue.add(pocket_atom_id[each[0]])
        
            # Now update labels with a 1
            for each in interact_residue:
                labels[(coord_idx, each)] = 1
                interact_count += 1


        to_pickle(obj_to_pickle=labels,
                    path=os.path.join(paths.RUN_TYPE_OUT_DIR,
                    "labels_{}.p".format(complex_code)))

        end_time = timer()
        run_time = end_time-start_time
        logging.info("|| complex code: {} "
                    "|| # atoms: {} "
                    "|| # residues: {} "
                    "|| # labels: {} "
                    "|| # 1: {}".format(complex_code,
                                        coords.shape[0],
                                        res_count,
                                        len(labels),
                                        interact_count))

        logging.info("|| complex_code: {} "
                    "|| run time: {} s".format(complex_code, run_time))
        
        return_msg = {complex_code: run_time}

    except Exception as err:
        logging.error("|| filename:{} "
                    "|| error thrown: {}".format(kwargs["protein_filepath"], traceback.format_exc()))
        end_time = timer()
        run_time = end_time-start_time
        return_msg = {kwargs['protein_filepath']: run_time}
        pass
    finally:
        return return_msg

# end interaction_labels_main(...)



















