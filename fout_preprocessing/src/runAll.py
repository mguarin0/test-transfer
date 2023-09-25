"""
@author: mguarin0
desc:
"""

import argparse
import os
from subprocess import Popen
from multiprocessing import Pool
import logging
from logging.handlers import QueueHandler
import traceback

from utils import *

__author__ = "Michael Guarino and Daniel Burkat"
__version__ = "0.1"

def get_args():
    """
    desc: get cli arguments
    returns:
    args: dictionary of cli arguments
    """
    parser = argparse.ArgumentParser(description="this script calculates edge features and hood indices for receptor protein")
    parser.add_argument("run_type", choices=["r_edge",
                                             "r_stride",
                                             "r_residueDepth",
                                             "r_protrusion_idx",
                                             "r_hsaac",
                                             "r_pssm_step_one",
                                             "r_pssm_step_two",
                                             "l_molecule_features",
                                             "interaction_labels"], help="which feature to calculate", type=str)
    parser.add_argument("full_path_to_data_dir", help="full path to directory containing dataset", type=str)
    parser.add_argument("full_path_to_stride_exec", help="full path to stride executable", type=str)
    parser.add_argument("full_path_to_msms_exec", help="full path to msms executable", type=str)
    parser.add_argument("full_path_to_psiblast_exec", help="full path to psi-blast executable", type=str)
    parser.add_argument("--r_edge_closest_n", default=20, help="edge feature calculation closest n residues to consider", type=int)
    parser.add_argument("--r_edge_std_dev", default=18, help="edge feature calculation standard deviation for gaussian", type=int)
    parser.add_argument("--r_edge_norm_agl", default=18, help="edge feature calculation normalize angle in radians", type=int)
    parser.add_argument("--process_in_parallel", default=True, help="process entry files in parallel", type=bool)
    parser.add_argument("--ligand_file_mol2", default=True, help="Default to mol2. Set False to read the .sdf files", type=bool)
    args = parser.parse_args()
    return args
# end

def run_cmd_router(kwargs):
    """
    desc: given a dictionary of cli args will create a list of commands to be run
          for a given run_type. This list of commands is all commands needed
          to perform all calculations for every entry file in defined
          full_path_to_data_dir.
    args:
        kwargs: dictionary of cli args
    returns:
        cmds: list of args to send to defined run_cmd
    """
    if kwargs["run_type"]=="r_edge":
        cmds = [{"run_type": kwargs["run_type"],
                 "ent_filename": os.path.join(kwargs["full_path_to_data_dir"],
                                              ent_filename),
                 "closest_n": kwargs["r_edge_closest_n"],
                 "std_dev": kwargs["r_edge_std_dev"],
                 "norm_agl": kwargs["r_edge_norm_agl"]}
                    for ent_filename in os.listdir(kwargs["full_path_to_data_dir"])
                        if ent_filename.split("/")[-1].split(".")[1] in ["ent", "bio1", "pdb"]]

    if kwargs["run_type"]=="r_stride":
        cmds = [{"run_type": kwargs["run_type"],
                 "ent_filename": os.path.join(kwargs["full_path_to_data_dir"],
                                              ent_filename),
                 "stride_path": kwargs["full_path_to_stride_exec"]}
                    for ent_filename in os.listdir(kwargs["full_path_to_data_dir"])
                        if ent_filename.split("/")[-1].split(".")[1] in ["ent", "bio1", "pdb"]]

    if kwargs["run_type"]=="r_residueDepth":
        cmds = [{"run_type": kwargs["run_type"],
                 "ent_filename": os.path.join(kwargs["full_path_to_data_dir"],
                                              ent_filename),
                 "msms_path": kwargs["full_path_to_msms_exec"]}
                    for ent_filename in os.listdir(kwargs["full_path_to_data_dir"])
                        if ent_filename.split("/")[-1].split(".")[1] in ["ent", "bio1", "pdb"]]

    if kwargs["run_type"]=="r_protrusion_idx":
        cmds = [{"run_type": kwargs["run_type"],
                 "tbl_filename": os.path.join(kwargs["full_path_to_data_dir"],
                                              ent_filename)}
                    for ent_filename in os.listdir(kwargs["full_path_to_data_dir"])
                        if ent_filename.split("/")[-1].split(".")[1] in ["tbl"]]

    if kwargs["run_type"]=="r_hsaac":
        cmds = [{"run_type": kwargs["run_type"],
                 "ent_filename": os.path.join(kwargs["full_path_to_data_dir"],
                                              ent_filename)}
                    for ent_filename in os.listdir(kwargs["full_path_to_data_dir"])
                        if ent_filename.split("/")[-1].split(".")[1] in ["ent", "bio1", "pdb"]]

    if kwargs["run_type"]=="r_pssm_step_one":
        cmds = [{"run_type": kwargs["run_type"],
                 "ent_filename": os.path.join(kwargs["full_path_to_data_dir"],
                                              ent_filename), 
                 "psiblast_path": kwargs["full_path_to_psiblast_exec"]}
                    for ent_filename in os.listdir(kwargs["full_path_to_data_dir"])
                        if ent_filename.split("/")[-1].split(".")[1] in ["ent", "bio1", "pdb"]]

    if kwargs["run_type"]=="r_pssm_step_two":
        cmds = [{"run_type": kwargs["run_type"],
                 "psiblast_output": os.path.join(kwargs["full_path_to_data_dir"],
                                                 out_filename)}
                     for out_filename in os.listdir(kwargs["full_path_to_data_dir"])
                        if out_filename.split("/")[-1].split(".")[1] in ["out"]]

    if kwargs["run_type"]=="l_molecule_features":
        cmds = [{"run_type": kwargs["run_type"],
                "filepath": os.path.join(kwargs["full_path_to_data_dir"],
                    ent_filename)}
                    for ent_filename in os.listdir(kwargs["full_path_to_data_dir"])
                        if ent_filename.split("/")[-1].split(".")[1] in ["mol2", "sdf"]]

    if kwargs["run_type"]=="interaction_labels":
        cmds = [{"run_type": kwargs["run_type"],
                "protein_filepath": os.path.join(kwargs["full_path_to_data_dir"],
                    ent_filename),
                "ligand_file_mol2":kwargs["ligand_file_mol2"]}
                    for ent_filename in os.listdir(kwargs["full_path_to_data_dir"])
                        if ent_filename.split("/")[-1].split(".")[1] in ["pdb"]]
    return cmds
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
    """

    args = get_args()
    if args.run_type=="r_pssm_step_one" or args.run_type=="r_pssm_step_two":
      run_type_output_dir = "_".join(args.run_type.split("_")[:-2])
    else:
      run_type_output_dir = args.run_type

    paths = prjPaths(run_type=run_type_output_dir)
    logger, queue_listener, queue = logger_init(path=paths.LOGS_DIR,
                                                filename=run_type_output_dir)
    if args.run_type=="r_edge":
        from edgeFeatures import edge_features_main as run_cmd
    if args.run_type=="r_stride":
        from strideFeatures import relative_accessible_surface_area_main as run_cmd
    if args.run_type=="r_residueDepth":
        from residueDepthFeatures import residue_depth_features_main as run_cmd
    if args.run_type=="r_protrusion_idx":
        from psaiaFeatures import protrusion_idx_hydrophobicity_main as run_cmd
    if args.run_type=="r_hsaac":
        from hsaacFeatures import half_sphere_amino_acid_comp_main as run_cmd
    if args.run_type=="r_pssm_step_one" or args.run_type=="r_pssm_step_two":
        from pssmFeatures import pssm_main as run_cmd
    if args.run_type=="l_molecule_features":
        from moleculeFeatures import molecule_features_main as run_cmd
    if args.run_type=="interaction_labels":
        from interactionLabels import interaction_labels_main as run_cmd

    cmds = run_cmd_router(vars(args))

    logging.info("process_id: {} "
                 "check how many cores this process uses with "
                 "`ps -o pid,psr,comm -p <pid>`".format(os.getpid()))

    if args.process_in_parallel:
        pool = Pool(os.cpu_count(), worker_init, [queue])
        complex_code_by_time = [time for time in pool.map(run_cmd, cmds)]
        pool.close()
        pool.join()
        queue_listener.stop()
    else:
        complex_code_by_time = [time for time in map(run_cmd, cmds)]

    path_to_run_time_pickle=os.path.join(paths.RUN_TYPE_OUT_DIR, "{}_times.p".format(run_type_output_dir))
    to_pickle(obj_to_pickle=complex_code_by_time,
              path=path_to_run_time_pickle)
# end

if __name__ == "__main__":
    main()
