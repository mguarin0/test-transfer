"""
@author: mguarin0
desc: database files from the UNIREF-90 fasta files using makeblastdb to generate db file
      parameters for makeblastdb and psiblast alignments:

    `makeblastdb -in mydb.fsa -dbtype prot -out mydb`

    `psiblast -query {sequence of the protein}
     -db {database name you created using makeblastdb or other tools}
     -num_iterations 3
     -out_ascii_pssm {output file}`
"""

__author__ = "Michael Guarino"
__version__ = "0.1"
__all__ = [
        "pssm_main"
]

from timeit import default_timer as timer
import os
import subprocess
import traceback
import pickle
import logging

from utils import *

def _create_fasta_file(pdb_file_path, fasta_output_path, three_ltr_path, complex_code):
  """
  desc: reads pdbb protein record file; extracts amino acid sequence;
        recodes 3 letter AA code to 1 letter AA code and makes a fasta file
  args:
    pdb_file_path: path to pdbb protein record file
    fasta_output_path: output of amino acid sequence (1 letter AA code) fasta file
  """
  with open(pdb_file_path, "r") as entry_file:
    pdb_lines = [list(filter(None,
                             line.split(" ")))
                             for line in entry_file.readlines()]
    atom_lines = [line for line in pdb_lines if line[0]=="ATOM"]
  
  fasta_three_ltr = []
  i_res = 0
  for i, line in enumerate(atom_lines):
    current_res_idx = line[5]
    chain_id = "{}_{}".format(line[4], complex_code)
    res_id = "{}_{}_{}".format(current_res_idx, i_res, line[3])
    if i == 0:
      i_res += 1
      previous_res_idx = current_res_idx 
      fasta_three_ltr.append([(chain_id, res_id), line[3]])
    if previous_res_idx != current_res_idx:
      i_res += 1
      previous_res_idx = current_res_idx 
      fasta_three_ltr.append([(chain_id, res_id), line[3]])
    else: continue

  fasta_one_ltr = "".join([one_letter_amino_acid_codes[code[1]] 
                           for code in fasta_three_ltr])

  with open(three_ltr_path, "wb") as three_ltr_file:
    pickle.dump(fasta_three_ltr, three_ltr_file)

  with open(fasta_output_path, "w") as fasta_file:
    fasta_file.write(fasta_one_ltr)
# end

def _run_psiblast(fasta_path, psiblast_path, psiblast_output_path):
  """
  desc: this function calls psiblast tool from the cli
  args:
    fasta_path: path to complex fasta file
    psiblast_path: root path to psiblast tool binary files
    psiblast_output_path: output path for psiblast tool to write pssm calcultion
  returns:
    process: pointer to process running psiblast
  """
  os.environ["BLASTDB"] = os.path.join(psiblast_path, "output_2")
  psiblast_cmd = [os.path.join(psiblast_path, "psiblast"),
                  "-query",
                  fasta_path,
                  "-db",
                  "out_2",
                  "-num_iterations",
                  "3",
                  "-out_ascii_pssm",
                  psiblast_output_path]
  process = subprocess.Popen(psiblast_cmd)
  return process
# end

def _psiblast_formatter(psiblast_output, three_ltr_path):
  """
  desc:
  args:
  returns:
  """
  with open(three_ltr_path, "r") as three_ltr_file:
    three_ltr_lines = three_ltr_file.readlines()

  with open(psiblast_output, "r") as psiblast_file:
    psiblast_lines = [list(filter(None,
                             line.rstrip("\n").split(" ")))
                             for line in psiblast_file.readlines()]

    #for res_i, residue_line in enumerate(psiblast_lines):
# end

def pssm_main(kwargs):
  """
  desc:
  args:
  returns:
  """
  start_time = timer()
  if kwargs["run_type"]=="r_pssm_step_one":
    try:
      assert(all(k in ["run_type", "psiblast_path", "ent_filename"]
                       for k in kwargs.keys())), "incorrect list of args"
      assert(os.path.exists(kwargs["psiblast_path"])), "psi-blast path does not exist"
      assert(os.path.exists(kwargs["ent_filename"])), "ent filename does not exist"

      run_type = "_".join(kwargs["run_type"].split("_")[:-2])
      paths = prjPaths(run_type=run_type)
      filename, ext = kwargs["ent_filename"].split("/")[-1].split(".")
      if ext == "ent":
        complex_code = filename[3:7]
      elif ext == "bio1" or ext == "pdb":
        complex_code = filename[:4]

      fasta_path = os.path.join(paths.RUN_TYPE_OUT_DIR,
                                "{}.fasta.txt".format(complex_code))
      three_ltr_path = os.path.join(paths.RUN_TYPE_OUT_DIR,
                                "three_ltr_{}.p".format(complex_code))
      psiblast_output_path = os.path.join(paths.RUN_TYPE_OUT_DIR,
                                          "psiblast_output_{}.out".format(complex_code))

      # create fasta file
      _create_fasta_file(pdb_file_path=kwargs["ent_filename"],
                         fasta_output_path=fasta_path,
                         three_ltr_path=three_ltr_path,
                         complex_code=complex_code)
      logging.info("|| complex_code: {} "
                   "|| created fasta file".format(complex_code))

      # run psiblast tool
      process = _run_psiblast(fasta_path=fasta_path,
                              psiblast_path=kwargs["psiblast_path"],
                              psiblast_output_path=psiblast_output_path)
      logging.info("|| complex_code: {} "
                   "|| psiblast running at: {} "
                   "|| writing output to {}".format(complex_code,
                                                    process.pid,
                                                    psiblast_output_path))
      process.wait()

      end_time = timer()
      run_time = end_time-start_time
      return_msg = {kwargs["ent_filename"]: run_time}

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

  elif kwargs["run_type"]=="r_pssm_step_two":
    try:
      assert(all(k in ["run_type", "psiblast_output"]
                       for k in kwargs.keys())), "incorrect list of args"
      assert(os.path.exists(kwargs["psiblast_output"])), "psi-blast output path does not exist"

      run_type = "_".join(kwargs["run_type"].split("_")[:-2])
      paths = prjPaths(run_type=run_type)
      filename, ext = kwargs["psiblast_output"].split("/")[-1].split(".")
      complex_code = filename[:4]

      three_ltr_path = os.path.join(paths.RUN_TYPE_OUT_DIR,
                                "three_ltr_{}.p".format(complex_code))
      psiblast_features_output_path = os.path.join(paths.RUN_TYPE_OUT_DIR,
                                                   "r_pssm_{}.p".format(complex_code))

      psiblast_features = _psiblast_formatter(psiblast_output=kwargs["psiblast_output"],
                                              three_ltr_path=three_ltr_path)
      to_pickle(obj_to_pickle=psiblast_features,
                path=psiblast_features_output_path)
      end_time = timer()
      run_time = end_time-start_time
      return_msg = {psiblast_features_output_path: run_time}

    except Exception as err:
      logging.error("|| filename: {} "
                    "|| error thrown: {}".format(kwargs["psiblast_output"],
                                                 traceback.format_exc()))
      end_time = timer()
      run_time = end_time-start_time
      return_msg = {kwargs["psiblast_output"]: run_time}
      pass
    finally:
      return return_msg
# end
