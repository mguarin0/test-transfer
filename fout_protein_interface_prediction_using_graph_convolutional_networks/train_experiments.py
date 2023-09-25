import os
import yaml
import traceback
import argparse
import logging
import pickle
import torch
import numpy as np
from utils import data_directory, experiment_directory, output_directory, seeds, pp
from train import ModelController

def get_args():
  parser = argparse.ArgumentParser(description="cpi prediction cli args")
  parser.add_argument("config_yaml_file",
                      type=str,
                      choices=["no_conv.yml",
                               "node_average.yml",
                               "node_edge_average.yml"],
                      help="yaml configuration file for current experiment")
  args = parser.parse_args()
  return args
# end

def main():
  args = get_args()

  # Load experiment specified in system args
  config_yaml_file = args.config_yaml_file
  print("Running Experiment File: {}".format(config_yaml_file))
  config_filename = config_yaml_file.split(".")[0] if "." in config_yaml_file else config_yaml_file
  exp_specs = yaml.load(open(os.path.join(experiment_directory, config_yaml_file), "r").read())

  # setup output directory
  outdir = os.path.join(output_directory, config_filename)
  if not os.path.exists(output_directory):
    os.mkdir(output_directory)
  if not os.path.exists(outdir):
    os.mkdir(outdir)

  logging.basicConfig(filename=os.path.join(outdir, "train_experiments.log"),
                      filemode="w",
                      format="%(name)s - %(levelname)s - %(message)s",
                      level=logging.DEBUG)

  # write experiment specifications to file
  with open(os.path.join(outdir, "experiment.yml"), "w") as f:
    f.write("{}\n".format(yaml.dump(exp_specs)))

  # perform each experiment
  prev_train_data_file = ""
  prev_val_data_file = ""
  prev_test_data_file = ""
  first_experiment = True

  # run through all experiments in yaml file
  for experiment_name, experiment_config in exp_specs["experiments"]:
    train_data_file = os.path.join(data_directory, experiment_config["train_data_file"])
    val_data_file = os.path.join(data_directory, experiment_config["val_data_file"])
    test_data_file = os.path.join(data_directory, experiment_config["test_data_file"])

    try:
      # Reuse train data if possible without having to reload
      if train_data_file != prev_train_data_file:
        print("Loading train data")
        train_list, train_data = pickle.load(open(train_data_file, "rb"), encoding="latin-1")
        prev_train_data_file = train_data_file

      if val_data_file != prev_val_data_file:
        print("Loading val data")
        val_list, val_data = pickle.load(open(val_data_file, "rb"), encoding="latin-1")
        prev_val_data_file = val_data_file

      if test_data_file != prev_test_data_file:
        print("Loading test data")
        test_list, test_data = pickle.load(open(test_data_file, "rb"), encoding="latin-1")
        prev_test_data_file = test_data_file

      # create data dictionary
      data = {"train": train_data, "val": val_data, "test": test_data}
      # perform experiment for each random seed
      for replica_number, seed_pair in enumerate(seeds):

        print("running experiment: {} replica num: {}".format(experiment_name, replica_number))

        if not os.path.exists(os.path.join(outdir, "chkpts_{}_{}".format(experiment_name, replica_number))):
          os.mkdir(os.path.join(outdir, "chkpts_{}_{}".format(experiment_name, replica_number)))
        if not os.path.exists(os.path.join(outdir, "persist_{}_{}".format(experiment_name, replica_number))):
          os.mkdir(os.path.join(outdir, "persist_{}_{}".format(experiment_name, replica_number)))

        # set torch and numpy seeds
        if torch.cuda.is_available():
          experiment_config["cuda"]=True
          torch.backends.cudnn.benchmark = False
          torch.backends.cudnn.deterministic = True
          torch.cuda.manual_seed_all(seed_pair["torch_seed"])
        else:
          experiment_config["cuda"]=False
          torch.manual_seed_all(seed_pair["torch_seed"])
        np.random.seed(int(seed_pair["np_seed"]))

        trainer = ModelController(replica_number=replica_number,
                                  experiment_name=experiment_name,
                                  experiment_config=experiment_config,
                                  data=data["train"],
                                  outdir=outdir)
        trainer.fit_model(exp_specs=exp_specs, data=data)
        results = trainer.inference(exp_specs=exp_specs,
                                    data=data["test"])
        #pp.pprint(results)

    except Exception as er:
      if er is KeyboardInterrupt:
        raise er
      ex_str = traceback.format_exc()
      logging.error(ex_str)
      logging.error("Experiment failed: {}".format(exp_specs))

if __name__=="__main__":
  print("torch version: {}".format(torch.__version__))
  print("cuda version: {}".format(torch.version.cuda))
  main()
