import torch
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm
import logging
import os
import glob
import numpy as np
import pickle
#import traceback
from tensorboardX import SummaryWriter

from utils import get_minibatch 
from gcn_model import CPI_GCN
#from results_processor import performance_metrics 

class ModelController:
  def __init__(self, replica_number, experiment_name, experiment_config, data, outdir):
    self.replica_number = replica_number
    self.experiment_name = experiment_name
    self.experiment_sig = "{}_replica_num_{}".format(self.experiment_name, self.replica_number)
    self.outdir = outdir
    self.chkpt_dir = os.path.join(self.outdir, "chkpts_{}_{}".format(self.experiment_name,
                                                                     self.replica_number))
    self.persist_dir = os.path.join(self.outdir, "persist_{}_{}".format(self.experiment_name,
                                                                        self.replica_number))
    self.summary_dir = os.path.join(self.outdir, "summaries")
    self.is_cuda = experiment_config["cuda"]
    self.clip_value = experiment_config["clip_value"]

    self.writer = SummaryWriter(self.summary_dir)

    self.model = CPI_GCN(layer_specs=experiment_config["layers"],
                         layer_args=experiment_config["layer_args"],
                         data=data,
                         outdir=self.outdir,
                         experiment_sig=self.experiment_sig)


    class_weights = torch.FloatTensor((experiment_config["pn_ratio"],
                                       1-experiment_config["pn_ratio"]))
    if self.is_cuda: 
      self.model = self.model.cuda()
      class_weights = class_weights.cuda()

    self.criterion = nn.NLLLoss(weight=class_weights)

    if experiment_config["optimizer"] == "SGD":
      self.optimizer = optim.SGD(self.model.parameters(), lr=experiment_config["learning_rate"])
    if experiment_config["optimizer"] == "Adam":
      self.optimizer = optim.Adam(self.model.parameters(), lr=experiment_config["learning_rate"])
  # end

  def fit_model(self, exp_specs, data):
    """
    trains model by iterating minibatches for specified number of epochs
    """

    assert("train" in data.keys() and "val" in data.keys()), "require train and val datasets"

    print("Fitting Model...")

    self.model.train() # set model to train mode

    complex_permutation = np.random.permutation(len(data["train"])) # train complex permutation

    global_train_step = 1
    for epoch in range(exp_specs["num_epochs"]):

      # loop through each complex
      for train_complex_idx in tqdm(complex_permutation,
                                    "|| training epoch {} of {} || deploying batches".format(epoch,
                                                                                             exp_specs["num_epochs"])):
        try: 
          for train_batch in get_minibatch(complex_data=data["train"][train_complex_idx],
                                           minibatch_size=exp_specs["train_minibatch_size"],
                                           is_cuda=self.is_cuda):
            # run train step
            try:
              train_ys, train_y_preds = self.model(train_batch)
              train_loss = self.criterion(train_y_preds, train_ys)
              self.optimizer.zero_grad()
              train_loss.backward()
              nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)
              self.optimizer.step()
            except RuntimeError as e: 
              logging.error("running train step error: {}".format(e))
              #traceback.print_exc()
              pass

            if global_train_step % exp_specs["report_every"] == 0:
              try:
                self.writer.add_scalar("{}/train_loss".format(self.experiment_sig),
                                       train_loss,
                                       global_train_step)
                for name, param in self.model.named_parameters():
                  if param.requires_grad:
                    self.writer.add_histogram("{}/params_{}".format(self.experiment_sig,
                                                                    name.replace(".", ",")),
                                              param.data,
                                              global_train_step)
                    self.writer.add_histogram("{}/grads_{}".format(self.experiment_sig, 
                                                                   name.replace(".", ",")),
                                              param.grad.data,
                                              global_train_step)
              except Exception as e:
                logging.error("train summaries error: {}".format(e))
                #traceback.print_exc()
                pass

            if global_train_step % exp_specs["checkpoint_every"] == 0:
              torch.save(self.model.state_dict(),
                         os.path.join(self.chkpt_dir,
                                      "cpi_{}.pth".format(global_train_step)))
  
              # keep only n latest checkpoints
              if len(self.chkpt_dir) > exp_specs["keep_n_chkpts"]:
                for rm_file in sorted(glob.glob(os.path.join(self.chkpt_dir,
                                                             "*.pth")),
                                      key=os.path.getmtime)[:-exp_specs["keep_n_chkpts"]]:
                  os.remove(rm_file)
  
            if global_train_step % exp_specs["val_every"] == 0:
  
              self.model.eval() # switch model mode to evaluate
              train_val_losses = []
              train_val_ys = []
              train_val_ys_preds = []
  
              for val_complex_idx in range(len(data["val"])):
                try:
                  for val_batch in get_minibatch(complex_data=data["val"][val_complex_idx],
                                                 minibatch_size=exp_specs["val_minibatch_size"],
                                                 is_cuda=self.is_cuda):
  
                    # run val step
                    try:
                      val_ys, val_y_preds = self.model(val_batch)
                      val_loss = self.criterion(val_y_preds, val_ys)
                      self.optimizer.zero_grad() # zero grads for good measure
                      train_val_losses.append(val_loss.detach().cpu()) # move tensor to cpu
                      train_val_ys.extend(val_ys.detach().cpu().numpy()) # move tensor to cpu
                      train_val_ys_preds.extend(val_y_preds.detach().cpu().numpy()) # move tensor to cpu
  
                    except RuntimeError as e: 
                      logging.error("running val step error: {}".format(e))
                      #traceback.print_exc()
                      pass
  
                    torch.cuda.empty_cache()
                except AssertionError as e:
                  logging.error("val error on complex of idx {}: {}".format(val_complex_idx, e))
                  #traceback.print_exc()
                  pass
                  continue

              try:
                self.writer.add_scalar("{}/val_mean_loss".format(self.experiment_sig),
                                       torch.mean(torch.Tensor(train_val_losses)),
                                       global_train_step)
  
              except Exception as e:
                logging.error("val summaries error: {}".format(e))
                #traceback.print_exc()
                pass

              self.model.train() # switch model mode back to train

            global_train_step += 1
            torch.cuda.empty_cache()

        except AssertionError as e:
          logging.error("train error on complex of idx {}: {}".format(train_complex_idx , e))
          #traceback.print_exc()
          pass
  # end

  def inference(self, exp_specs, data):
    """
    trains model by iterating minibatches for specified number of epochs
    """
    print("Running evaluation...")

    data_type = "test"
    latest_chkpt = sorted(glob.glob(os.path.join(self.chkpt_dir,
                                                 "*.pth")),
                          key=os.path.getmtime)[-1]
    self.model.load_state_dict(torch.load(latest_chkpt))
    if self.is_cuda: 
      self.model = self.model.cuda()
    self.model.eval() # switch model mode to evaluate

    all_complex_results = {}
    for eval_complex_idx in range(len(data)):
      complex_eval_losses = []
      complex_eval_ys = []
      complex_eval_ys_preds = []
      try: 
        for eval_batch in get_minibatch(complex_data=data[eval_complex_idx],
                                        minibatch_size=exp_specs["eval_minibatch_size"],
                                        is_cuda=self.is_cuda):
          # run eval step
          try:
            ys, y_preds = self.model(eval_batch)
            loss = self.criterion(y_preds, ys)
            self.optimizer.zero_grad() # zero grads for good measure
              
            loss = loss.detach().cpu().numpy() # move tensor to cpu
            ys = ys.detach().cpu().numpy() # move tensor to cpu
            y_preds = y_preds.detach().cpu().numpy() # move tensor to cpu

            complex_eval_losses.append(loss.tolist()) 
            complex_eval_ys.extend(ys)
            complex_eval_ys_preds.append(y_preds)

          except RuntimeError as e: 
            logging.error("running eval step error: {}".format(e))
            #traceback.print_exc()
            pass

        all_complex_results[str(data[eval_complex_idx]["complex_code"])]={"loss": complex_eval_losses,
                                                                          "ys": np.asarray(complex_eval_ys),
                                                                          "y_preds": np.vstack(complex_eval_ys_preds)}
      except AssertionError as e:
        logging.error("{} eval error on complex of idx {}: {}".format(data_type, eval_complex_idx, e))
        #traceback.print_exc()
        pass
        continue

        torch.cuda.empty_cache()

    pickle.dump(all_complex_results, open(os.path.join(self.persist_dir, "all_complex_results_{}.p".format(data_type)), "wb"))
    return all_complex_results
# end
