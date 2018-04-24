#!/usr/bin/env python
import os
import argparse
import pandas as pd
from matplotlib import *
from matplotlib.pyplot import *

def get_log_parsing_script():
  dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  return dirname + '/parse_log.py'

def parse_args():
  description = ('plot train and test loss in one figure')
  parser = argparse.ArgumentParser(description=description)

  parser.add_argument('logfile_path',
            help='Path to log file')

  parser.add_argument('output_dir',
            help='Directory in which to place output CSV files')

  args = parser.parse_args()
  return args

def parse_log(file_log, dir_out):
  os.system('%s %s %s' % (get_log_parsing_script(), file_log, dir_out))

  log_basename = os.path.basename(file_log)
  file_log_train = os.path.join(dir_out, log_basename + '.train')
  file_log_test = os.path.join(dir_out, log_basename + '.test')    
  return file_log_train, file_log_test

def plot_train_test_loss(file_log_train, file_log_test, dir_out):
  train_log = pd.read_csv(file_log_train)
  test_log = pd.read_csv(file_log_test)
  _, ax1 = subplots(figsize=(15, 10))
  ax1.plot(train_log["NumIters"], train_log["loss"], alpha=0.4)
  ax1.plot(test_log["NumIters"], test_log["loss"], 'g')
  ax1.set_xlabel('iteration')
  ax1.set_ylabel('train loss')
  savefig(os.path.join(dir_out, "train_test_image.png")) #save image as png    

def main():
  args = parse_args()
  file_log_train, file_log_test = parse_log(args.logfile_path, args.output_dir)
  plot_train_test_loss(file_log_train, file_log_test, args.output_dir)

if __name__ == '__main__':
  main()
