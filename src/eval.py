import pytorch_lightning as pl
from src.pix2pix_model import Pix2PixModel
from options.train_options import TrainOptions
from provided_code.general_functions import get_paths
from src.kbp_dataset import KBPDataset
from torch.utils.data import DataLoader
from provided_code.dose_evaluation_class import EvaluateDose
import numpy as np

primary_directory = '/Users/mkazi/Google Drive/KBP_Challenge'

args = ['--which_model_netG', 'unet_128_3d',
        '--which_direction', 'AtoB',
        '--batchSize', '1',
        '--input_nc', '1',
        '--lambda_A', '100']
opt = TrainOptions().parse(args)

dataset_dir = '{}/data'.format(primary_directory)
training_data_dir = '{}/train-pats'.format(dataset_dir)

plan_paths = get_paths(training_data_dir, ext='')  # gets the path of each plan's directory
num_train_pats = np.minimum(100, len(plan_paths))  # number of plans that will be used to train model
hold_out_paths = plan_paths[num_train_pats:]  # list of paths used for held out testing
data_loader_hold_out_eval = KBPDataset(hold_out_paths, mode_name='evaluation')

prediction_dir = '{}/data/results/pix2pix_default/hold-out-tests-predictions'.format(primary_directory)
prediction_paths = get_paths(prediction_dir, ext='csv')
hold_out_prediction_loader = KBPDataset(prediction_paths, mode_name='predicted_dose')

dose_evaluator = EvaluateDose(data_loader_hold_out_eval, hold_out_prediction_loader)

if not data_loader_hold_out_eval.file_paths_list:
    print('No patient information was given to calculate metrics')
else:
    dvh_score, dose_score = dose_evaluator.make_metrics()
    print('For this out-of-sample test:\n'
          '\tthe DVH score is {:.3f}\n '
          '\tthe dose score is {:.3f}'.format(dvh_score, dose_score))
