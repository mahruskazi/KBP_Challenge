from src.kbp_dataset import KBPDataset
from torch.utils.data import DataLoader
from provided_code.general_functions import get_paths
import numpy as np
from tqdm import tqdm

primary_directory = '/Users/mkazi/Google Drive/KBP_Challenge'

dataset_dir = '{}/data'.format(primary_directory)
training_data_dir = '{}/train-pats'.format(dataset_dir)
# training_data_dir = '{}/validation-pats-no-dose'.format(dataset_dir)

plan_paths = get_paths(training_data_dir, ext='')  # gets the path of each plan's directory
num_train_pats = np.minimum(100, len(plan_paths))  # number of plans that will be used to train model
training_paths = plan_paths[:num_train_pats]


dataset = KBPDataset(plan_paths)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

max_val = 0.0
for _, batch in enumerate(tqdm(loader)):
# for batch in loader:
    arr = batch['ct'].flatten().numpy()

    # cond = arr<0
    # if np.any(cond):
    #     print(batch['patient_list'])
    if np.amax(arr) > max_val:
        max_val = np.amax(arr)
print(max_val)