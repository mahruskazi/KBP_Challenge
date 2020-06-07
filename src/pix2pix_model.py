import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from collections import OrderedDict
from kbp_dataset import KBPDataset
from provided_code.general_functions import get_paths, make_directory_and_return_path, sparse_vector_function
import networks
import sys
import os


class Pix2PixModel(pl.LightningModule):

    def __init__(self, opt, model_name='pix2pix_default', stage='training'):
        super(Pix2PixModel, self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.model_name = model_name
        self.stage = stage

        self.generator = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                           opt.norm, not opt.no_dropout, opt.init_type)

        use_sigmoid = opt.no_lsgan
        self.discriminator = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD,
                                               opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type)

        self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan)
        self.criterionL1 = torch.nn.L1Loss()

    def get_inputs(self, data):
        input_A = data['ct']
        input_B = data['dose']

        return Variable(input_A), Variable(input_B)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_id, optimizer_idx):
        real_A, real_B = self.get_inputs(batch)

        real_A = real_A[0].permute(0, 4, 1, 2, 3).float()
        real_B = real_B[0].permute(0, 4, 1, 2, 3).float()
        fake = self.forward(real_A)

        if optimizer_idx == 0:
            loss = self.backward_G(fake, real_A, real_B)
            tqdm_dict = {'g_loss': loss}
        elif optimizer_idx == 1:
            loss = self.backward_D(fake, real_A, real_B)
            tqdm_dict = {'d_loss': loss}
        else:
            raise Exception("Invalid optimizer ID")

        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def backward_D(self, fake, real_A, real_B):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((real_A, fake), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.discriminator(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = self.discriminator(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        return loss_D

    def backward_G(self, fake, real_A, real_B):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((real_A, fake), 1)
        pred_fake = self.discriminator(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_G_L1 = self.criterionL1(fake, real_B) * self.opt.lambda_A
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1

        return loss_G

    def configure_optimizers(self):
        beta2 = 0.999

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, beta2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, beta2))

        def lambda_rule(epoch):
            lr_l = 1.0 - \
                max(0, epoch + 1 + self.opt.epoch_count - self.opt.niter) / \
                float(self.opt.niter_decay + 1)
            return lr_l

        sched_g = lr_scheduler.LambdaLR(opt_g, lr_lambda=lambda_rule)
        sched_d = lr_scheduler.LambdaLR(opt_d, lr_lambda=lambda_rule)

        return [opt_g, opt_d], [sched_g, sched_d]

    def prepare_data(self):
        primary_directory = '/Users/mkazi/repos/python_projects/pytorch'
        sys.path.insert(0, primary_directory)

        # Define parent directory
        training_data_dir = '{}/train-pats'.format(primary_directory)
        validation_data_dir = '{}/validation-pats-no-dose'.format(primary_directory)
        # path where any data generated by this code (e.g., predictions, models) are stored
        results_dir = '{}/results'.format(primary_directory)  # parent path where results are stored

        model_results_path = '{}/{}'.format(results_dir, self.model_name)
        self.prediction_dir = '{}/{}-predictions'.format(model_results_path, self.stage)
        os.makedirs(self.prediction_dir, exist_ok=True)

        # Prepare the data directory
        plan_paths = get_paths(training_data_dir, ext='')  # gets the path of each plan's directory
        num_train_pats = np.minimum(100, len(plan_paths))  # number of plans that will be used to train model
        self.training_paths = plan_paths[:num_train_pats]  # list of training plans
        self.hold_out_paths = plan_paths[num_train_pats:]  # list of paths used for held out testing

    def train_dataloader(self):
        dataset = KBPDataset(self.training_paths)
        print("Number of patients: %d" % len(dataset))
        # Torch Dataloader combines a dataset and sampler, provides settings.
        return DataLoader(dataset, batch_size=self.opt.batchSize, shuffle=True, num_workers=0)

    # ---------------------- TEST_STEP ---------------------- #
    def test_step(self, batch, batch_id):
        # Get patient ID and make a prediction
        pat_id = batch['patient_list'][0]
        image = Variable(batch['ct'])
        image = image[0].permute(0, 4, 1, 2, 3).float()

        dose_pred_gy = self.generator(image)
        dose_pred_gy = dose_pred_gy.permute(0, 2, 3, 4, 1).float()
        # dose_pred_gy = dose_pred_gy * batch['possible_dose_mask'][0]
        # Prepare the dose to save
        dose_pred_gy = np.squeeze(dose_pred_gy)
        dose_pred_gy = dose_pred_gy.numpy()
        dose_to_save = sparse_vector_function(dose_pred_gy)
        dose_df = pd.DataFrame(data=dose_to_save['data'].squeeze(), index=dose_to_save['indices'].squeeze(), columns=['data'])
        file_name = '{}/{}.csv'.format(self.prediction_dir, pat_id)
        print("Saving: " + file_name)
        dose_df.to_csv(file_name)

    def test_dataloader(self):
        dataset = KBPDataset(self.hold_out_paths)
        # Torch Dataloader combines a dataset and sampler, provides settings.
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)