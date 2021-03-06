import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy import random
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from collections import OrderedDict
from src.dataloaders.kbp_dataset import KBPDataset
from src.dataloaders.data_augmentation import ToTensor, ToRightShape, RandomAugment, NormalizeData
from provided_code.general_functions import get_paths, sparse_vector_function
from provided_code.dose_evaluation_class import EvaluateDose
import src.models.networks as networks
from torchvision import transforms
import os


class Pix2PixModel(pl.LightningModule):

    def __init__(self, opt, model_name='pix2pix_default', stage='training'):
        super(Pix2PixModel, self).__init__()
        self.opt = opt
        self.model_name = model_name
        self.stage = stage

        self.generator = networks.define_G(self.opt)

        self.discriminator = networks.define_D(self.opt)

        self.criterionGAN = networks.GANLoss(use_lsgan=True, target_real_label=0.9)
        self.criterionFeat = networks.get_loss(self.opt.loss_function)

        self.inst_noise_sigma_curr = 0.0

    def get_inputs(self, data):
        input_A = data['ct']  # Returns tensors of size [batch_size, 1, 128, 128, 128, 1]
        input_B = data['dose']

        input_A.requires_grad = True
        input_B.requires_grad = True

        return input_A, input_B

    def forward(self, z):
        return self.generator(z)

    # Called every training iteration
    def training_step(self, batch, batch_id, optimizer_idx):
        ct_image, dose = self.get_inputs(batch)

        # Create the instance noise vectors based on specfied inst_noise_sigma
        self.inst_noise_sigma_curr = 0 if self.current_epoch > self.opt.inst_noise_sigma_iters else (1 - self.current_epoch/self.opt.inst_noise_sigma_iters)*self.opt.inst_noise_sigma
        self.inst_noise_mean = torch.full((ct_image.size()[0], 2, 128, 128, 128), 0, dtype=torch.float).type_as(ct_image)
        self.inst_noise_std = torch.full((ct_image.size()[0], 2, 128, 128, 128), self.inst_noise_sigma_curr, dtype=torch.float).type_as(ct_image)

        fake = self.forward(ct_image)
        fake = fake * batch['possible_dose_mask']

        if optimizer_idx == 0:  # Generator step
            loss, tqdm_dict = self.backward_G(fake, ct_image, dose)
        elif optimizer_idx == 1:  # Discriminator step
            loss, tqdm_dict = self.backward_D(fake, ct_image, dose)
        else:
            raise Exception("Invalid optimizer ID")

        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def training_epoch_end(self, outputs):
        results = {
            'log': {
                'current_noise': torch.Tensor([self.inst_noise_sigma_curr])
            }
        }

        return results

    # def on_after_backward(self):
        # self.plot_grad_flow(self.discriminator.named_parameters())
        # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.opt.weight_cliping_limit)

    def backward_D(self, fake, ct_image, dose):
        """Calculate GAN loss for the discriminator"""
        D_losses = {}
        inst_noise = torch.normal(mean=self.inst_noise_mean, std=self.inst_noise_std).type_as(ct_image)  # Create the noise tensor

        pred_fake_label = False
        pred_real_label = True

        if self.current_epoch > self.opt.inst_noise_sigma_iters:  # Change the labels for fake/real occasionally
            if random.randint(10) == 0:
                pred_fake_label = True
                pred_real_label = False

        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((ct_image, fake), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.discriminator(fake_AB.detach() + inst_noise)  # Add noise to discriminator input to make its life harder
        D_losses['loss_D_fake'] = self.criterionGAN(pred_fake, pred_fake_label)
        # Real
        real_AB = torch.cat((ct_image, dose), 1)
        pred_real = self.discriminator(real_AB + inst_noise)
        D_losses['loss_D_real'] = self.criterionGAN(pred_real, pred_real_label)
        # combine loss and calculate gradients
        loss_D = sum(D_losses.values())/len(D_losses)  # Loss is the average of the real and fake losses
        D_losses['d_loss'] = loss_D

        return loss_D, D_losses

    def backward_G(self, fake, ct_image, dose):
        """Calculate GAN, L1 and VGG loss for the generator"""
        G_losses = {}
        inst_noise = torch.normal(mean=self.inst_noise_mean, std=self.inst_noise_std).type_as(ct_image)  # Create the noise tensor

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((ct_image, fake), 1)
        with torch.no_grad():  # D requires no gradients when optimizing G
            pred_fake = self.discriminator(fake_AB + inst_noise)
        G_losses['loss_G_GAN'] = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B

        G_losses['loss_G_Feat'] = self.criterionFeat(fake, dose) * self.opt.lambda_A

        # combine loss and calculate gradients
        loss_G = sum(G_losses.values())/len(G_losses)  # Average the loss values
        G_losses['g_loss'] = loss_G

        return loss_G, G_losses

    def plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow

        Found: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/17'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

        self.logger.experiment.log_figure(figure=plt)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr_G, betas=(self.opt.beta1, self.opt.beta2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr_D, betas=(self.opt.beta1, self.opt.beta2))

        sched_g = {
            'scheduler': networks.get_scheduler(opt_g, self.opt),
            'monitor': 'loss',
            'name': 'generator_lr'
        }
        sched_d = {
            'scheduler': networks.get_scheduler(opt_d, self.opt),
            'monitor': 'loss',
            'name': 'discriminator_lr'
        }

        dict_g = {
            'optimizer': opt_g,
            'frequency': self.opt.n_generator,  # Number of times the generator step will run in a row
            'lr_scheduler': sched_g
        }
        dict_d = {
            'optimizer': opt_d,
            'frequency': self.opt.n_critic,  # Number of times the discriminator step will run in a row
            'lr_scheduler': sched_d
        }

        return (dict_g, dict_d)

    def prepare_data(self):
        # Define parent directory
        dataset_directory = '{}/data'.format(self.opt.primary_directory)
        training_data_dir = '{}/train-pats'.format(dataset_directory)
        validation_data_dir = '{}/validation-pats-no-dose'.format(dataset_directory)
        # path where any data generated by this code (e.g., predictions, models) are stored

        results_dir = '{}/results'.format(dataset_directory)  # parent path where results are stored

        model_results_path = '{}/{}'.format(results_dir, self.model_name)
        self.prediction_dir = '{}/{}-predictions'.format(model_results_path, self.stage)
        os.makedirs(self.prediction_dir, exist_ok=True)

        # Prepare the data directory
        plan_paths = get_paths(training_data_dir, ext='')  # gets the path of each plan's directory
        num_train_pats = np.minimum(150, len(plan_paths))  # number of plans that will be used to train model
        self.training_paths = plan_paths[:num_train_pats]  # list of training plans
        self.hold_out_paths = plan_paths[num_train_pats:]  # list of paths used for held out testing

    def train_dataloader(self):
        # Create the dataset using the specfied augmentations
        dataset = KBPDataset(self.opt, self.training_paths, mode_name='training_model', transform=transforms.Compose([
            ToTensor(),
            RandomAugment(self.opt, mask_size=self.opt.cut_blur_mask, augment=not self.opt.no_augment),
            NormalizeData(self.opt),
            ToRightShape()
        ]))
        print("Number of training patients: %d" % len(dataset))
        return DataLoader(dataset, batch_size=self.opt.batchSize, shuffle=True, num_workers=0)

    def generate_csv(self, batch):
        # Get patient ID and make a prediction
        # if self.val_dataloader().batch_size != 1:
        #     raise Exception("Batch size for validation must be 1!")

        pat_id = np.squeeze(batch['patient_list'])
        pat_path = np.squeeze(batch['patient_path_list']).tolist()
        image = batch['ct']

        generated = self.generator(image)
        # if not self.opt.no_scaling:
        #     generated = 40.0*generated + 40.0  # Scale back dose to 0 - 80
        dose_pred_gy = generated * batch['possible_dose_mask']
        dose_pred_gy = dose_pred_gy.view(1, 1, 128, 128, 128, 1)
        # Prepare the dose to save
        dose_pred_gy = np.squeeze(dose_pred_gy)
        dose_pred_gy = dose_pred_gy.cpu().numpy()
        dose_to_save = sparse_vector_function(dose_pred_gy)
        dose_df = pd.DataFrame(data=dose_to_save['data'].squeeze(), index=dose_to_save['indices'].squeeze(), columns=['data'])
        file_name = '{}/{}.csv'.format(self.prediction_dir, pat_id)
        dose_df.to_csv(file_name)

        return pat_path, file_name

    def validation_step(self, batch, batch_idx):
        dose_path, pred_path = self.generate_csv(batch)

        output = {
            'dose_path': dose_path,
            'pred_path': pred_path
        }

        return output

    def validation_epoch_end(self, outputs):
        dose_files = []
        pred_files = []

        for f in outputs:
            dose_files.append(f['dose_path'])
            pred_files.append(f['pred_path'])

        data_loader_hold_out_eval = KBPDataset(self.opt, dose_files, mode_name='evaluation')
        hold_out_prediction_loader = KBPDataset(self.opt, pred_files, mode_name='predicted_dose')

        dose_evaluator = EvaluateDose(data_loader_hold_out_eval, hold_out_prediction_loader)

        dvh_score = -1
        dose_score = -1

        if not data_loader_hold_out_eval.file_paths_list:
            print('No patient information was given to calculate metrics')
        else:
            dvh_score, dose_score = dose_evaluator.make_metrics()

        results = {"log": {"dose_score": torch.tensor(dose_score).cuda(), "dvh_score": torch.tensor(dvh_score).cuda()}}
        print(results)
        return results

    def val_dataloader(self):
        dataset = KBPDataset(self.opt, self.hold_out_paths, mode_name='dose_prediction', transform=transforms.Compose([
            ToTensor(),
            NormalizeData(self.opt),
            ToRightShape()
        ]))
        print("Number of validation patients: %d" % len(dataset))
        return DataLoader(dataset, batch_size=1, shuffle=False)

    # ---------------------- TEST_STEP ---------------------- #
    def test_step(self, batch, batch_id):
        self.generate_csv(batch)

    def test_dataloader(self):
        dataset = KBPDataset(self.opt, self.hold_out_paths, mode_name='dose_prediction')
        return DataLoader(dataset, batch_size=1, shuffle=False)
