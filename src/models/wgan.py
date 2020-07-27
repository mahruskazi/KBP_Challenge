import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import src.models.networks as networks
import src.models.medicalzoo.medzoo as medzoo
from src.dataloaders.kbp_dataset import KBPDataset
from provided_code.general_functions import get_paths, sparse_vector_function
from provided_code.dose_evaluation_class import EvaluateDose
import torch.autograd as autograd
import os
import numpy as np
import pandas as pd
from collections import OrderedDict


class WGan(pl.LightningModule):

    def __init__(self, opt, model_name='wgan_default', stage='training'):
        super(WGan, self).__init__()
        self.opt = opt
        self.model_name = model_name
        self.stage = stage

        self.generator = networks.define_G(self.opt)

        self.discriminator = networks.define_D(self.opt)

        self.one = torch.tensor(1, dtype=torch.float).cuda()
        self.mone = self.one * -1

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.gpu = 0

    def get_inputs(self, data):
        input_A = data['ct']  # Returns tensors of size [batch_size, 1, 128, 128, 128, 1]
        input_B = data['dose']

        return Variable(input_A)[..., 0].float(), Variable(input_B)[..., 0].float()

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_id, optimizer_idx):
        ct_image, dose = self.get_inputs(batch)

        fake = self.forward(ct_image)

        if optimizer_idx == 0:
            loss, tqdm_dict = self.backward_G(fake, ct_image, dose)
        elif optimizer_idx == 1:
            loss, tqdm_dict = self.backward_D(fake, ct_image, dose)
        else:
            raise Exception("Invalid optimizer ID")

        shim = torch.FloatTensor([0.0]).cuda()
        shim.requires_grad = True

        output = OrderedDict({
            'loss': shim,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def backward_D(self, fake, real_A, real_B):
        """Calculate GAN loss for the discriminator"""
        D_losses = {}

        for p in self.discriminator.parameters():
            p.requires_grad = True

        for p in self.discriminator.parameters():
            p.data.clamp_(-self.opt.weight_cliping_limit, self.opt.weight_cliping_limit)

        # Real
        real_AB = torch.cat((real_A, real_B), 1)
        d_loss_real = self.discriminator(real_AB)
        d_loss_real = d_loss_real.mean()
        d_loss_real.backward(self.mone)
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((real_A, fake), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        d_loss_fake = self.discriminator(fake_AB.detach())
        d_loss_fake = d_loss_fake.mean()
        d_loss_fake.backward(self.one)

        # gradient_penalty = self.calc_gradient_penalty(self.discriminator, real_AB.data, fake_AB.data)
        # gradient_penalty.backward()
        # for p in d_loss.parameters():
        #     p.requires_grad = False
        d_loss = d_loss_fake - d_loss_real  # + gradient_penalty
        D_losses['d_loss'] = d_loss
        D_losses['Wasserstein_D'] = d_loss_real - d_loss_fake

        return d_loss, D_losses

    def backward_G(self, fake, real_A, real_B):
        """Calculate GAN, L1 and VGG loss for the generator"""
        G_losses = {}

        for p in self.discriminator.parameters():
            p.requires_grad = False

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((real_A, fake), 1)
        g_loss = self.discriminator(fake_AB)
        g_loss = g_loss.mean()
        g_loss.backward(self.mone)
        g_cost = -g_loss
        G_losses['g_cost'] = g_cost

        return g_loss, G_losses

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        # print real_data.size()
        alpha = torch.rand(self.opt.batchSize, 2, 128, 128, 128)
        alpha = alpha.cuda(self.gpu) if self.use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if self.use_cuda:
            interpolates = interpolates.cuda(self.gpu)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(self.gpu) if self.use_cuda else torch.ones(
                                  disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        LAMBDA = 10
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def configure_optimizers(self):
        beta2 = 0.999

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr_G, betas=(self.opt.beta1, beta2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr_D, betas=(self.opt.beta1, beta2))

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
            'frequency': self.opt.n_generator,
            'lr_scheduler': sched_g
        }
        dict_d = {
            'optimizer': opt_d,
            'frequency': self.opt.n_critic,
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
        dataset = KBPDataset(self.opt, self.training_paths, mode_name='training_model')
        print("Number of training patients: %d" % len(dataset))
        return DataLoader(dataset, batch_size=self.opt.batchSize, shuffle=True, num_workers=0)

    def generate_csv(self, batch):
        # Get patient ID and make a prediction
        # if self.val_dataloader().batch_size != 1:
        #     raise Exception("Batch size for validation must be 1!")

        pat_id = np.squeeze(batch['patient_list'])
        pat_path = np.squeeze(batch['patient_path_list']).tolist()
        image = Variable(batch['ct'])
        image = image[..., 0].float()

        generated = self.generator(image)
        if not self.opt.no_scaling:
            generated = 40.0*generated + 40.0  # Scale back dose to 0 - 80
        dose_pred_gy = generated.view(1, 1, 128, 128, 128, 1)
        dose_pred_gy = dose_pred_gy * batch['possible_dose_mask']
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

        if not data_loader_hold_out_eval.file_paths_list:
            print('No patient information was given to calculate metrics')
            results = {
                'dvh_score': -1,
                'dose_score': -1
            }
        else:
            dvh_score, dose_score = dose_evaluator.make_metrics()

            results = {
                'dvh_score': dvh_score,
                'dose_score': dose_score
            }
        print(results)
        self.logger.experiment.log_metrics(results)
        return results

    def val_dataloader(self):
        dataset = KBPDataset(self.opt, self.hold_out_paths, mode_name='dose_prediction')
        print("Number of validation patients: %d" % len(dataset))
        return DataLoader(dataset, batch_size=1, shuffle=False)

    # ---------------------- TEST_STEP ---------------------- #
    def test_step(self, batch, batch_id):
        self.generate_csv(batch)

    def test_dataloader(self):
        dataset = KBPDataset(self.opt, self.hold_out_paths, mode_name='dose_prediction')
        return DataLoader(dataset, batch_size=1, shuffle=False)
