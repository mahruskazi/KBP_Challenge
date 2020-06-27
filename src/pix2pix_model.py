import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from collections import OrderedDict
from src.kbp_dataset import KBPDataset
from provided_code.general_functions import get_paths, sparse_vector_function
from provided_code.dose_evaluation_class import EvaluateDose
import src.networks as networks
import os


class Pix2PixModel(pl.LightningModule):

    def __init__(self, opt, dataset_directory, model_name='pix2pix_default', stage='training'):
        super(Pix2PixModel, self).__init__()
        self.opt = opt
        self.model_name = model_name
        self.stage = stage
        self.dataset_directory = dataset_directory

        self.generator = networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.which_model_netG,
                                           self.opt.norm, not self.opt.no_dropout, self.opt.init_type)

        use_sigmoid = self.opt.no_lsgan
        self.discriminator = networks.define_D(self.opt.input_nc + self.opt.output_nc, self.opt.ndf, self.opt.which_model_netD,
                                               self.opt.n_layers_D, self.opt.norm, use_sigmoid, self.opt.init_type)

        self.criterionGAN = networks.GANLoss(use_lsgan=not self.opt.no_lsgan)
        # if not self.opt.no_vgg_loss:
        #     self.criterionVGG = networks.VGGLoss()

        self.criterionL1 = torch.nn.L1Loss()

    def get_inputs(self, data):
        input_A = data['ct']
        input_B = data['dose']

        return Variable(input_A), Variable(input_B)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_id, optimizer_idx):
        ct_image, dose = self.get_inputs(batch)  # Returns tensors of size torch.Size([batch_size, 1, 128, 128, 128, 1])
        ct_image = ct_image[..., 0].float()
        dose = dose[..., 0].float()

        fake = self.forward(ct_image)

        if optimizer_idx == 0:
            loss, tqdm_dict = self.backward_G(fake, ct_image, dose)
        elif optimizer_idx == 1:
            loss, loss_D_fake, loss_D_real = self.backward_D(fake, ct_image, dose)
            tqdm_dict = {
                'd_loss': loss,
                'loss_D_fake': loss_D_fake,
                'loss_D_real': loss_D_real
            }
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

        return loss_D, loss_D_fake, loss_D_real

    def backward_G(self, fake, real_A, real_B):
        """Calculate GAN, L1 and VGG loss for the generator"""
        G_losses = {}

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((real_A, fake), 1)
        with torch.no_grad():  # D requires no gradients when optimizing G
            pred_fake = self.discriminator(fake_AB)
        G_losses['loss_G_GAN'] = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        G_losses['loss_G_L1'] = self.criterionL1(fake, real_B) * self.opt.lambda_A

        # if not self.opt.no_vgg_loss:
        #     G_losses['vgg_G_loss'] = self.criterionVGG(fake, real_B) * self.opt.lambda_vgg

        # combine loss and calculate gradients
        loss_G = sum(G_losses.values()).mean()
        G_losses['g_loss'] = loss_G

        return loss_G, G_losses

    def configure_optimizers(self):
        beta2 = 0.999

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, beta2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, beta2))

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

        return [opt_g, opt_d], [sched_g, sched_d]

    def prepare_data(self):
        # Define parent directory
        training_data_dir = '{}/train-pats'.format(self.dataset_directory)
        validation_data_dir = '{}/validation-pats-no-dose'.format(self.dataset_directory)
        # path where any data generated by this code (e.g., predictions, models) are stored

        results_dir = '{}/results'.format(self.dataset_directory)  # parent path where results are stored

        model_results_path = '{}/{}'.format(results_dir, self.model_name)
        self.prediction_dir = '{}/{}-predictions'.format(model_results_path, self.stage)
        os.makedirs(self.prediction_dir, exist_ok=True)

        # Prepare the data directory
        plan_paths = get_paths(training_data_dir, ext='')  # gets the path of each plan's directory
        num_train_pats = np.minimum(100, len(plan_paths))  # number of plans that will be used to train model
        self.training_paths = plan_paths[:num_train_pats]  # list of training plans
        self.hold_out_paths = plan_paths[num_train_pats:]  # list of paths used for held out testing

    def train_dataloader(self):
        dataset = KBPDataset(self.training_paths, mode_name='training_model')
        print("Number of training patients: %d" % len(dataset))
        # Torch Dataloader combines a dataset and sampler, provides settings.
        return DataLoader(dataset, batch_size=self.opt.batchSize, shuffle=True)

    def generate_csv(self, batch):
        # Get patient ID and make a prediction
        # if self.val_dataloader().batch_size != 1:
        #     raise Exception("Batch size for validation must be 1!")

        pat_id = np.squeeze(batch['patient_list'])
        pat_path = np.squeeze(batch['patient_path_list']).tolist()
        image = Variable(batch['ct'])
        image = image[..., 0].float()

        generated = self.generator(image)
        # dose_pred_gy = 40.0*generated + 40.0  # Scale back dose to 0 - 80
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

        data_loader_hold_out_eval = KBPDataset(dose_files, mode_name='evaluation')
        hold_out_prediction_loader = KBPDataset(pred_files, mode_name='predicted_dose')

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
        dataset = KBPDataset(self.hold_out_paths, mode_name='dose_prediction')
        print("Number of validation patients: %d" % len(dataset))
        return DataLoader(dataset, batch_size=1, shuffle=False)

    # ---------------------- TEST_STEP ---------------------- #
    def test_step(self, batch, batch_id):
        self.generate_csv(batch)

    def test_dataloader(self):
        dataset = KBPDataset(self.hold_out_paths, mode_name='dose_prediction')
        return DataLoader(dataset, batch_size=1, shuffle=False)
