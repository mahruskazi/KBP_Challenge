import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from collections import OrderedDict
from src.dataloaders.kbp_dataset import KBPDataset
from provided_code.general_functions import get_paths, sparse_vector_function
from provided_code.dose_evaluation_class import EvaluateDose
import src.models.networks as networks
import os


class Pix2PixModel(pl.LightningModule):

    def __init__(self, opt, model_name='pix2pix_default', stage='training'):
        super(Pix2PixModel, self).__init__()
        self.opt = opt
        self.model_name = model_name
        self.stage = stage

        self.generator = networks.define_G(self.opt)

        self.discriminator = networks.define_D(self.opt)

        if self.opt.wasserstein:
            self.criterionGAN = networks.get_loss('wasserstein')
        else:
            self.criterionGAN = networks.GANLoss(use_lsgan=not self.opt.no_lsgan)

        if not self.opt.no_perceptual_loss:
            self.criterionResnet = networks.PerceptualLoss(self.opt)

        self.criterionFeat = networks.get_loss(self.opt.loss_function)

    def get_inputs(self, data):
        input_A = data['ct']  # Returns tensors of size [batch_size, 1, 128, 128, 128, 1]
        input_B = data['dose']

        return Variable(input_A)[..., 0].float(), Variable(input_B)[..., 0].float()

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_id, optimizer_idx):
        ct_image, dose = self.get_inputs(batch)

        if optimizer_idx == 0:
            loss, tqdm_dict = self.backward_G(ct_image, dose)
        elif optimizer_idx == 1:
            loss, tqdm_dict = self.backward_D(ct_image, dose)
        else:
            raise Exception("Invalid optimizer ID")

        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def backward_D(self, ct_image, real_dose):
        """Calculate GAN loss for the discriminator"""
        D_losses = {}

        with torch.no_grad():
            fake_dose = self.forward(ct_image)
            fake_dose = fake_dose.detach()
            fake_dose.requires_grad_()

        pred_fake, pred_real = self.discriminate(ct_image, fake_dose, real_dose)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False)
        D_losses['D_real'] = self.criterionGAN(pred_real, True)

        D_loss = sum(D_losses.values()).mean()
        D_losses['D_loss'] = D_loss

        return D_loss, D_losses

    def backward_G(self, ct_image, real_dose):
        """Calculate GAN, L1 and VGG loss for the generator"""
        G_losses = {}

        fake_dose = self.forward(ct_image)

        pred_fake, pred_real = self.discriminate(ct_image, fake_dose, real_dose)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True)

        num_D = len(pred_fake)
        print(num_D)
        GAN_Feat_loss = torch.FloatTensor(1).fill_(0)
        GAN_Feat_loss = GAN_Feat_loss.type_as(ct_image)
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                GAN_Feat_loss += unweighted_loss * self.opt.lambda_A / num_D
        G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_perceptual_loss:
            G_losses['perceptual'] = self.criterionResnet(fake_dose, real_dose) * self.opt.lambda_perceptual

        loss_G = sum(G_losses.values()).mean()
        G_losses['loss_G'] = loss_G

        return loss_G, G_losses

    def discriminate(self, ct_image, fake_dose, real_dose):
        fake_concat = torch.cat([ct_image, fake_dose], dim=1)
        real_concat = torch.cat([ct_image, real_dose], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.discriminator(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

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

        dict_g = {
            'optimizer': torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, beta2)),
            'frequency': 1,
            'lr_scheduler': sched_g
        }
        dict_d = {
            'optimizer': torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, beta2)),
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
