import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from collections import OrderedDict
from data.create_dataset import CreateDataset
import networks


class Pix2PixModel(pl.LightningModule):

    def __init__(self, opt):
        super(Pix2PixModel, self).__init__()
        self.opt = opt
        self.gpu_ids = None

        self.generator = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                           opt.norm, not opt.no_dropout, opt.init_type)

        use_sigmoid = opt.no_lsgan
        self.discriminator = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD,
                                               opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type)

        tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=tensor)
        self.criterionL1 = torch.nn.L1Loss()

    def get_inputs(self, data):
        input_A = data['A']
        input_B = data['B']

        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0])
            input_B = input_B.cuda(self.gpu_ids[0])

        input_A = Variable(input_A)
        input_B = Variable(input_B)

        return input_A, input_B

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_id, optimizer_idx):
        real_A, real_B = self.get_inputs(batch)
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
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.discriminator(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
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

    def train_dataloader(self):
        dataset = CreateDataset(self.opt)
        # Torch Dataloader combines a dataset and sampler, provides settings.
        return torch.utils.data.DataLoader(
            dataset,  # dataset class
            batch_size=self.opt.batchSize,  # how many samples/batch to load
            shuffle=not self.opt.serial_batches,  # reshuffle per epoch
            num_workers=int(self.opt.nThreads))
