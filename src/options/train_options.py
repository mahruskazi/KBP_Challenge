from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--display_freq',
            type=int,
            default=200,
            help='frequency of showing training results on screen')
        self.parser.add_argument(
            '--display_single_pane_ncols',
            type=int,
            default=0,
            help='if positive, display all images in a single visdom web panel with certain number of images per row.'
        )
        self.parser.add_argument(
            '--update_html_freq',
            type=int,
            default=1000,
            help='frequency of saving training results to html')
        self.parser.add_argument(
            '--print_freq',
            type=int,
            default=200,
            help='frequency of showing training results on console')
        self.parser.add_argument(
            '--save_latest_freq',
            type=int,
            default=5000,
            help='frequency of saving the latest results')
        self.parser.add_argument(
            '--save_epoch_freq',
            type=int,
            default=5,
            help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument(
            '--continue_train',
            action='store_true',
            help='continue training: load the latest model')
        self.parser.add_argument(
            '--epoch_count',
            type=int,
            default=50,
            help='number of epochs to train the model'
        )
        self.parser.add_argument(
            '--load_epoch',
            type=int,
            default=-1,
            help='continue training from this epoch'
        )
        self.parser.add_argument(
            '--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument(
            '--which_epoch',
            type=str,
            default='latest',
            help='which epoch to load? set to latest to use latest cached model'
        )
        self.parser.add_argument(
            '--niter',
            type=int,
            default=100,
            help='# of iter at starting learning rate')
        self.parser.add_argument(
            '--niter_decay',
            type=int,
            default=100,
            help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument(
            '--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument(
            '--beta2', type=float, default=0.999, help='The exponential decay rate for the second-moment estimates')
        self.parser.add_argument(
            '--lr_G',
            type=float,
            default=0.0001,
            help='initial learning rate for adam')
        self.parser.add_argument(
            '--lr_D',
            type=float,
            default=0.0004,
            help='initial learning rate for adam')
        self.parser.add_argument(
            '--no_lsgan',
            action='store_true',
            help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument(
            '--wasserstein',
            action='store_true',
            help='If true, use ')
        self.parser.add_argument(
            '--lambda_A',
            type=float,
            default=10.0,
            help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument(
            '--lambda_B',
            type=float,
            default=10.0,
            help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument(
            '--lambda_identity',
            type=float,
            default=0.5,
            help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.'
            'For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1'
        )
        self.parser.add_argument(
            '--pool_size',
            type=int,
            default=50,
            help='the size of image buffer that stores previously generated images')
        self.parser.add_argument(
            '--no_html',
            action='store_true',
            help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/'
        )
        self.parser.add_argument(
            '--lr_policy',
            type=str,
            default='lambda',
            help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument(
            '--lr_decay_iters',
            type=int,
            default=50,
            help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument(
            '--lr_step_size',
            type=int,
            default=50,
            help='half cycle length for cyclic lr policy')
        self.parser.add_argument(
            '--lr_max',
            type=float,
            default=0.01,
            help='peak lr for cyclic lr policy')
        self.parser.add_argument(
            '--loss_function',
            type=str,
            default='smoothed_L1',
            help='loss function: L1|smoothed_L1|perceptual')
        self.parser.add_argument(
            '--norm_D',
            type=str,
            default='spectralbatch',
            help='loss function: batch|spectralbatch')
        self.parser.add_argument(
            '--norm_G',
            type=str,
            default='spectralbatch',
            help='loss function: batch|spectralbatch')
        self.parser.add_argument(
            '--training_size',
            type=int,
            default=100,
            help='The number of images used for training')
        self.parser.add_argument(
            '--patience',
            type=int,
            default=10,
            help='used for plateau lr policy')
        self.parser.add_argument('--lambda_perceptual', type=float, default=10.0, help='weight for perceptual loss')
        self.parser.add_argument(
            '--resnet_depth',
            type=int,
            default=50,
            help='Pretrained resnet depth: 18|50')
        self.parser.add_argument(
            '--num_D',
            type=int,
            default=1,
            help='Number of discriminators used for training')
        self.parser.add_argument(
            '--n_critic',
            type=int,
            default=1,
            help='Number of times iterations of discriminator per generator iteration')
        self.parser.add_argument(
            '--n_generator',
            type=int,
            default=1,
            help='Number of times iterations of generator per discriminator iteration')
        self.parser.add_argument(
            '--weight_cliping_limit',
            type=float,
            default=0.01,
            help='clipping parameter for WGAN')
        self.parser.add_argument(
            '--no_perceptual_loss',
            action='store_true',
            help='do NOT use perceptual loss while training the generator'
        )
        self.parser.add_argument(
            '--no_augment',
            action='store_true',
            help='do NOT augment the data for training'
        )
        self.parser.add_argument('--inst_noise_sigma', type=float, default=0.0, help='Noise to add to the discriminator input, 0.0 for no noise')
        self.parser.add_argument('--inst_noise_sigma_iters', type=int, default=200, help='Number of iterations to bring noise to 0.0')
        self.parser.add_argument('--cut_blur_mask', type=int, default=64, help='Size of the image to CutBlur (cubed)')

        self.isTrain = True
