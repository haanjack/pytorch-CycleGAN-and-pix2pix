import os
import torch
from .base_model import BaseModel
from . import networks


try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        self.use_fp16 = opt.use_fp16
        self.opt_level = opt.opt_level

        assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

        if opt.channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if opt.sync_bn:
            import apex
            print('usign apex synced BN')
            self.netG = apex.parallel.convert_syncbn_model(self.netG)
            self.netD = apex.parallel.convert_syncbn_model(self.netD)

        self.netG = self.netG.cuda().to(memory_format=memory_format)
        self.netD = self.netD.cuda().to(memory_format=memory_format)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        if self.use_fp16:
            self.netG, self.optimizer_G = amp.initialize(self.netG, self.optimizer_G, opt_level=self.opt_level)
            self.netD, self.optimizer_D = amp.initialize(self.netD, self.optimizer_D, opt_level=self.opt_level)

        if opt.distributed:
            self.netG = DDP(self.netG, delay_allreduce=True)
            self.netD = DDP(self.netD, delay_allreduce=True)
        
        if self.isTrain:
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        if self.opt.prof >= 0: torch.cuda.nvtx.range_push("set_input")
        
        AtoB = self.opt.direction == 'AtoB'
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_A = input['A' if AtoB else 'B']
        self.real_B = input['B' if AtoB else 'A']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.opt.prof >= 0: torch.cuda.nvtx.range_pop()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        if self.use_fp16:
            with amp.scale_loss(self.loss_D, self.optimizer_D) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        if self.use_fp16:
            with amp.scale_loss(self.loss_G, self.optimizer_G) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_G.backward()

    def optimize_parameters(self):
        if self.opt.prof >= 0: torch.cuda.nvtx.range_push("forward")
        self.forward()                   # compute fake images: G(A)
        if self.opt.prof >= 0: torch.cuda.nvtx.range_pop()

        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        if self.opt.prof >= 0: torch.cuda.nvtx.range_push("backward_D")
        self.backward_D()                # calculate gradients for D
        if self.opt.prof >= 0: torch.cuda.nvtx.range_pop()

        if self.opt.prof >= 0: torch.cuda.nvtx.range_push("optimizer_D.step")
        self.optimizer_D.step()          # update D's weights
        if self.opt.prof >= 0: torch.cuda.nvtx.range_pop()

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        if self.opt.prof >= 0: torch.cuda.nvtx.range_push("backward_G")
        self.backward_G()                   # calculate graidents for G
        if self.opt.prof >= 0: torch.cuda.nvtx.range_pop()

        if self.opt.prof >= 0: torch.cuda.nvtx.range_push("optimizer_G.step")
        self.optimizer_G.step()             # udpate G's weights
        if self.opt.prof >= 0: torch.cuda.nvtx.range_pop()
