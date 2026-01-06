import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from torchvision import models
from .vggloss import Vgg19_out
from .DenseNet import SAD, FE
from unet_test.nets.unet_training import CE_Loss

from CLIP import clip
from torchvision import transforms

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.


        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=1, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=1, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_SA', type=float, default=1, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_SB', type=float, default=1, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=-1, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['unet', 'clip']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['I_e']
        visual_names_B = []
        # visual_names_A = ['rec_A']
        # visual_names_B = []


        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['LOW', 'HIGH2', 'HIGH3', 'UNET']
        else:  # during test time, only load Gs
            self.model_names = ['LOW', 'HIGH2', 'HIGH3']

        self.netVgg19 = Vgg19_out()
        # self.netSAD = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'SAD', opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netLOW = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'LOW', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netHIGH2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'HIGH2', opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netHIGH3 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'HIGH3', opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netUNET = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'UNET', opt.norm,
        #                                   not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netSAD = SAD().cuda()
        self.netSAD.load_state_dict(torch.load('HSRUIE/checkpoints/maps_cyclegan/latest_net_SAD.pth',map_location={'cuda:1':'cuda:0'}))

        self.netNC = FE().cuda()
        self.netNC.load_state_dict(
            torch.load('HSRUIE/checkpoints/maps_cyclegan/latest_net_NC.pth',
                       map_location={'cuda:1': 'cuda:0'}))


        device = "cuda" if torch.cuda.is_available() else "cpu"
        

        if self.isTrain:
            self.criterionCycle = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(itertools.chain(self.netLOW.parameters(),self.netHIGH2.parameters(),self.netHIGH3.parameters(),
                                                              self.netUNET.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer1 = torch.optim.Adam(itertools.chain(self.netUNET.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip, preprocess = clip.load("RN50", device=device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整到 224x224 大小
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))  # 标准化
        ])
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.input = input['A'].to(self.device)
        self.label = input['B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.f_raw = self.netVgg19(self.input)
        self.f_c = self.netNC(self.f_raw)
        [self.fc1, self.fc2, self.fc3] = self.f_c
        [self.fraw1, self.fraw2, self.fraw3] = self.f_raw
        # self.fe1 = self.netLOW(self.input, self.fc1)
        self.fe1 = self.fc1
        self.fe2 = self.netHIGH2(self.input, self.fc2)
        self.fe3 = self.netHIGH3(self.fc3)
        self.f_e = [self.fe1, self.fe2, self.fe3]
        self.I_e = self.netSAD(self.f_e)

    def backward(self, epoch):
        """Calculate the loss for generators G_A and G_B"""
        # Identity loss
        self.loss_unet = CE_Loss(self.unet_out, self.label)
        self.loss_G = self.loss_unet
        self.loss_G.backward()
        

    def optimize_parameters(self,epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        self.optimizer.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward(epoch)             # calculate gradients for G_A and G_B
        self.optimizer.step()       # update G_A and G_B's weights
