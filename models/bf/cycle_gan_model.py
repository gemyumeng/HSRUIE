import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from torchvision import models
from .vggloss import Vgg19_out
from .DenseNet import SAD
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
        self.loss_names = ['idt', 'dis', 'clip']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'I_c']
        visual_names_B = []
        # visual_names_A = ['rec_A']
        # visual_names_B = []


        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['ND', 'NC']
        else:  # during test time, only load Gs
            self.model_names = ['FE']

        self.netVgg19 = Vgg19_out()
        # self.netSAD = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'SAD', opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netND = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'FE', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netNC = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'FE', opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netSAD = SAD().cuda()
        self.netSAD.load_state_dict(torch.load('/home/DATA_yuanbao/ym/Mynet/model_data/latest_net_SAD.pth'))
        # self.netSAD.load_state_dict(torch.jit.load('H:/code/Mynet/checkpoints/maps_cyclegan_s1/latest_net_SAD.pth'))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip, preprocess = clip.load("RN50", device=device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整到 224x224 大小
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))  # 标准化
        ])

        if self.isTrain:
            self.criterionCycle = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(itertools.chain(self.netND.parameters(),self.netNC.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        # self.real_B = input['C'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # underwater
        self.f_raw = self.netVgg19(self.real_A)
        self.f_c = self.netNC(self.f_raw)
        self.f_d = self.netND(self.f_raw)

        self.I_c = self.netSAD(self.f_c)   # G_B(G_A(A))
        self.f_ic = self.netVgg19(self.I_c)

        self.raw_clip = (self.real_A + 1) / 2
        self.raw_clip = self.raw_clip.squeeze(0)
        self.raw_clip = self.transform(self.raw_clip)
        self.raw_clip = self.raw_clip.unsqueeze(0)
        self.ic_clip = (self.I_c + 1) / 2
        self.ic_clip = self.ic_clip.squeeze(0)
        self.ic_clip = self.transform(self.ic_clip)
        self.ic_clip = self.ic_clip.unsqueeze(0)

        self.f_raw_clip = self.clip.encode_image(self.raw_clip)
        self.f_ic_clip = self.clip.encode_image(self.ic_clip)


    def backward(self):
        """Calculate the loss for generators G_A and G_B"""
        # Identity loss
        # Forward cycle loss || G_B(G_A(A)) - A||
        [raw1, raw2, raw3] = self.f_raw
        [d1, d2, d3] = self.f_d
        [c1, c2, c3] = self.f_c
        u1=d1+c1
        u2=d2+c2
        u3=d3+c3
        self.loss_dis = (self.criterionCycle(raw1, u1) + self.criterionCycle(raw2, u2) + self.criterionCycle(raw3, u3))* 1
        # Backward cycle loss || G_A(G_B(B)) - B||
        [c1, c2, c3] = self.f_c
        [ic1, ic2, ic3] = self.f_ic
        self.loss_idt = (self.criterionCycle(c1, ic1) + self.criterionCycle(c2, ic2) + self.criterionCycle(c3, ic3))* 1

        [_, rcl1, rcl2, rcl3, _] = self.f_raw_clip
        [_, iccl1, iccl2, iccl3, _] = self.f_ic_clip
        self.loss_clip = (self.criterionCycle(rcl1, iccl1) + self.criterionCycle(rcl2, iccl2) + self.criterionCycle(rcl3, iccl3))* 1


        self.loss_G = self.loss_dis + self.loss_idt + self.loss_clip
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        self.optimizer.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward()             # calculate gradients for G_A and G_B
        self.optimizer.step()       # update G_A and G_B's weights
