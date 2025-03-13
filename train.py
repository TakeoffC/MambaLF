import os
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
from network.model import PointModel
from loss_function import KeypointLoss
from mambalf.datasets.hp_loader import PatchesDataset
from torch.utils.data import DataLoader
from evaluation.evaluate import evaluate_keypoint_net


def log_training_info(filename, message):
    with open(filename, 'a') as f:
        f.write(message + '\n')


def log_evaluation_info(filename, message):
    with open(filename, 'a') as f:
        f.write(message + '\n')


def evaluation(epoch, min_loss, evaluation_summary_file):
    evaluation_messages = []
    msg = f"Evaluating epoch{epoch}, Min_loss is {min_loss}."
    print(msg)
    evaluation_messages.append(msg)

    # Load data in 320x240
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.set_device(0)

    # Load data in 640x480
    hp_dataset_640x480 = PatchesDataset(root_dir="hpatches/hpatches-sequences-release",
                                        use_color=True, output_shape=(640, 480),
                                        type='all')
    data_loader_640x480 = DataLoader(hp_dataset_640x480,
                                     batch_size=1,
                                     pin_memory=False,
                                     shuffle=False,
                                     num_workers=0,
                                     worker_init_fn=None,
                                     sampler=None)

    torch.manual_seed(0)
    ckpt = torch.load(
        f'checkpoints/mambalf-1001_{epoch}.pth')
    model_val = PointModel(is_test=True)
    model_val.load_state_dict(ckpt['model_state'], strict=False)
    model_val = model_val.cuda()
    model_val = model_val.eval()

    print('Evaluating in 640x480, 1000 points')
    evaluation_messages.append('Evaluating in 640x480, 1000 points')
    rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net(
        data_loader_640x480,
        model_val,
        output_shape=(640, 480),
        top_k=1000)

    msg = 'Repeatability: {0:.3f}'.format(rep)
    print(msg)
    evaluation_messages.append(msg)

    msg = 'Localization Error: {0:.3f}'.format(loc)
    print(msg)
    evaluation_messages.append(msg)

    msg = 'H-1 Accuracy: {:.3f}'.format(c1)
    print(msg)
    evaluation_messages.append(msg)

    msg = 'H-3 Accuracy: {:.3f}'.format(c3)
    print(msg)
    evaluation_messages.append(msg)

    msg = 'H-5 Accuracy: {:.3f}'.format(c5)
    print(msg)
    evaluation_messages.append(msg)

    msg = 'Matching Score: {:.3f}'.format(mscore)
    print(msg)
    evaluation_messages.append(msg)
    print('\n')

    for message in evaluation_messages:
        log_evaluation_info(evaluation_summary_file, message)
    return


class Trainer(object):
    def __init__(self, config, train_loader=None):
        self.config = config
        # data parameters
        self.train_loader = train_loader
        self.num_train = len(self.train_loader)

        # training parameters
        self.max_epoch = config.max_epoch
        self.start_epoch = config.start_epoch
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.lr_factor = config.lr_factor
        self.display = config.display

        # misc params
        self.use_gpu = config.use_gpu
        self.random_seed = config.seed
        self.gpu = config.gpu
        self.ckpt_dir = config.ckpt_dir
        self.ckpt_name = '{}-{}'.format(config.ckpt_name, config.seed)

        # build model
        self.model = PointModel(is_test=False)

        # training on GPU
        if self.use_gpu:
            torch.cuda.set_device(self.gpu)
            self.model.cuda()

        print('Number of model parameters: {:,}'.format(sum([p.data.nelement() for p in self.model.parameters()])))

        # build loss functional
        self.loss_func = KeypointLoss(config)

        # build optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[4, 8], gamma=self.lr_factor)

        # resume
        if int(self.config.start_epoch) > 0:
            self.config.start_epoch, self.model, self.optimizer, self.lr_scheduler = self.load_checkpoint(
                int(self.config.start_epoch), self.model, self.optimizer, self.lr_scheduler)

    def train(self):
        training_summary_file = os.path.join(self.ckpt_dir, 'training_summary.txt')
        evaluation_summary_file = os.path.join(self.ckpt_dir, 'evaluation_summary.txt')

        if not os.path.exists(training_summary_file):
            open(training_summary_file, 'w').close()
        if not os.path.exists(evaluation_summary_file):
            open(evaluation_summary_file, 'w').close()

        print("\nTrain on {} samples".format(self.num_train))
        self.save_checkpoint(0, self.model, self.optimizer, self.lr_scheduler)
        for epoch in range(self.start_epoch, self.max_epoch):
            print("\nEpoch: {}/{} --lr: {:.6f}".format(epoch + 1, self.max_epoch, self.lr))
            # train for one epoch
            self.train_one_epoch(epoch, training_summary_file, evaluation_summary_file)
            if self.lr_scheduler:
                self.lr_scheduler.step()
            # self.save_checkpoint(epoch+1, self.model, self.optimizer, self.lr_scheduler)

    def train_one_epoch(self, epoch, training_summary_file, evaluation_summary_file):
        self.model.train()
        min_loss = float('inf')
        for (i, data) in enumerate(tqdm(self.train_loader)):

            if self.use_gpu:
                source_img = data['image_aug'].cuda()
                target_img = data['image'].cuda()
                homography = data['homography'].cuda()

            source_img = Variable(source_img)
            target_img = Variable(target_img)
            homography = Variable(homography)

            # forward propogation
            output = self.model(source_img, target_img, homography)

            # compute loss
            loss, loc_loss, desc_loss, score_loss, corres_loss = self.loss_func(output)

            # compute gradients and update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print training info
            msg_batch = "Epoch:{} Iter:{} lr:{:.4f} " \
                        "loc_loss={:.4f} desc_loss={:.4f} score_loss={:.4f} corres_loss={:.4f} " \
                        "loss={:.4f} " \
                .format((epoch + 1), i, self.lr, loc_loss.data, desc_loss.data, score_loss.data, corres_loss.data,
                        loss.data)

            if ((i % self.display) == 0):
                print(msg_batch)
                log_training_info(training_summary_file, msg_batch)

            if loss < min_loss:
                min_loss = loss
                self.save_checkpoint(epoch, self.model, self.optimizer, self.lr_scheduler)
                msg_batch = f"Minimum loss updated. Saved checkpoint.\n{msg_batch}"
                print(msg_batch)
                log_training_info(training_summary_file, msg_batch)
        evaluation(epoch, min_loss, evaluation_summary_file)
        return

    def save_checkpoint(self, epoch, model, optimizer, lr_scheduler):
        filename = self.ckpt_name + '_' + str(epoch) + '.pth'
        torch.save(
            {'epoch': epoch,
             'model_state': model.state_dict(),
             'optimizer_state': optimizer.state_dict(),
             'lr_scheduler_state': lr_scheduler.state_dict()},
            os.path.join(self.ckpt_dir, filename))

    def load_checkpoint(self, epoch, model, optimizer, lr_scheduler):
        filename = self.ckpt_name + '_' + str(epoch) + '.pth'
        ckpt = torch.load(os.path.join(self.ckpt_dir, filename))
        epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler_state'])

        print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt['epoch']))

        return epoch, model, optimizer, lr_scheduler
