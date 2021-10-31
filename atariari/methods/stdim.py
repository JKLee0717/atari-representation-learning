import random

import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy, Cutout
from .trainer import Trainer
from .utils import EarlyStopping
from torchvision import transforms
import torchvision.transforms.functional as TF
from atariari.benchmark.envs import make_vec_envs  # action space 불러오기
from atariari.methods.utils import get_argparser   # action space 불러오기

############ Action space #############
parser = get_argparser()
args = parser.parse_args()
env_action_space_size = make_vec_envs(args.env_name, args.seed,  args.num_processes, args.num_frame_stack, not args.no_downsample, args.color).action_space.n
#######################################

class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class InfoNCESpatioTemporalTrainer(Trainer):
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.patience = self.config["patience"]
        self.classifier1 = nn.Linear(self.encoder.hidden_size, self.encoder.local_layer_depth).to(device)  # x1 = global, x2=patch, n_channels = 32
        self.classifier2 = nn.Linear(self.encoder.local_layer_depth, self.encoder.local_layer_depth).to(device)
        self.classifier3 = nn.Linear(self.encoder.hidden_size * 2, env_action_space_size).to(device)  # action classifier , num_of_actions = 6 현재 하드코딩인데.. 나중에 바꿔줘야 할지도??
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.device = device
        self.optimizer = torch.optim.Adam(list(self.classifier1.parameters()) + list(self.encoder.parameters()) +
                                          list(self.classifier2.parameters()),
                                          lr=config['lr'], eps=1e-5)
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder")
        self.transform = transforms.Compose([Cutout(n_holes=1, length=80)])

    def generate_batch(self, episodes, acts):
        total_steps = sum([len(e) for e in episodes])
        print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.batch_size, drop_last=True)
        #print(len(acts[10]), len(episodes[10]), len(acts[20]), len(episodes[20]), len(acts[30]), len(episodes[30]), len(acts[40]), len(episodes[40]) )
        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            acts_batch = [acts[x] for x in indices]
            #print(len(acts_batch[0]), len(episodes_batch[0]))
            x_t, x_tprev, x_that, ts, thats, act_tprev = [], [], [], [], [], []
            for i, episode in enumerate(episodes_batch):
                act = acts_batch[i]
                # Get one sample from this episode
                t, t_hat = 0, 0
                t, t_hat = np.random.randint(0, len(episode)), np.random.randint(0, len(episode))
                #print(len(episode), len(act))
                x_t.append(episode[t])
                x_tprev.append(episode[t - 1])
                act_tprev.append(act[t])
                ts.append([t])
            yield torch.stack(x_t).float().to(self.device) / 255., torch.stack(x_tprev).float().to(self.device) / 255. , torch.stack(act_tprev).float().to(self.device)

    def do_one_epoch(self, epoch, episodes, acts):
        mode = "train" if self.encoder.training and self.classifier1.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        accuracy1, accuracy2 = 0., 0.
        epoch_loss1, epoch_loss2 = 0., 0.
        data_generator = self.generate_batch(episodes, acts)
        for x_t, x_tprev, act_tprev  in data_generator:
            f_t_maps, f_t_prev_maps = self.encoder(x_t, fmaps=True), self.encoder(x_tprev, fmaps=True)

            # Loss 1: Global at time t, f5 patches at time t-1
            f_t, f_t_prev = f_t_maps['out'], f_t_prev_maps['f5']
            sy = f_t_prev.size(1)
            sx = f_t_prev.size(2)

            N = f_t.size(0)
            loss1 = 0.
            for y in range(sy):
                for x in range(sx):
                    predictions = self.classifier1(f_t)
                    positive = f_t_prev[:, y, x, :]
                    logits = torch.matmul(predictions, positive.t())
                    step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
                    loss1 += step_loss
            loss1 = loss1 / (sx * sy)

            # Loss 2: f5 patches at time t, with f5 patches at time t-1
            f_t = f_t_maps['f5']
            loss2 = 0.
            for y in range(sy):
                for x in range(sx):
                    predictions = self.classifier2(f_t[:, y, x, :])
                    positive = f_t_prev[:, y, x, :]
                    logits = torch.matmul(predictions, positive.t())
                    step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
                    loss2 += step_loss
            loss2 = loss2 / (sx * sy)

            ######## Loss 3: 새롭게 추가한 부분 ##########
            f_t_out, f_t_prev_out = f_t_maps['out'], f_t_prev_maps['out']
            x = torch.cat([f_t_out, f_t_prev_out], dim=1)  # 어떤 dim으로 concat할지?? 
            net_out = self.classifier3(x)
            act_tprev=act_tprev.to(torch.long)
            #print(act_tprev)
            actions_one_hot = torch.squeeze(F.one_hot(act_tprev, env_action_space_size)).float()  
            #print(net_out, actions_one_hot)
            loss3 = nn.MSELoss()(net_out, actions_one_hot)
            #print(loss1, loss2, loss3)
            loss = loss1 + loss2 + loss3*0.5
            ##############################################

            #loss = loss1 + loss2  # Baseline Loss

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            epoch_loss1 += loss1.detach().item()
            epoch_loss2 += loss2.detach().item()
            #preds1 = torch.sigmoid(self.classifier1(x1, x2).squeeze())
            #accuracy1 += calculate_accuracy(preds1, target)
            #preds2 = torch.sigmoid(self.classifier2(x1_p, x2_p).squeeze())
            #accuracy2 += calculate_accuracy(preds2, target)
            steps += 1
        self.log_results(epoch, epoch_loss1 / steps, epoch_loss2 / steps, epoch_loss / steps, prefix=mode)
        if mode == "val":
            self.early_stopper(-epoch_loss / steps, self.encoder)

    def train(self, tr_eps, val_eps, tr_acts, val_acts):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        for e in range(self.epochs):
            self.encoder.train(), self.classifier1.train(), self.classifier2.train(), self.classifier3.train()
            self.do_one_epoch(e, tr_eps, tr_acts)

            self.encoder.eval(), self.classifier1.eval(), self.classifier2.eval(), self.classifier3.eval()
            self.do_one_epoch(e, val_eps, val_acts)

            if self.early_stopper.early_stop:
                break
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.config['env_name'] + '.pt'))

    def log_results(self, epoch_idx, epoch_loss1, epoch_loss2, epoch_loss, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}, {}".format(prefix.capitalize(), epoch_idx, epoch_loss,
                                                                     prefix.capitalize()))
        self.wandb.log({prefix + '_loss': epoch_loss,
                        prefix + '_loss1': epoch_loss1,
                        prefix + '_loss2': epoch_loss2}, step=epoch_idx, commit=False)
