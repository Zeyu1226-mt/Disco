import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.distributions import Categorical
from tqdm import  tqdm
from collections import OrderedDict
import os
import copy
import time
import random
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
from torch import einsum, no_grad
# import models.models as models
from util.utils import *
from player_dataloader.dataloader_modified_dataset1 import read_dataset
from network.TimeSformer.timesformer.models.vit import TimeSformer, TimeSformerModified, TimeSformer_IA

parser = argparse.ArgumentParser(description='XZY')

# Dataset specification
parser.add_argument('--dataset', default='nba', type=str, help='volleyball or nba')
parser.add_argument('--data_path', default='/data_1T/xzy/NBA_dataset/', type=str, help='data path')
parser.add_argument('--image_width', default=224, type=int, help='Image width to resize')
parser.add_argument('--image_height', default=224, type=int, help='Image height to resize')
parser.add_argument('--random_sampling', action='store_true', help='random sampling strategy')
parser.add_argument('--num_frame', default=20, type=int, help='number of frames for each clip')
# parser.add_argument('--num_total_frame', default=72, type=int, help='number of total frames for each clip')
parser.add_argument('--num_activities', default=321, type=int, help='number of activity classes')

# Model parameters
parser.add_argument('--base_model', default=True, action='store_true', help='average pooling base model')
parser.add_argument('--backbone', default='resnet18', type=str, help='feature extraction backbone')
parser.add_argument('--dilation', action='store_true', help='use dilation or not')
parser.add_argument('--hidden_dim', default=256, type=int, help='transformer channel dimension')

# Motion parameters
parser.add_argument('--motion', default=False,  help='use motion feature computation')
parser.add_argument('--multi_corr', default=False,  help='motion correlation block at 4th and 5th')
parser.add_argument('--motion_layer', default=4, type=int, help='backbone layer for calculating correlation')
parser.add_argument('--corr_dim', default=64, type=int, help='projection for correlation computation dimension')
parser.add_argument('--neighbor_size', default=5, type=int, help='correlation neighborhood size')

# Transformer parameters
parser.add_argument('--nheads', default=4, type=int, help='number of heads')
parser.add_argument('--enc_layers', default=6, type=int, help='number of encoder layers')
parser.add_argument('--pre_norm', action='store_true', help='pre normalization')
parser.add_argument('--ffn_dim', default=512, type=int, help='feed forward network dimension')
parser.add_argument('--position_embedding', default='sine', type=str, help='various position encoding')
parser.add_argument('--num_tokens', default=12, type=int, help='number of queries')

# Aggregation parameters

parser.add_argument('--nheads_agg', default=4, type=int, help='number of heads for partial context aggregation')

# Training parameters
parser.add_argument('--random_seed', default=1, type=int, help='random seed for reproduction')
parser.add_argument('--epochs', default=100 , type=int, help='Max epochs')
parser.add_argument('--test_freq', default=1, type=int, help='print frequency')
parser.add_argument('--batch', default=4, type=int, help='Batch size')
parser.add_argument('--test_batch', default=4, type=int, help='Test batch size')
parser.add_argument('--lr', default=5e-7, type=float, help='Initial learning rate')
parser.add_argument('--max_lr', default=5e-5, type=float, help='Max learning rate')
parser.add_argument('--lr_step', default=5, type=int, help='step size for learning rate scheduler')
parser.add_argument('--lr_step_down', default=65, type=int, help='step down size (cyclic) for learning rate scheduler')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')
parser.add_argument('--gradient_clipping', action='store_true', help='use gradient clipping')
parser.add_argument('--max_norm', default=1.0, type=float, help='gradient clipping max norm')

# GPU
parser.add_argument('--device', default="0", type=str, help='GPU device')
#parser.add_argument('--accumulation_step', default="2", type=str, help='梯度累计')

# Load model
parser.add_argument('--load_model', default=False, action='store_true', help='load model')
parser.add_argument('--model_path', default="/home/xzy/xzy_nba/MLLM_VC/Stage_one/Playeraction/NBA_result_timesformer/resave/epoch66_90.54%.pth", type=str, help='pretrained model path')

args = parser.parse_args()
best_player_mca = 0.0
best_player_mpca = 0.0
best_mca_epoch = 0
best_mpca_epoch = 0


def main():
    global args

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())  #时间戳-年月日时分秒
    exp_name = 'dataset1_96_K600_adapter'
    save_path = '/media/nvme/XZY/Results/name321_action3/%s' % exp_name

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_set, test_set = read_dataset(args)

    train_loader = data.DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = data.DataLoader(test_set, batch_size=args.test_batch, shuffle=False, num_workers=8, pin_memory=True)

    # load the model
    #model = models.BaseModel(args)
    ori_model = TimeSformer(
        img_size=224,
        num_classes=321,
        num_frames=20,
        attention_type="divided_space_time",
        pretrained_model="/home/xzy/xzy_nba/MLLM_VC/Stage_one/Playeraction/network/TimeSformer/pretrained_model/TimeSformer_divST_96x4_224_K600.pyth",
    )
    # checkpoint = torch.load(args.model_path)
    # # print(checkpoint['state_dict'].keys())
    # ori_model.load_state_dict(checkpoint['state_dict'])

    model = TimeSformer_IA(
        original_model=ori_model,
        num_classes_identity=321,  # 身份类别数
        num_classes_action=3,  # 动作类别数
        split_layer=5  # 从第x层开始分叉
    )
    model = model.cuda()
    # model = torch.nn.DataParallel(model).cuda()
    # new_state_dict = {}
    #
    #
    # checkpoint = torch.load(args.model_path)
    # for k, v in checkpoint['state_dict'].items():
    #     new_state_dict[k.replace('module.', '')] = v
    # model.load_state_dict(new_state_dict, strict=True)
    # print(new_state_dict.keys())
    #print(model)
# /home/xzy/xzy_nba/LLM_VC/Player_identify/stage_one/network/TimeSformer/pretrained_model/TimeSformer_divST_32x32_224_HowTo100M.pyth
# /home/xzy/xzy_nba/LLM_VC/Player_identify/stage_one/NBA_result_timesformer/[nba]_DFGAR_<2024-09-22_00-19-11>/epoch44_91.13%.pth


    # 加载预训练权重，只针对共享层和分支部分
    # pretrained_state_dict = model.state_dict()  # 获取预训练模型的权重
    #
    # # 获取修改后的模型的当前权重
    # modified_state_dict = modified_model.state_dict()
    #
    # # 只保留预训练模型中与共享层和两个分支相关的权重
    # pretrained_keys = pretrained_state_dict.keys()
    # modified_keys = modified_state_dict.keys()
    # print("modified keys: ", modified_keys)
    #
    #
    # # 过滤掉分类头的参数（identity_head 和 action_head），因为这些是新定义的
    # pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if
    #                    k in modified_keys and 'identity_head' not in k and 'action_head' not in k}
    #
    # # 更新修改后的模型权重
    # modified_state_dict.update(pretrained_dict)
    #
    # # 加载更新后的权重到修改后的模型中
    # modified_model.load_state_dict(modified_state_dict)



    # model = torch.nn.DataParallel(modified_model).cuda()

    # get the number of model parameters
    parameters = 'Number of full model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
    print_log(save_path, '--------------------Number of parameters--------------------')
    print_log(save_path, parameters)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, args.lr, args.max_lr, step_size_up=args.lr_step,
                                                  step_size_down=args.lr_step_down, mode='triangular2',
                                                  cycle_momentum=False)

    if args.load_model:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 1

    # training phase
    for epoch in range(start_epoch, args.epochs + 1):
        print_log(save_path, '----- %s at epoch #%d' % ("Train", epoch))
        train_log = train(train_loader, model, criterion, optimizer, epoch)
        print_log(save_path, 'Player_accuracy: %.2f%%, Action_accuracy: %.2f%%, Loss: %.4f, Using %.1f seconds' %
                  (train_log['player_acc'], train_log['action_acc'], train_log['loss'], train_log['time']))
        print('Current learning rate is %f' % scheduler.get_last_lr()[0])
        scheduler.step()

        if epoch % args.test_freq == 0:
            print_log(save_path, '----- %s at epoch #%d' % ("Test", epoch))
            test_log = validate(test_loader, model, criterion, epoch)
            print_log(save_path, 'Player_accuracy: %.2f%%, Player_Mean-ACC: %.2f%%, Action_accuracy: %.2f%%, Action_Mean-ACC: %.2f%%, Loss: %.4f, Using %.1f seconds' %
                      (test_log['player_acc'], test_log['mean_player_acc'], test_log['action_acc'], test_log['mean_action_acc'], test_log['loss'], test_log['time']))

            print_log(save_path, '----------Best Player MCA: %.2f%% at epoch #%d.' %
                      (test_log['best_player_mca'], test_log['best_mca_epoch']))
            print_log(save_path, '----------Best Player MPCA: %.2f%% at epoch #%d.' %
                      (test_log['best_player_mpca'], test_log['best_mpca_epoch']))
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            result_path = save_path + '/epoch%d_%.2f%%.pth' % (epoch, test_log['player_acc'])
            torch.save(state, result_path)
            # if epoch == test_log['best_mca_epoch'] or epoch == test_log['best_mpca_epoch']:
            #     state = {
            #         'epoch': epoch,
            #         'state_dict': model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'scheduler': scheduler.state_dict(),
            #     }
            #     result_path = save_path + '/epoch%d_%.2f%%.pth' % (epoch, test_log['player_acc'])
            #     torch.save(state, result_path)


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    temperature = 0.2
    epoch_timer = Timer()
    losses = AverageMeter()
    player_accuracies = AverageMeter()
    action_accuracies = AverageMeter()
   # print(accuracies)

    # switch to train mode
    model.train()

    #for i, (images, activities) in enumerate(train_loader):
    for i, (videos, video_mask, players, actions) in enumerate(tqdm(train_loader, total=9684, position=0)):

        # for i, (videos, video_mask, players, actions, identity_positive_samples, identity_negative_samples, action_positive_samples, action_negative_samples) in enumerate(tqdm(train_loader, total=9684, position=0)):
    #     #print(i)

        # identity and action samples
        #print("identity_positive_samples: ", identity_positive_samples.shape)  # torch.Size([B, 2, 20, 3, 224, 224])
        #print("identity_negative_samples: ", identity_negative_samples.shape)  # torch.Size([B, 3, 20, 3, 224, 224])
        #print("action_positive_samples: ", action_positive_samples.shape)      # torch.Size([B, 2, 20, 3, 224, 224])
        #print("action_negative_samples: ", action_negative_samples.shape)      # torch.Size([B, 3, 20, 3, 224, 224])

        b_size = videos.shape[0]
        videos = videos.float().cuda()  # videos:  torch.Size([B, 20, 3, 224, 224])
        #print("videos: ", videos.shape)# [B, T, 3, H, W]
        players = players.cuda()                              # [B, T]
        actions = actions.cuda()

        num_batch = videos.shape[0]
        #print(num_batch)
        num_frame = videos.shape[1]
        players_in = players[:, 0].reshape((num_batch, ))
        #print("player_in: ", players_in)  # player_in:  tensor([108, 295], device='cuda:0') 这里batch=2
        actions_in = actions[:, 0].reshape((num_batch,)) # actions_in:  tensor([4, 5], device='cuda:0')
        #print("actions_in: ", actions_in)
        # compute output
        player_score, player_feature, action_score, action_feature = model(videos)


        # compute positive: identity and action
        # identity
        # with no_grad():
        #     positive_player_list = []
        #     for i in range(b_size):
        #         sample = identity_positive_samples[i]
        #         sample = sample.float().cuda()
        #         _, positive_player_feature, _, _ = model(sample)
        #         #print("sample: ", sample.shape)
        #         #print("positive_player_feature: ", positive_player_feature.shape)
        #         positive_player_list.append(positive_player_feature.unsqueeze(0))
        #     positive_identity_features = torch.cat(positive_player_list, dim=0)
        #     #print("positive_identity_features: ", positive_identity_features.shape)  # torch.Size([B, 2, 768])
        #
        # # action
        # with no_grad():
        #     positive_action_list = []
        #     for i in range(b_size):
        #         sample = action_positive_samples[i]
        #         sample = sample.float().cuda()
        #         _, _, _, positive_action_feature = model(sample)
        #         #print("sample: ", sample.shape)
        #         #print("positive_action_feature: ", positive_action_feature.shape)
        #         positive_action_list.append(positive_action_feature.unsqueeze(0))
        #     positive_action_features = torch.cat(positive_action_list, dim=0)
        #     #print("positive_action_features: ", positive_action_features.shape)  # torch.Size([B, 2, 768])
        #
        #
        # # compute negative: identity and action
        # # identity
        # with no_grad():
        #     negative_player_list = []
        #     for i in range(b_size):
        #         sample = identity_negative_samples[i]
        #         sample = sample.float().cuda()
        #         _, negative_player_feature, _, _ = model(sample)
        #         # print("sample: ", sample.shape)
        #         # print("negative_player_feature: ", negative_player_feature.shape)
        #         negative_player_list.append(negative_player_feature.unsqueeze(0))
        #     negative_identity_features = torch.cat(negative_player_list, dim=0)
        #     #print("negative_identity_features: ", negative_identity_features.shape)  # torch.Size([B, 3, 768])
        #
        # # action
        # with no_grad():
        #     negative_action_list = []
        #     for i in range(b_size):
        #         sample = action_negative_samples[i]
        #         sample = sample.float().cuda()
        #         _, _, _, negative_action_feature = model(sample)
        #         # print("sample: ", sample.shape)
        #         # print("negative_action_feature: ", negative_action_feature.shape)
        #         negative_action_list.append(negative_action_feature.unsqueeze(0))
        #     negative_action_features = torch.cat(negative_action_list, dim=0)
        #     # print("negative_action_features: ", negative_action_features.shape)  # torch.Size([B, 3, 768])

        # compute InfoNCE loss
        # anchor_identity_features = player_feature.unsqueeze(1)
        #print("anchor_identity_features: ", anchor_identity_features.shape)
        # anchor_action_features = action_feature.unsqueeze(1)
        #print("anchor_action_features: ", anchor_action_features.shape)  # torch.Size([B, 1, 768])

        # positive_sim_identity = F.cosine_similarity(anchor_identity_features, positive_identity_features, dim=-1)
        # negative_sim_identity = F.cosine_similarity(anchor_identity_features, negative_identity_features, dim=-1)
        # logits_identity = torch.cat([positive_sim_identity, negative_sim_identity], dim=1)
        # #print(logits_identity.shape)  # torch.Size([2, 5])
        # labels_identity = torch.cat([
        #     torch.ones(2, dtype=torch.float32, device=anchor_identity_features.device),
        #     torch.zeros(3, dtype=torch.float32, device=negative_identity_features.device)
        # ]).unsqueeze(0).repeat(b_size,1)#.long()
        # #print(labels_identity.size())  # torch.Size([5])
        # logits_identity = logits_identity / temperature
        # loss_identity_contra = F.cross_entropy(logits_identity, labels_identity)
        #
        # positive_sim_action = F.cosine_similarity(anchor_action_features, positive_action_features, dim=-1)
        # negative_sim_action = F.cosine_similarity(anchor_action_features, negative_action_features, dim=-1)
        # logits_action = torch.cat([positive_sim_action, negative_sim_action], dim=1)
        # labels_action = torch.cat([
        #     torch.ones(2, dtype=torch.float32, device=anchor_identity_features.device),
        #     torch.zeros(3, dtype=torch.float32, device=negative_identity_features.device)
        # ]).unsqueeze(0).repeat(b_size,1)#.long()
        # logits_action = logits_action / temperature
        # loss_action_contra = F.cross_entropy(logits_action, labels_action)

        # print("player_feature: ", player_feature.shape)  # torch.Size([2, 768])
        # print("action_feature: ", action_feature.shape)  # torch.Size([2, 768])
        #print("player_score: ", player_score.shape)
        #print("action_score: ", action_score.shape)  #
        #print(score)# [B, C]

        # calculate loss
        loss1 = criterion(player_score, players_in)
        loss2 = criterion(action_score, actions_in)

        loss = loss1 + loss2

        #loss += loss / 4

        #print(loss)

        # measure accuracy and record loss
        player_acc = accuracy(player_score, players_in)
        action_acc = accuracy(action_score, actions_in)
        losses.update(loss, num_batch)
        # 加的内容： loss = loss / accumulation_steps
        #loss = loss / accumulation_steps
        player_accuracies.update(player_acc, num_batch)
        action_accuracies.update(action_acc, num_batch)

        #loss.backward()

        # compute gradient and do SGD step
        # 加的内容：
        #if ((i+1) % 4) == 0:
        optimizer.zero_grad()
        loss.backward()
        if args.gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()

    train_log = {
        'epoch': epoch,
        'time': epoch_timer.timeit(),
        'loss': losses.avg,
        'player_acc': player_accuracies.avg * 100.0,
        'action_acc': action_accuracies.avg * 100.0
    }

    return train_log


@torch.no_grad()
def validate(test_loader, model, criterion, epoch):
    global best_player_mca, best_player_mpca, best_action_mca, best_action_mpca, best_mca_epoch, best_mpca_epoch
    epoch_timer = Timer()
    losses = AverageMeter()
    player_accuracies = AverageMeter()
    action_accuracies = AverageMeter()
    player_true = []
    player_pred = []
    action_true = []
    action_pred = []

    # switch to eval mode
    model.eval()

    #for i, (images, activities) in enumerate(test_loader):
    for i, (videos, video_mask, players, actions) in enumerate(tqdm(test_loader, total=2570, position=0)):
        videos = videos.float().cuda()
        # print("videos: ", videos.shape)# [B, T, 3, H, W]
        players = players.cuda()  # [B, T]
        actions = actions.cuda()  # [B, T]

        num_batch = videos.shape[0]
        # print(num_batch)
        num_frame = videos.shape[1]
        players_in = players[:, 0].reshape((num_batch,))
        actions_in = actions[:, 0].reshape((num_batch,))

        # compute output
        player_score, _, action_score, _ = model(videos)

        # player
        player_true = player_true + players_in.tolist()
        player_pred = player_pred + torch.argmax(player_score, dim=1).tolist()

        # action
        action_true = action_true + actions_in.tolist()
        action_pred = action_pred + torch.argmax(action_score, dim=1).tolist()


        # calculate loss
        loss1 = criterion(player_score, players_in)
        loss2 = criterion(action_score, actions_in)
        loss = loss1 + loss2

        # measure accuracy and record loss
        player_acc = accuracy(player_score, players_in)
        action_acc = accuracy(action_score, actions_in)
        losses.update(loss, num_batch)
        player_accuracies.update(player_acc, num_batch)
        action_accuracies.update(action_acc, num_batch)

    # print("player_true: ", player_true)
    # print("player_pred: ", player_pred)
    # print("action_true: ", action_true)
    # print("action_pred: ", action_pred)

    player_acc = player_accuracies.avg * 100.0
    player_confusion = confusion_matrix(player_true, player_pred)
    player_mean_acc = np.mean([player_confusion[i, i] / player_confusion[i, :].sum() for i in range(player_confusion.shape[0])]) * 100.0

    action_acc = action_accuracies.avg * 100.0
    action_confusion = confusion_matrix(action_true, action_pred)
    action_mean_acc = np.mean([action_confusion[i, i] / action_confusion[i, :].sum() for i in range(action_confusion.shape[0])]) * 100.0

    # player
    if player_acc > best_player_mca:
        best_player_mca = player_acc
        best_mca_epoch = epoch
    if player_mean_acc > best_player_mpca:
        best_player_mpca = player_mean_acc
        best_mpca_epoch = epoch

    test_log = {
        'time': epoch_timer.timeit(),
        'loss': losses.avg,
        'player_acc': player_acc,
        'action_acc': action_acc,
        'mean_player_acc': player_mean_acc,
        'mean_action_acc': action_mean_acc,
        'best_player_mca': best_player_mca,
        'best_player_mpca': best_player_mpca,
        'best_mca_epoch': best_mca_epoch,
        'best_mpca_epoch': best_mpca_epoch,
    }

    return test_log


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """
    class to do timekeeping
    """
    def __init__(self):
        self.last_time = time.time()

    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time


def accuracy(output, target):
    output = torch.argmax(output, dim=1)
    # print(target.shape)
    # print(output.shape)
    #print(output.shape[0])
    correct = torch.sum(torch.eq(target, output)).float()
    # print("correct", correct)
    # print("fenmu", output.shape[0])
    return correct.item() / output.shape[0]


if __name__ == '__main__':
    main()


