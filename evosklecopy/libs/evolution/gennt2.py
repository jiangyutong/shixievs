import torch.nn as nn
import numpy as np
import os
from torch.autograd import Variable
import libs.parser.parse as parse
import datetime
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
from torch.utils.data import  Dataset,DataLoader,TensorDataset
from libs.evolution.parameter import parse_arg
def show3Dpose(channels,
               ax,
               lcolor="#3498db",
               rcolor="#e74c3c",
               add_labels=True,
               gt=False,
               pred=False,
               plot_dot=False
               ):  # blue, orange
    """
    Visualize a 3d skeleton

    Args
      channels: 96x1 vector. The pose to plot.
      ax: matplotlib 3d axis to draw on
      lcolor: color for left part of the body
      rcolor: color for right part of the body
      add_labels: whether to add coordinate labels
    Returns
      Nothing. Draws on ax.
    """

    if channels.shape[0] == 96:
        vals = np.reshape(channels, (32, -1))
    else:
        vals = channels
    I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1  # start points
    J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1  # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    dim_use_3d = [3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25,
                  26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53,
                  54, 55, 56, 57, 58, 59, 75, 76, 77, 78, 79, 80, 81, 82, 83]
    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        if gt:
            ax.plot(x, y, z, lw=4, c='k')
        #        ax.plot(x,y, z,  lw=2, c='k')

        elif pred:
            ax.plot(x, z, -y, lw=4, c='r')
        #        ax.plot(x,y, z,  lw=2, c='r')

        else:
            #        ax.plot(x,z, -y,  lw=2, c=lcolor if LR[i] else rcolor)
            ax.plot(x, y, z, lw=4, c=lcolor if LR[i] else rcolor)
            # ax.text(x[0],y[0],z[0],I[i])
            # ax.text(x[1],y[1],z[1],J[i])
    if plot_dot:
        joints = channels.reshape(96)
        joints = joints[dim_use_3d].reshape(16, 3)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='k', marker='o')
    RADIUS = 400  # space around the subject
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

#     if add_labels:
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")
#         ax.set_zlabel("z")

    ax.set_aspect('auto')
    ax.view_init(90, 90)
    plt.show()
def cast_to_float(dic, dtype=np.float32):
    # cast to float 32 for space saving
    for key in dic.keys():
        dic[key] = dic[key].detach().cpu().numpy()
        dic[key] = dic[key].astype(dtype)
    return dic



# ???????????????  #####Discriminator######????????????????????????????????????
# ?????????28x28?????????784????????????????????????????????????????????????????????????0.2???LeakyReLU???????????????
# ?????????sigmoid????????????????????????0???1?????????????????????????????????
class discriminator(nn.Module):
    def __init__(self):

        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(96, 256),  # ??????????????????784????????????256
            nn.LeakyReLU(0.2),  # ?????????????????????
            nn.Linear(256, 256),  # ????????????????????????
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # ????????????????????????????????????????????????
            # sigmoid???????????????????????????0,1????????????????????????
            # ????????????softmax??????
        )

    def forward(self, x):
        x = self.dis(x)
        return x


# ###### ??????????????? Generator #####
# ????????????100??????0???1????????????????????????????????????????????????????????????????????????256???,
# ????????????LeakyReLU???????????????????????????????????????????????????????????????LeakyReLU???????????????
# ????????????????????????????????????784??????????????????Tanh??????????????????????????????????????????????????????
# ?????????-1???1?????????
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.gen = nn.Sequential(
            nn.Linear(96, 256),  # ?????????????????????????????????256???
            nn.ReLU(True),  # relu??????
            nn.Linear(256, 256),  # ????????????
            nn.ReLU(True),  # relu??????
            nn.Linear(256, 96),  # ????????????
            # nn.Tanh()  # Tanh????????????????????????????????????-1,1????????????????????????????????????????????????transforms????????????????????????
        )

    def forward(self, x):
        x = self.gen(x)
        return x
class trainset(Dataset):
    def __init__(self, data):
        #????????? image ?????????
        self.target = data
    def __getitem__(self, index):
        target = self.target[index]
        return target

    def __len__(self):
        return self.target.shape[0]

from libs.skeleton.anglelimits import \
to_local, to_global, get_skeleton, to_spherical, \
nt_parent_indices, nt_child_indices, \
is_valid_local, is_valid
nt_parent_indices = [13, 17, 18, 25, 26, 6, 7, 1, 2, 13]
nt_child_indices = [15, 18, 19, 26, 27, 7, 8, 2, 3, 14]
root = "../../resources/constraints"
template = np.load(os.path.join(root, 'template.npy'), allow_pickle=True).reshape(32,-1)
template_bones = template[nt_parent_indices, :] - template[nt_child_indices, :]
template_bone_lengths = to_spherical(template_bones)[:, 0]
nt_parent_indices = [13, 17, 18, 25, 26, 6, 7, 1, 2]
nt_child_indices = [15, 18, 19, 26, 27, 7, 8, 2, 3]
def re_order(skeleton):
    # the ordering of coordinate used by the Prior was x,z and y
    return skeleton[:, [0,2,1]]
def get_bone_length(skeleton):
    """
    Compute limb length for a given skeleton.
    """
    bones = skeleton[nt_parent_indices, :] - skeleton[nt_child_indices, :]
    bone_lengths = to_spherical(bones)[:, 0]
    return bone_lengths
def modify_pose(skeleton, local_bones, bone_length, ro=False):
    # get a new pose by modify an existing pose with input local bone vectors
    # and bone lengths
    new_bones = to_global(skeleton, local_bones)['bg']
    new_pose = get_skeleton(new_bones, skeleton, bone_length=bone_length)
    if ro:
        new_pose = re_order(new_pose)
    return new_pose.reshape(-1)
def set_z(pose, target):
    if pose is None:
        return None
    original_shape = pose.shape
    pose = pose.reshape(32, 3)
    min_val = pose[:, 2].min()
    pose[:, 2] -= min_val - target
    return pose.reshape(original_shape)

if __name__ == '__main__':

    i=0
    j=0
    post_processing = True
    finaldatakey_vail={}
    opt = parse_arg()
    dict_path = os.path.join('../../data/human3.6M/', 'threeDPose_train.npy')
    G = torch.load('/media/zlz422/82BEDFE2BEDFCD33/jyt/project/EvSkeleton/data/human3.6M/2021-10-23 06:53:59.623275_gennet/G')
    G =G.cuda()
    train_dict_3d = np.load(dict_path, allow_pickle=True).item()
    finaldatakey = {}
    # for key in train_dict_3d.keys():
    #     finaldatakey[key] = train_dict_3d[key]
    for key in train_dict_3d.keys():
        offall=[]
        for i in range(0,train_dict_3d[key].shape[0],256):
            datatemp=train_dict_3d[key][i:i+2456]
            mean = np.mean(datatemp,axis=0)
            mean = np.mean(mean,axis=0)
            std = np.std(datatemp,axis=0)
            std = np.std(std,axis=0)
            z=torch.normal(mean=mean,std=std,size=(256,96))
            z = Variable(z).cuda()
            fake_data = G(z)  # ????????????????????????????????????????????????????????????
            offsprings_r = fake_data.detach().cpu().numpy()
            # np.save("/media/zlz422/82BEDFE2BEDFCD33/jyt/project/EvSkeleton/data/jyt.npy",offsprings)
            tmpLeng = len(offsprings_r)
            tmpi = 0
            for m in range(tmpLeng):
                offsprings_r_tmpi = offsprings_r[tmpi]
                offsprings_r_tmpi = offsprings_r_tmpi.reshape(-1, 3)
                offsprings_r_tmpi = offsprings_r_tmpi
                offsprings_r_tmpi = re_order(offsprings_r_tmpi)
                offsprings_r_tmpi_length = get_bone_length(offsprings_r_tmpi)
                bones_offsprings_r_tmpi = to_local(offsprings_r_tmpi)
                if opt.C:
                    # apply joint angle constraint as the fitness function
                    valid_vec_fa = is_valid_local(bones_offsprings_r_tmpi)
                if not opt.C or valid_vec_fa.sum() >= 9:
                    offsprings_r_tmpi = modify_pose(offsprings_r_tmpi, bones_offsprings_r_tmpi,
                                                    offsprings_r_tmpi_length,
                                                    ro=True)
                    if post_processing:
                        # move the poses to the ground plane
                        set_z(offsprings_r_tmpi, np.random.normal(loc=20.0, scale=3.0))
                    offsprings_r[tmpi] = offsprings_r_tmpi
                    offall.append(offsprings_r_tmpi)
                    ax1 = plt.subplot(1, 1, 1, projection='3d')
                    ax1.grid(False)
                    plt.axis('off')
                    show3Dpose(offsprings_r_tmpi, ax1, add_labels=False, plot_dot=True)
                else:
                    offsprings_r = np.delete(offsprings_r, tmpi, 0)
                    tmpi = tmpi - 1
                    tmpLeng = tmpLeng - 1
                tmpi = tmpi + 1
        keynew = []
        keynew.append(key[0])
        keynew.append(key[1])
        keynew2 = key[2] + "new"
        keynew.append(keynew2)
        keynew = tuple(keynew)
        finaldatakey_vail[keynew] = offall
    # save_path = "/media/zlz422/82BEDFE2BEDFCD33/jyt/project/EvSkeleton/data/human3.6M/jyt_gennet_random"
    # np.save(save_path,finaldatakey)
    save_path = "/media/zlz422/82BEDFE2BEDFCD33/jyt/project/EvSkeleton/data/human3.6M/jyt_gennet_random_vail_getfromG"
    np.save(save_path,finaldatakey_vail)