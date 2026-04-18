# 这个工具文件承载训练过程中的通用辅助函数，供实验流程和结果可视化共用。

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


# 调整训练过程中的学习率。
def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == 'unchanged':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


# 提前停止训练并保存当前最优模型。
class EarlyStopping:
    # 初始化提前停止所需的状态。
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    # 根据验证集损失决定是否保存模型。
    def __call__(self, val_loss, model, path):
        self.save_checkpoint(val_loss, model, path)
        '''
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
        '''

    # 保存当前模型参数到检查点文件。
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


# 提供可通过点号访问的字典包装。
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# 提供标准化与反标准化的数值变换。
class StandardScaler():
    # 初始化标准化器的均值和标准差。
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    # 将输入数据转换为标准化结果。
    def transform(self, data):
        return (data - self.mean) / self.std

    # 将标准化后的数据还原回原始尺度。
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# 绘制并保存预测结果对比图。
def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        # 先画预测，再画真值，保证重合区间由 GroundTruth 覆盖 Prediction。
        plt.plot(preds, label='Prediction', linewidth=2, color='tab:blue', zorder=2)
        plt.plot(true, label='GroundTruth', linewidth=2, color='tab:orange', zorder=3)
    else:
        plt.plot(true, label='GroundTruth', linewidth=2, color='tab:orange', zorder=3)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


# 对异常检测的预测结果做连通性修正。
def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


# 计算分类准确率。
def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
