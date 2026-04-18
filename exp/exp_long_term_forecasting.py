"""这个实验文件负责 TimeFilter 的长期预测训练、验证和测试流程，并与 `models/TimeFilter.py`、`data_provider/data_factory.py`、`utils/tools.py` 配合工作。"""

import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.dtw_metric import accelerated_dtw
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings('ignore')


# 长期预测实验类，负责完整的训练、验证和测试流程。
class Exp_Long_Term_Forecast(Exp_Basic):
    """封装 TimeFilter 长期预测实验流程。"""

    # 初始化实验对象并提前构建图掩码。
    def __init__(self, args):
        """初始化实验对象和 TimeFilter 图掩码。"""
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.masks = self._get_mask()

    # 构建当前实验所需的模型。
    def _build_model(self):
        """根据参数构建模型实例。"""
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # 读取指定阶段的数据集和数据加载器。
    def _get_data(self, flag):
        """获取训练、验证或测试数据。"""
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # 选择优化器。
    def _select_optimizer(self):
        """创建 Adam 优化器。"""
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # 选择损失函数。
    def _select_criterion(self):
        """返回默认的 MSE 损失函数。"""
        criterion = nn.MSELoss()
        return criterion

    # 构建 TimeFilter 使用的区域掩码。
    def _get_mask(self):
        """构建节点与 patch 级别的静态分区掩码。"""
        dtype = torch.float32
        total_tokens = self.args.seq_len * self.args.c_out // self.args.patch_len
        patch_count = self.args.seq_len // self.args.patch_len
        masks = []
        for index in range(total_tokens):
            same_node = ((torch.arange(total_tokens) % patch_count == index % patch_count)
                         & (torch.arange(total_tokens) != index)).to(dtype).to(self.device)
            same_patch = ((torch.arange(total_tokens) >= index // patch_count * patch_count)
                          & (torch.arange(total_tokens) < index // patch_count * patch_count + patch_count)
                          & (torch.arange(total_tokens) != index)).to(dtype).to(self.device)
            others = torch.ones(total_tokens, dtype=dtype, device=self.device) - same_node - same_patch
            others[index] = 0.0
            masks.append(torch.stack([same_node, same_patch, others], dim=0))
        masks = torch.stack(masks, dim=0)
        return masks

    # 兼容普通 batch 与带额外输入字典的 batch。
    def _unpack_batch(self, batch):
        """统一拆分 DataLoader 返回的 batch。"""
        if not isinstance(batch, (list, tuple)):
            raise ValueError(f'期望 DataLoader 返回 list/tuple，实际为 {type(batch)}')
        if len(batch) == 4:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            extra_inputs = None
        elif len(batch) == 5:
            batch_x, batch_y, batch_x_mark, batch_y_mark, extra_inputs = batch
        else:
            raise ValueError(f'不支持的 batch 长度: {len(batch)}')
        return batch_x, batch_y, batch_x_mark, batch_y_mark, extra_inputs

    # 把额外输入字典里的张量搬到当前设备。
    def _move_extra_inputs(self, extra_inputs):
        """递归地把天气和爆破侧信息搬到训练设备。"""
        if extra_inputs is None:
            return None
        moved_inputs = {}
        for key, value in extra_inputs.items():
            if torch.is_tensor(value):
                moved_inputs[key] = value.float().to(self.device)
            else:
                moved_inputs[key] = value
        return moved_inputs

    # 统一执行 TimeFilter 前向过程，兼容 AMP 和 moe_loss 返回值。
    def _forward_model(self, batch_x, is_training, extra_inputs=None):
        """执行模型前向并返回预测结果和 moe_loss。"""
        with torch.cuda.amp.autocast(enabled=self.args.use_amp and self.args.use_gpu):
            outputs, moe_loss = self.model(
                batch_x,
                self.masks,
                is_training=is_training,
                extra_inputs=extra_inputs,
            )
        return outputs, moe_loss

    # 截取当前任务真正用于监督学习的预测区间。
    def _slice_prediction(self, outputs, batch_y):
        """按 features 配置切出预测部分。"""
        feature_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, feature_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, feature_dim:]
        return outputs, batch_y

    # 对预测结果做反归一化，兼容任意 batch 大小。
    def _maybe_inverse_transform(self, dataset, array_data):
        """在需要时对输出做反归一化。"""
        if not (dataset.scale and self.args.inverse):
            return array_data

        original_shape = array_data.shape
        flat_array = array_data.reshape(original_shape[0] * original_shape[1], -1)
        restored = dataset.inverse_transform(flat_array).reshape(original_shape)
        return restored

    # 在验证集上评估当前模型。
    def vali(self, vali_data, vali_loader, criterion):
        """执行验证流程并返回平均损失。"""
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch in vali_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, extra_inputs = self._unpack_batch(batch)
                del batch_x_mark, batch_y_mark
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                extra_inputs = self._move_extra_inputs(extra_inputs)

                outputs, moe_loss = self._forward_model(batch_x, is_training=False, extra_inputs=extra_inputs)
                outputs, batch_y = self._slice_prediction(outputs, batch_y)
                loss = criterion(outputs, batch_y) + 0.05 * moe_loss
                total_loss.append(loss.detach().item())

        total_loss = float(np.average(total_loss)) if total_loss else 0.0
        self.model.train()
        return total_loss

    # 执行完整训练流程并保存最佳模型。
    def train(self, setting):
        """执行训练、验证、测试和早停逻辑。"""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        checkpoint_path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp and self.args.use_gpu)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for batch_index, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, extra_inputs = self._unpack_batch(batch)
                del batch_x_mark, batch_y_mark
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                extra_inputs = self._move_extra_inputs(extra_inputs)

                outputs, moe_loss = self._forward_model(batch_x, is_training=True, extra_inputs=extra_inputs)
                outputs, batch_y = self._slice_prediction(outputs, batch_y)
                loss = criterion(outputs, batch_y) + 0.05 * moe_loss
                train_loss.append(loss.detach().item())

                if (batch_index + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(batch_index + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - batch_index)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp and self.args.use_gpu:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = float(np.average(train_loss)) if train_loss else 0.0
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(checkpoint_path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    # 在测试集上评估模型并保存结果文件。
    def test(self, setting, test=0):
        """执行测试流程并保存预测结果和评估指标。"""
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            best_model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        preds = []
        trues = []
        inputs = []

        test_result_path = os.path.join('./test_results', setting)
        if not os.path.exists(test_result_path):
            os.makedirs(test_result_path)

        self.model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, extra_inputs = self._unpack_batch(batch)
                del batch_x_mark, batch_y_mark
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                extra_inputs = self._move_extra_inputs(extra_inputs)

                outputs, _ = self._forward_model(batch_x, is_training=False, extra_inputs=extra_inputs)
                outputs, batch_y = self._slice_prediction(outputs, batch_y)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x_np = batch_x.detach().cpu().numpy()

                outputs = self._maybe_inverse_transform(test_data, outputs)
                batch_y = self._maybe_inverse_transform(test_data, batch_y)
                batch_x_np = self._maybe_inverse_transform(test_data, batch_x_np)

                preds.append(outputs)
                trues.append(batch_y)
                inputs.append(batch_x_np)

                if batch_index % 20 == 0:
                    gt = np.concatenate((batch_x_np[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pred = np.concatenate((batch_x_np[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pred, os.path.join(test_result_path, str(batch_index) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputs = np.concatenate(inputs, axis=0)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        result_path = os.path.join('./results', setting)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for index in range(preds.shape[0]):
                pred_item = preds[index].reshape(-1, 1)
                true_item = trues[index].reshape(-1, 1)
                if index % 100 == 0:
                    print("calculating dtw iter:", index)
                distance, _, _, _ = accelerated_dtw(pred_item, true_item, dist=manhattan_distance)
                dtw_value = distance
                dtw_list.append(dtw_value)
            dtw_score = float(np.array(dtw_list).mean()) if dtw_list else 0.0
        else:
            dtw_score = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw_score))
        with open("result_long_term_forecast.txt", 'a') as result_file:
            result_file.write(setting + "  \n")
            result_file.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw_score))
            result_file.write('\n\n')

        np.save(os.path.join(result_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(result_path, 'input.npy'), inputs)
        np.save(os.path.join(result_path, 'pred.npy'), preds)
        np.save(os.path.join(result_path, 'true.npy'), trues)

        return
