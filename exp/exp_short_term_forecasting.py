"""这个实验文件负责短期预测任务的训练、验证和测试流程，并兼容 M4 评估逻辑与当前 TimeFilter 的直接预测接口。"""

import os
import time
import warnings

import numpy as np
import pandas
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings('ignore')


# 短期预测实验类，保留 M4 专用评估流程，并适配 TimeFilter 接口。
class Exp_Short_Term_Forecast(Exp_Basic):
    """封装短期预测任务的训练、验证和测试流程。"""

    # 初始化实验对象，并在 TimeFilter 场景下提前构建图掩码。
    def __init__(self, args):
        """初始化实验对象和可选的 TimeFilter 掩码。"""
        super(Exp_Short_Term_Forecast, self).__init__(args)
        self.masks = self._get_mask() if self.args.model == 'TimeFilter' else None

    # 构建短期预测模型。
    def _build_model(self):
        """根据数据集配置构建模型。"""
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]
            self.args.seq_len = 2 * self.args.pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # 获取指定阶段的数据。
    def _get_data(self, flag):
        """返回数据集与数据加载器。"""
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # 选择优化器。
    def _select_optimizer(self):
        """创建 Adam 优化器。"""
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # 选择短期预测的损失函数。
    def _select_criterion(self, loss_name='MSE'):
        """根据损失名称返回对应损失函数。"""
        if loss_name == 'MSE':
            return nn.MSELoss()
        if loss_name == 'MAPE':
            return mape_loss()
        if loss_name == 'MASE':
            return mase_loss()
        if loss_name == 'SMAPE':
            return smape_loss()
        raise ValueError(f'不支持的短期预测损失函数: {loss_name}')

    # 构建 TimeFilter 需要的静态区域掩码。
    def _get_mask(self):
        """为 TimeFilter 构建节点和 patch 级别的掩码。"""
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
        return torch.stack(masks, dim=0)

    # 按模型接口执行一次前向推理。
    def _forward_model(self, batch_x, batch_y=None, is_training=False):
        """兼容直接预测模型与 encoder-decoder 模型的前向调用。"""
        if self.args.model == 'TimeFilter':
            with torch.cuda.amp.autocast(enabled=self.args.use_amp and self.args.use_gpu):
                outputs, moe_loss = self.model(batch_x, self.masks, is_training=is_training)
            return outputs, moe_loss

        if batch_y is None:
            raise ValueError('encoder-decoder 短期预测模型需要 batch_y 来构建解码器输入。')

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        with torch.cuda.amp.autocast(enabled=self.args.use_amp and self.args.use_gpu):
            outputs = self.model(batch_x, None, dec_inp, None)
        return outputs, None

    # 根据 features 设置截取预测结果。
    def _slice_prediction(self, outputs, batch_y):
        """截取预测窗口和目标窗口。"""
        feature_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, feature_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, feature_dim:]
        return outputs, batch_y

    # 截取短期预测损失需要的时间标记。
    def _slice_prediction_mark(self, batch_y_mark):
        """按 features 配置切出目标时间标记。"""
        feature_dim = -1 if self.args.features == 'MS' else 0
        batch_y_mark = batch_y_mark[:, -self.args.pred_len:, feature_dim:]
        return batch_y_mark

    # 统一计算短期预测损失，兼容 MSE 和 M4 专用损失。
    def _compute_loss(self, criterion, batch_x, outputs, batch_y, batch_y_mark):
        """根据损失函数类型计算训练或验证损失。"""
        if isinstance(criterion, nn.MSELoss):
            return criterion(outputs, batch_y)

        insample = batch_x[:, :, 0] if batch_x.dim() == 3 else batch_x
        forecast = outputs.squeeze(-1) if outputs.dim() == 3 and outputs.shape[-1] == 1 else outputs
        target = batch_y.squeeze(-1) if batch_y.dim() == 3 and batch_y.shape[-1] == 1 else batch_y
        mask = batch_y_mark.squeeze(-1) if batch_y_mark.dim() == 3 and batch_y_mark.shape[-1] == 1 else batch_y_mark
        return criterion(insample, self.args.frequency_map, forecast, target, mask)

    # 对长批次验证/测试数据做分段预测，避免一次性占满显存。
    def _forecast_last_window(self, x, stride):
        """对 last_insample_window 做分段预测。"""
        batch_size, _, channel_count = x.shape
        outputs = torch.zeros((batch_size, self.args.pred_len, channel_count), dtype=torch.float32, device=self.device)
        id_list = np.arange(0, batch_size, stride)
        id_list = np.append(id_list, batch_size)
        for index in range(len(id_list) - 1):
            left = id_list[index]
            right = id_list[index + 1]
            batch_x = x[left:right]
            if self.args.model == 'TimeFilter':
                outputs[left:right], _ = self._forward_model(batch_x, is_training=False)
            else:
                dec_inp = torch.zeros((right - left, self.args.pred_len, channel_count), dtype=torch.float32, device=self.device)
                dec_inp = torch.cat([batch_x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
                with torch.cuda.amp.autocast(enabled=self.args.use_amp and self.args.use_gpu):
                    outputs[left:right] = self.model(batch_x, None, dec_inp, None)
        return outputs

    # 在验证集上评估模型。
    def vali(self, train_loader, vali_loader, criterion):
        """执行短期预测验证。"""
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device).unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            outputs = self._forecast_last_window(x, stride=500)
            feature_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, feature_dim:]
            pred = outputs
            true = torch.from_numpy(np.array(y)).to(self.device)
            batch_y_mark = torch.ones(true.shape, device=self.device)
            loss = self._compute_loss(criterion, x, pred, true, batch_y_mark)

        self.model.train()
        return loss

    # 执行短期预测训练。
    def train(self, setting):
        """执行训练、验证和早停逻辑。"""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        checkpoint_path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp and self.args.use_gpu)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                del batch_x_mark
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, moe_loss = self._forward_model(batch_x, batch_y=batch_y, is_training=True)
                outputs, batch_y = self._slice_prediction(outputs, batch_y)
                batch_y_mark = self._slice_prediction_mark(batch_y_mark).to(self.device)

                loss = self._compute_loss(criterion, batch_x, outputs, batch_y, batch_y_mark)
                if moe_loss is not None:
                    loss = loss + 0.05 * moe_loss
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
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
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

    # 执行短期预测测试并输出 M4 结果文件。
    def test(self, setting, test=0):
        """执行测试流程并生成 M4 结果。"""
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device).unsqueeze(-1)

        if test:
            print('loading model')
            best_model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        test_result_path = os.path.join('./test_results', setting)
        if not os.path.exists(test_result_path):
            os.makedirs(test_result_path)

        self.model.eval()
        with torch.no_grad():
            outputs = self._forecast_last_window(x, stride=1)
            feature_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, feature_dim:]
            preds = outputs.detach().cpu().numpy()
            trues = y
            x_np = x.detach().cpu().numpy()

            plot_step = max(1, preds.shape[0] // 10)
            for index in range(0, preds.shape[0], plot_step):
                gt = np.concatenate((x_np[index, :, 0], trues[index]), axis=0)
                pred = np.concatenate((x_np[index, :, 0], preds[index, :, 0]), axis=0)
                visual(gt, pred, os.path.join(test_result_path, str(index) + '.pdf'))

        print('test shape:', preds.shape)

        m4_result_path = os.path.join('./m4_results', self.args.model)
        if not os.path.exists(m4_result_path):
            os.makedirs(m4_result_path)

        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{index + 1}' for index in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.to_csv(os.path.join(m4_result_path, self.args.seasonal_patterns + '_forecast.csv'))

        print(self.args.model)
        if 'Weekly_forecast.csv' in os.listdir(m4_result_path) \
                and 'Monthly_forecast.csv' in os.listdir(m4_result_path) \
                and 'Yearly_forecast.csv' in os.listdir(m4_result_path) \
                and 'Daily_forecast.csv' in os.listdir(m4_result_path) \
                and 'Hourly_forecast.csv' in os.listdir(m4_result_path) \
                and 'Quarterly_forecast.csv' in os.listdir(m4_result_path):
            m4_summary = M4Summary(m4_result_path, self.args.root_path)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            print('smape:', smape_results)
            print('mape:', mape)
            print('mase:', mase)
            print('owa:', owa_results)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return
