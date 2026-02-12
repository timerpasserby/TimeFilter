import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.signal import stft       

np.random.seed(42)
torch.manual_seed(42)

# 1) 数据：多分量 + 外生变量 + 缺失值 + 异常点
def generate_virtual_series(T=2000, num_exog=3):
    t = np.arange(T)
    # 趋势（分段线性）
    trend = 0.0008 * t
    trend[t > 1200] += 0.3# 结构变化跳升

    # 多季节：短周期 & 长周期
    seasonal_short = 0.8 * np.sin(2 * np.pi * t / 14.0 + 0.5)  # 每两周
    seasonal_long = 0.4 * np.cos(2 * np.pi * t / 60.0)         # 每两个月
    # 周期强度随时间变化（时变季节）
    seasonal_envelope = 1.0 + 0.5 * np.sin(2 * np.pi * t / 500.0)
    seasonal = seasonal_envelope * (seasonal_short + seasonal_long)

    # 噪声
    noise = 0.3 * np.random.randn(T)

    # 偶发尖峰
    spikes_idx = np.random.choice(np.arange(100, T-100), size=20, replace=False)
    spikes = np.zeros(T)
    spikes[spikes_idx] = np.random.choice([2.5, -2.0, 3.0, -3.5], size=20)

    # 目标序列
    y = 1.5 + trend + seasonal + noise + spikes

    # 外生变量构造（不同模式）
    exogs = []
    # exog1: 温度样式（季节+噪声）
    exog1 = 15 + 8 * np.sin(2 * np.pi * t / 365.0) + 1.5*np.random.randn(T)
    # exog2: 营销指数样式（断续高频）
    exog2 = 50 + 10 * (np.sin(2 * np.pi * t / 7.0) > 0).astype(float) + 5*np.random.randn(T)
    # exog3: 需求扰动（多分量）
    exog3 = 5*np.sin(2 * np.pi * t / 30.0 + 1.0) + 3*np.cos(2 * np.pi * t / 120.0) + 2*np.random.randn(T)
    exogs = np.stack([exog1, exog2, exog3], axis=1)[:,:num_exog]

    # 引入缺失值并线性插值
    miss_idx = np.random.choice(np.arange(T), size=80, replace=False)
    y_with_nan = y.copy()
    y_with_nan[miss_idx] = np.nan
    # 简易插值
    def linear_interpolate(arr):
        nans = np.isnan(arr)
        x = np.arange(arr.shape[0])
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
        return arr
    y = linear_interpolate(y_with_nan)

    # 标准化（训练更稳）
    y_mean, y_std = np.mean(y), np.std(y)
    y_norm = (y - y_mean) / (y_std + 1e-8)
    exog_mean = np.mean(exogs, axis=0, keepdims=True)
    exog_std = np.std(exogs, axis=0, keepdims=True)
    exogs_norm = (exogs - exog_mean) / (exog_std + 1e-8)

    data = np.concatenate([y_norm[:, None], exogs_norm], axis=1)
    return data, y_norm, (y_mean, y_std), exogs_norm, t

# 2) 序列窗口化
class SeqDataset(Dataset):
    def __init__(self, data, lookback=128, horizon=24):
        self.data = data  # shape [T, F], F = 1 + num_exog
        self.lookback = lookback
        self.horizon = horizon
        self.T = data.shape[0]
        self.F = data.shape[1]
        self.indices = []
        for s in range(self.T - lookback - horizon):
            self.indices.append(s)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        s = self.indices[idx]
        x = self.data[s:s+self.lookback, :]      # [L, F]
        y = self.data[s+self.lookback:s+self.lookback+self.horizon, 0]  # 仅预测目标维度
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# 3) 模型定义：CNN（多尺度） + TCN（膨胀卷积） + LSTM（融合）
class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3,5,7,11)):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            pad = (k - 1) // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=pad),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels)
                )
            )
    def forward(self, x):# x: [B, C_in, T]
        outs = [branch(x) for branch in self.branches]  # list of [B, C_out, T]
        return torch.cat(outs, dim=1)  # [B, C_out*len(kernel_sizes), T]

class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.resample = None
        if in_channels != out_channels:
            self.resample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        out = self.conv(x)
        # 因果裁剪：保证不看未来
        out = out[:, :, :-self.conv.padding[0]] if self.conv.padding[0] > 0 else out
        out = self.relu(self.norm(out))
        out = self.dropout(out)
        res = x if self.resample is None else self.resample(x)
        res = res[:, :, :out.size(-1)]
        return out + res

class TCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=4, kernel_size=3, dilations=(1,2,4,8), dropout=0.1):
        super().__init__()
        layers = []
        c_in = in_channels
        for i in range(num_layers):
            d = dilations[i] if i < len(dilations) else 2**i
            layers.append(CausalConvBlock(c_in, hidden_channels, kernel_size, dilation=d, dropout=dropout))
            c_in = hidden_channels
        self.net = nn.Sequential(*layers)
    def forward(self, x):# x: [B, C_in, T]
        return self.net(x)  # [B, hidden, T']

class CNN_TCN_LSTM(nn.Module):
    def __init__(self, in_features, cnn_out=32, tcn_hidden=64, lstm_hidden=64, lstm_layers=1, horizon=24):
        super().__init__()
        # 输入按 [B, F, T] 处理
        self.cnn = MultiScaleCNN(in_channels=in_features, out_channels=cnn_out, kernel_sizes=(3,5,7,11))
        self.tcn = TCN(in_channels=in_features, hidden_channels=tcn_hidden, num_layers=4, kernel_size=3, dilations=(1,2,4,8))
        # 融合后通道数
        self.fused_channels = cnn_out * 4 + tcn_hidden
        self.lstm = nn.LSTM(input_size=self.fused_channels, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, horizon)

    def forward_features(self, x):# x: [B, L, F]
        x = x.transpose(1, 2)  # -> [B, F, L]
        cnn_feat = self.cnn(x)     # [B, cnn_out*4, L]
        tcn_feat = self.tcn(x)     # [B, tcn_hidden, L']，L'与L一致（采用因果裁剪）
        # 对齐长度
        L = min(cnn_feat.size(-1), tcn_feat.size(-1))
        cnn_feat = cnn_feat[:, :, :L]
        tcn_feat = tcn_feat[:, :, :L]
        fused = torch.cat([cnn_feat, tcn_feat], dim=1)  # [B, fused_channels, L]
        return fused.transpose(1, 2), cnn_feat, tcn_feat  # -> [B, L, C], [B, Ccnn, L], [B, Ctcn, L]

    def forward(self, x):
        fused_seq, _, _ = self.forward_features(x)
        out, (h, c) = self.lstm(fused_seq)   # [B, L, H]
        # 取最后一步隐藏态
        last_h = out[:, -1, :]               # [B, H]
        yhat = self.fc(last_h)               # [B, horizon]
        return yhat

# 4) 训练与评估工具
def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val = np.inf
    history = {'train': [], 'val': []}
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                yhat = model(xb)
                loss = criterion(yhat, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_cnn_tcn_lstm.pt')
        print(f'Epoch {ep+1}/{epochs} - train: {train_loss:.4f} - val: {val_loss:.4f}')
    return history

def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            yhat = model(xb)
            preds.append(yhat.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues)**2))
    mape = np.mean(np.abs((trues - preds) / (trues + 1e-8)))
    return preds, trues, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# 5) 可视化
def plot_time_components(t, y, trend, seasonal, noise, spikes):
    plt.figure(figsize=(12, 6))
    plt.plot(t, y, color='magenta', label='总序列 y', linewidth=2)
    plt.plot(t, trend, color='lime', label='趋势 trend', linewidth=2)
    plt.plot(t, seasonal, color='gold', label='季节 seasonal', linewidth=2)
    plt.plot(t, noise, color='cyan', label='噪声 noise', alpha=0.7)
    plt.scatter(t, spikes, color='red', label='尖峰 spikes', s=15)
    plt.title('图1：时间序列分解可视化')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_power_spectrum(y, fs=1.0):
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1/fs)
    power = np.abs(Y)**2
    plt.figure(figsize=(12, 5))
    plt.plot(freqs, power, color='orange', linewidth=2)
    plt.title('图2：频谱功率密度')
    plt.xlabel('频率')
    plt.ylabel('功率')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_acf(y, max_lag=60):
    y = y - np.mean(y)
    acf_vals = [1.0]
    var = np.var(y)
    for lag in range(1, max_lag+1):
        acf = np.sum(y[:-lag] * y[lag:]) / ((len(y)-lag) * var + 1e-8)
        acf_vals.append(acf)
    plt.figure(figsize=(12, 5))
    plt.stem(range(0, max_lag+1), acf_vals, linefmt='g-', markerfmt='ro', basefmt='k-')
    plt.title('图3：自相关函数（ACF）')
    plt.xlabel('滞后阶数')
    plt.ylabel('相关性')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(y, fs=1.0, nperseg=128, noverlap=96):
    f, tt, Zxx = stft(y, fs=fs, nperseg=nperseg, noverlap=noverlap)
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(tt, f, np.abs(Zxx), shading='gouraud', cmap='plasma')
    plt.title('图4：STFT时频谱')
    plt.xlabel('时间')
    plt.ylabel('频率')
    plt.colorbar(label='幅值')
    plt.tight_layout()
    plt.show()

def plot_predictions(true_seq, pred_seq, title='图5：预测 vs 真值'):
    plt.figure(figsize=(12,6))
    plt.plot(true_seq, color='blue', label='真值', linewidth=2)
    plt.plot(pred_seq, color='red', label='预测', linewidth=2)
    plt.fill_between(np.arange(len(pred_seq)), pred_seq-0.2, pred_seq+0.2, color='orange', alpha=0.2, label='伪置信带')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_training_curves(history):
    plt.figure(figsize=(12,5))
    plt.plot(history['train'], color='green', label='训练损失', linewidth=2)
    plt.plot(history['val'], color='purple', label='验证损失', linewidth=2)
    plt.title('图6：训练/验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_maps(cnn_feat, tcn_feat):
    # 选择一个样本的特征做热图
    cnn_map = cnn_feat[0].detach().cpu().numpy()  # [Ccnn, L]
    tcn_map = tcn_feat[0].detach().cpu().numpy()  # [Ctcn, L]
    plt.figure(figsize=(12, 4))
    plt.imshow(cnn_map, aspect='auto', cmap='inferno')
    plt.colorbar()
    plt.title('图7：CNN多尺度特征热图')
    plt.xlabel('时间步')
    plt.ylabel('卷积通道')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.imshow(tcn_map, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('图8：TCN膨胀卷积特征热图')
    plt.xlabel('时间步')
    plt.ylabel('卷积通道')
    plt.tight_layout()
    plt.show()

# 6) 主流程
def main():
    data, y_norm, norm_stats, exogs_norm, t = generate_virtual_series(T=2200, num_exog=3)

    # 可视化：我们知道生成组件（trend/seasonal/noise/spikes），这里重现它们用于解释
    # 注意：生成函数内部 trend/seasonal/noise/spikes 并未返回；为演示，我们重新近似估计：
    # 简单移动平均提取趋势（仅用于展示）
    win = 51
    trend_est = np.convolve(y_norm, np.ones(win)/win, mode='same')
    seasonal_est = y_norm - trend_est
    noise_est = seasonal_est - np.convolve(seasonal_est, np.ones(7)/7, mode='same')
    # spikes估计：超阈值点（仅为演示）
    spikes_est = np.zeros_like(y_norm)
    spikes_est[np.abs(y_norm - trend_est - seasonal_est) > 2.0] = (y_norm - trend_est - seasonal_est)[np.abs(y_norm - trend_est - seasonal_est) > 2.0]

    # 图1：序列分解（估计）
    plot_time_components(t, y_norm, trend_est, seasonal_est, noise_est, spikes_est)

    # 图2：功率谱
    plot_power_spectrum(y_norm)

    # 图3：ACF
    plot_acf(y_norm, max_lag=80)

    # 图4：STFT时频谱
    plot_spectrogram(y_norm, fs=1.0, nperseg=128, noverlap=96)

    # 划分训练/验证/测试
    lookback, horizon = 128, 24
    train_data = data[:1800]
    val_data = data[1800:2000]
    test_data = data[2000:]

    train_ds = SeqDataset(train_data, lookback, horizon)
    val_ds = SeqDataset(val_data, lookback, horizon)
    test_ds = SeqDataset(test_data, lookback, horizon)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = 'cuda'if torch.cuda.is_available() else'cpu'
    model = CNN_TCN_LSTM(in_features=data.shape[1], cnn_out=16, tcn_hidden=32, lstm_hidden=64, lstm_layers=1, horizon=horizon).to(device)

    history = train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, device=device)

    # 图6：训练曲线
    plot_training_curves(history)

    # 加载最优参数
    model.load_state_dict(torch.load('best_cnn_tcn_lstm.pt', map_location=device))

    preds, trues, metrics = evaluate_model(model, test_loader, device=device)
    print('测试集指标：', metrics)

    # 绘制一个样本的预测 vs 真值
    sample_true = trues[0]
    sample_pred = preds[0]
    plot_predictions(sample_true, sample_pred, title='图5：测试样本预测 vs 真值')

    # 中间特征可视化（CNN/TCN）
    xb, yb = next(iter(test_loader))
    fused_seq, cnn_feat, tcn_feat = model.forward_features(xb.to(device))
    plot_feature_maps(cnn_feat, tcn_feat)

if __name__ == '__main__':
    main()