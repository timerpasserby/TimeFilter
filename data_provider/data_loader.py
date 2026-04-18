"""
这个数据读取文件负责把不同数据集整理成训练需要的序列样本，和 `data_factory.py`、`run.py` 配合使用。
"""

import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
            
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_weather_module = bool(getattr(args, 'use_weather_module', False))
        self.use_blast_module = bool(getattr(args, 'use_blast_module', False))
        self.extra_inputs_enabled = self.use_weather_module or self.use_blast_module
        self.patch_len = int(getattr(args, 'patch_len', self.seq_len))
        self.patch_stride = self.patch_len
        self.num_patches = int((self.seq_len - self.patch_len) / self.patch_stride + 1)
        self.patch_end_indices = np.arange(
            self.patch_len - 1,
            self.patch_len - 1 + self.num_patches * self.patch_stride,
            self.patch_stride,
            dtype=np.int64,
        )
        self.blast_max_events = int(getattr(args, 'blast_max_events', 64))
        self.weather_feature_names = ['rainfall', 'temperature', 'humidity']
        self.weather_x = None
        self.weather_mask = None
        self.blast_events = None
        self.segment_dates = None

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    # 统一识别并转换时间列，避免不同自定义表头带来的额外分支。
    def _normalize_date_column(self, data_frame):
        """把输入表中的时间列统一成 date。"""
        normalized = data_frame.copy()
        if 'date' in normalized.columns:
            normalized['date'] = pd.to_datetime(normalized['date'])
            return normalized

        alias_candidates = ['report_time', 'datetime', 'time', 'timestamp']
        alias_col = next((col for col in alias_candidates if col in normalized.columns), None)
        if alias_col is None:
            raise ValueError("custom dataset must contain a 'date' column or a supported time alias")
        normalized = normalized.rename(columns={alias_col: 'date'})
        normalized['date'] = pd.to_datetime(normalized['date'])
        return normalized

    # 读取并对齐天气数据，为每个样本提供同窗口天气序列。
    def _load_weather_side_data(self, all_dates, border1s, border2s, border1, border2):
        """按主序列时间轴对齐天气序列，并只保留当前切分区间。"""
        weather_path = getattr(self.args, 'weather_path', '')
        if not weather_path:
            raise ValueError('use_weather_module=True 时必须提供 weather_path。')
        if not os.path.exists(weather_path):
            raise FileNotFoundError(f'天气文件不存在: {weather_path}')

        weather_frame = self._normalize_date_column(pd.read_csv(weather_path))
        required_columns = set(self.weather_feature_names)
        if not required_columns.issubset(set(weather_frame.columns)):
            raise ValueError(
                f'天气文件缺少必要列，当前需要 {self.weather_feature_names}，实际为 {weather_frame.columns.tolist()}'
            )

        weather_frame = weather_frame[['date'] + self.weather_feature_names].drop_duplicates(subset='date', keep='last')
        aligned_dates = pd.DataFrame({'date': all_dates.reset_index(drop=True)})
        aligned_weather = aligned_dates.merge(weather_frame, on='date', how='left', sort=False)

        observed_mask = (~aligned_weather[self.weather_feature_names].isna().any(axis=1)).astype(np.float32).to_numpy()
        filled_weather = aligned_weather[self.weather_feature_names].ffill().bfill().fillna(0.0)

        weather_scaler = StandardScaler()
        weather_scaler.fit(filled_weather.iloc[border1s[0]:border2s[0]].values)
        weather_values = weather_scaler.transform(filled_weather.values).astype(np.float32)
        weather_mask = observed_mask.astype(np.float32).reshape(-1, 1)

        self.weather_x = weather_values[border1:border2]
        self.weather_mask = weather_mask[border1:border2]

    # 读取爆破日志，供每个样本按自己的历史窗口裁切事件列表。
    def _load_blast_side_data(self):
        """加载并清洗爆破事件日志。"""
        blast_path = getattr(self.args, 'blast_path', '')
        if not blast_path:
            raise ValueError('use_blast_module=True 时必须提供 blast_path。')
        if not os.path.exists(blast_path):
            raise FileNotFoundError(f'爆破文件不存在: {blast_path}')

        blast_frame = self._normalize_date_column(pd.read_csv(blast_path))
        coordinate_candidates = [
            ['location_x', 'location_y', 'location_z'],
            ['grid_x', 'grid_y', 'grid_z'],
            ['x', 'y', 'z'],
        ]
        coordinate_columns = next(
            (columns for columns in coordinate_candidates if all(column in blast_frame.columns for column in columns)),
            None,
        )
        if coordinate_columns is None or 'intensity' not in blast_frame.columns:
            raise ValueError(
                '爆破文件必须包含位置列和 intensity 列，支持的位置列命名为 '
                "['location_x', 'location_y', 'location_z']、['grid_x', 'grid_y', 'grid_z'] 或 ['x', 'y', 'z']。"
            )

        self.blast_events = (
            blast_frame[['date'] + coordinate_columns + ['intensity']]
            .rename(columns={
                coordinate_columns[0]: 'coord_x',
                coordinate_columns[1]: 'coord_y',
                coordinate_columns[2]: 'coord_z',
            })
            .sort_values('date')
            .reset_index(drop=True)
        )

    # 根据当前样本时间窗口构建对齐后的爆破输入。
    def _build_blast_window_inputs(self, seq_dates):
        """把当前样本窗口内的爆破事件整理成定长张量。"""
        seq_start = seq_dates.iloc[0]
        seq_end = seq_dates.iloc[-1]
        window_events = self.blast_events[
            (self.blast_events['date'] >= seq_start) & (self.blast_events['date'] <= seq_end)
        ].tail(self.blast_max_events)

        blast_locs = np.zeros((self.blast_max_events, 3), dtype=np.float32)
        blast_times = np.zeros((self.blast_max_events,), dtype=np.float32)
        blast_intensity = np.zeros((self.blast_max_events,), dtype=np.float32)

        if not window_events.empty:
            event_count = len(window_events)
            blast_locs[:event_count] = window_events[['coord_x', 'coord_y', 'coord_z']].to_numpy(dtype=np.float32)
            relative_hours = ((window_events['date'] - seq_start).dt.total_seconds() / 3600.0).to_numpy(dtype=np.float32)
            blast_times[:event_count] = relative_hours
            blast_intensity[:event_count] = window_events['intensity'].to_numpy(dtype=np.float32)

        patch_dates = seq_dates.iloc[self.patch_end_indices]
        patch_times = ((patch_dates - seq_start).dt.total_seconds() / 3600.0).to_numpy(dtype=np.float32)
        return blast_locs, blast_times, blast_intensity, patch_times

    # 整理当前样本对应的天气与爆破侧信息。
    def _build_extra_inputs(self, s_begin, s_end):
        """根据样本窗口返回模型需要的额外输入字典。"""
        extra_inputs = {}
        if self.use_weather_module:
            extra_inputs['weather_seq'] = self.weather_x[s_begin:s_end].astype(np.float32)
            extra_inputs['weather_mask'] = self.weather_mask[s_begin:s_end].astype(np.float32)

        if self.use_blast_module:
            seq_dates = self.segment_dates.iloc[s_begin:s_end].reset_index(drop=True)
            blast_locs, blast_times, blast_intensity, patch_times = self._build_blast_window_inputs(seq_dates)
            extra_inputs['blast_locs'] = blast_locs
            extra_inputs['blast_times'] = blast_times
            extra_inputs['blast_intensity'] = blast_intensity
            extra_inputs['patch_times'] = patch_times
        return extra_inputs

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = self._normalize_date_column(df_raw)

        # 自定义数据集在 features 为 M/MS 时支持直接使用整张多变量表。
        cols = list(df_raw.columns)
        cols.remove('date')
        if self.target in cols:
            cols.remove(self.target)
            ordered_cols = ['date'] + cols + [self.target]
        else:
            if self.features in ['S', 'MS']:
                raise ValueError(f"target column {self.target!r} not found in custom dataset")
            ordered_cols = ['date'] + cols
        df_raw = df_raw[ordered_cols].copy()
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.segment_dates = df_raw['date'].iloc[border1:border2].reset_index(drop=True)

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            if self.extra_inputs_enabled:
                raise ValueError('启用天气或爆破模块时，当前暂不支持对 custom 数据集同时做序列增强。')
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp
        if self.use_weather_module:
            self._load_weather_side_data(df_raw['date'], border1s, border2s, border1, border2)
        if self.use_blast_module:
            self._load_blast_side_data()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.extra_inputs_enabled:
            extra_inputs = self._build_extra_inputs(s_begin, s_end)
            return seq_x, seq_y, seq_x_mark, seq_y_mark, extra_inputs
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        print('data file:', data_file)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        if self.set_type == 2:
            s_begin = index * 12
        else:
            s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 2:
            return (len(self.data_x) - self.seq_len - self.pred_len + 1) // 12
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Climate(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='climate.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))


        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # 注意没有seq_x_mark和seq_y_mark，用seq_x和seq_y代替
        return seq_x, seq_y, seq_x[:, :4], seq_y[:, :4]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
