import gc
import glob
import os
import random
import shutil
from abc import abstractmethod
from typing import List, Tuple, Any

import numpy as np
import pandas as pd  # type: ignore
import pyunpack
import wget  # type: ignore
from py7zr import unpack_7zarchive
from sklearn.preprocessing import StandardScaler  # type: ignore
from tqdm import tqdm  # type: ignore

from src.column_types import ValueTypes
from src.config import Paths

datasets_factory = {}


def register_dataset(cls):
    if cls.__name__ in datasets_factory:
        raise ValueError(f'Dataset {cls.name} is duplicated.')
    else:
        datasets_factory[cls.name] = cls
    return cls


class AbstractDataset:
    name = ''
    unzip_dir = 'raw'
    filename = ''
    data_url = ''
    features_filename = ''
    _input_cols: List[Tuple[str, Any]] = []
    _id_col = 'id'

    def __init__(self,
                 ts_len: int,
                 n_enc_steps: int,
                 sample_sz: int | None = None,
                 unknown_category_name: str = 'other', **kwargs):
        self._ts_len = ts_len
        self._n_enc_len = n_enc_steps
        self._sample_sz = sample_sz
        self._unknown_category_name = unknown_category_name
        self._targ_idx: List[int] = []
        self._known_num_idx: List[int] = []
        self._known_cat_idx: List[int] = []
        self._known_cat_vocab_sz: List[int] = []
        self._known_cat_vocab: dict = {}
        self._unknown_num_idx: List[int] = []
        self._unknown_cat_idx: List[int] = []
        self._unknown_cat_vocab_sz: List[int] = []
        self._unknown_cat_vocab: dict = {}
        self._static_num_idx: List[int] = []
        self._static_cat_idx: List[int] = []
        self._static_cat_vocab_sz: List[int] = []
        self._static_cat_vocab: dict = {}
        self._targ_scaler_objs: dict = {}
        self._num_scaler_objs: dict = {}
        self._cat_vocabs: dict = {}

        # from each example extract multiple x, y pairs using a sliding window of fixed step
        self._step_time_steps = 7

    @property
    def targ_idx(self) -> List[int]:
        """
        Returns indices of the target columns, in the provided dataset.
        """
        assert len(self._targ_idx) > 0
        return self._targ_idx

    @property
    def target_columns(self) -> list:
        return [c[0] for c in self._input_cols if c[1] == ValueTypes.TARGET]

    @property
    def known_num_idx(self) -> List[int] | None:
        """
        Indices of numerical values known in the forecast horizon,
        or ``None``, if no such values exist.

        Returns
        -------
        list or None
            Corresponding indices in the provided dataset.
        """
        return self._known_num_idx if len(self._known_num_idx) > 0 else None

    @property
    def known_numeric_columns(self) -> list:
        return [c[0] for c in self._input_cols if c[1] == ValueTypes.KNOWN_NUMERIC]

    @property
    def known_cat_idx(self) -> List[int] | None:
        """
        Indices of categorical values known in the forecast horizon,
        or None, if no such values exist (e.g. day of the week etc.).

        Returns
        -------
        list or None
            Corresponding indices in the provided dataset.
        """
        return self._known_cat_idx if len(self._known_cat_idx) > 0 else None

    @property
    def known_categorical_columns(self) -> list:
        return [c[0] for c in self._input_cols if c[1] == ValueTypes.KNOWN_CATEGORICAL]

    @property
    def known_cat_vocab_sz(self) -> List[int] | None:
        """
        Number of known categorical classes or ``None`` if no
        such inputs exist.
        Returns a list where each element specifies the number
        of categories for each categorical column. So the
        lengths should match.
        """
        if self._known_cat_vocab_sz is not None and self._known_cat_idx is not None:
            assert len(self._known_cat_vocab_sz) == len(self._known_cat_idx)
            return self._known_cat_vocab_sz
        else:
            return None

    @property
    def unknown_num_idx(self) -> List[int] | None:
        """
        Indices of numerical values known until the time of prediction,
        or ``None``, if no such values exist.

        Returns
        -------
        list or None
            Corresponding indices in the provided dataset.
        """
        return self._unknown_num_idx if len(self._unknown_num_idx) > 0 else None

    @property
    def unknown_numeric_columns(self) -> list:
        return [c[0] for c in self._input_cols if c[1] == ValueTypes.UNKNOWN_NUMERIC]

    @property
    def unknown_cat_idx(self) -> List[int] | None:
        """
        Indices of categorical values known until the time of prediction,
        or None, if no such values exist.

        Returns
        -------
        list or None
            Corresponding indices in the provided dataset.
        """
        return self._unknown_cat_idx if len(self._unknown_cat_idx) > 0 else None

    @property
    def unknown_categorical_columns(self) -> list:
        return [c[0] for c in self._input_cols if c[1] == ValueTypes.UNKNOWN_CATEGORICAL]

    @property
    def unknown_cat_vocab_sz(self) -> List[int] | None:
        """
        Number of unknown categorical classes or ``None`` if no
        such inputs exist.
        """
        if self._unknown_cat_vocab_sz is not None and self._unknown_cat_idx is not None:
            assert len(self._unknown_cat_vocab_sz) == len(self._unknown_cat_idx)
            return self._unknown_cat_vocab_sz
        else:
            return None

    @property
    def static_num_idx(self) -> List[int] | None:
        """
        Indices of static numeric values, constant throughout the
        time series, or ``None``, if no such values exist.

        Returns
        -------
        list or None
            Corresponding indices in the provided dataset.
        """
        return self._static_num_idx if len(self._static_num_idx) > 0 else None

    @property
    def static_numeric_columns(self) -> list:
        return [c[0] for c in self._input_cols if c[1] == ValueTypes.STATIC_NUMERIC]

    @property
    def static_cat_idx(self) -> List[int] | None:
        """
        Indices of static categorical values, constant throughout the
        length of the time series, or ``None``, if no such values exist.

        Returns
        -------
        list or None
            Corresponding indices in the provided dataset.
        """
        return self._static_cat_idx if len(self._static_cat_idx) > 0 else None

    @property
    def static_categorical_columns(self) -> list:
        return [c[0] for c in self._input_cols if c[1] == ValueTypes.STATIC_CATEGORICAL]

    @property
    def static_cat_vocab_sz(self) -> List[int] | None:
        """
        Number of static categorical classes or ``None`` if no
        such inputs exist.
        """
        if self._static_cat_vocab_sz is not None and self._static_cat_idx is not None:
            assert len(self._static_cat_vocab_sz) == len(self._static_cat_idx)
            return self._static_cat_vocab_sz
        else:
            return None

    def extract_xy_pairs(self, ts_len: int, n_dec_steps: int) -> (
            Tuple)[Tuple[Any, Any], Tuple[Any, Any], Tuple[Any, Any]]:
        # if dataset is not present locally - download it
        download_dir = self._make_dataset_dir()
        if not self._data_exists():
            print(f'Dataset {self.name} does not exist under '
                  f'the {Paths.data_base_dir} directory')
            self._download_dataset(self.name, download_dir=download_dir)

        unzip_fp = os.path.join(download_dir, self.unzip_dir)
        if not os.path.exists(unzip_fp):
            dl_fp = os.path.join(download_dir, f'{self.filename}.zip')
            self.unzip_files(dl_fp, unzip_fp)

        features_fp = self._make_features_dataset_filepath()
        if not os.path.exists(features_fp):
            self._create_features()
        else:
            print(f'Preparing dataset using precalculated features: {features_fp}')

        features_df = pd.read_csv(features_fp, index_col=0)

        train_df, val_df, test_df = self._split(features_df)

        x_train, y_train = self._fit_transform(train_df, shuffle=True, dataset_tag='training')
        x_val, y_val = self._transform_and_batch(val_df, dataset_tag='validation')
        x_test, y_test = self._transform_and_batch(test_df, dataset_tag='testing')

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def _download_dataset(self, dataset_name: str, download_dir: str) -> None:
        """
        Download dataset.
        """
        dl_fp = os.path.join(download_dir, f'{self.filename}.zip')
        print(f'Downloading {dataset_name} dataset. Files will'
              f' be saved under {dl_fp}. It might take a few minutes.')
        wget.download(self.data_url, dl_fp)

    def _data_exists(self) -> bool:
        dataset_fp = os.path.join(Paths.data_base_dir, self.name)  # type: ignore
        # Fixes: a bug where it just checks if the path exists, it should list dirs and check if there is anything in it
        return len(os.listdir(dataset_fp)) != 0

    def _make_dataset_dir(self) -> Any:
        dataset_fp = os.path.join(Paths.data_base_dir, self.name)  # type: ignore
        if not os.path.exists(dataset_fp):
            os.makedirs(dataset_fp)
        return dataset_fp

    @abstractmethod
    def _split(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split into train, validation and test sets."""
        raise NotImplementedError

    @abstractmethod
    def _create_features(self):
        """Create features and save to disk."""
        raise NotImplementedError

    def _fit_transform(self,
                       data: pd.DataFrame,
                       shuffle: bool = True,
                       dataset_tag: str = '') -> Tuple[np.ndarray, np.ndarray]:
        self._fit(data)

        x = self._transform_and_batch(data, shuffle=shuffle, dataset_tag=dataset_tag)
        return x

    def _fit(self, train_df: pd.DataFrame) -> None:
        """
        Identify feature types and fit transformation steps accordingly.
        E.g. identify numeric features and fit scaler objects.
        """
        self._scale(train_df)

        self._create_vocabs(train_df)

        self._define_column_indices()

    def _scale(self, data: pd.DataFrame) -> None:
        col_names = data.columns.tolist()

        # make sure that expected columns exist in the dataset
        for c in self._input_cols:
            assert c[0] in col_names


        targ_cols = self.target_columns

        # pool numeric columns so that you scale them together. Static numeric are not scaled.
        num_cols = self.known_numeric_columns + self.unknown_numeric_columns

        # define scaler objects for numeric values -> specific to each example
        for example_id, df in data.groupby(self._id_col):
            if df.shape[0] < self._ts_len:
                continue

            # fit scaler so that you can use it during the transformation step
            if len(num_cols) > 0:
                num_df = df[num_cols].copy()
                scaler = StandardScaler()
                scaler.fit(num_df)
                self._num_scaler_objs[example_id] = scaler

            # center-scale targets
            targ_df = df[targ_cols].copy()
            scaler = StandardScaler()
            scaler.fit(targ_df)
            self._targ_scaler_objs[example_id] = scaler

    def _create_vocabs(self, data) -> None:
        """
        Create category to index lookup tables for all categorical
        columns.
        If the unknown category name is provided an extra
        category will be added to that dictionary (e.g.
        to handle unknown cases seen during inference).

        Parameters
        ----------
        data: pd.DataFrame
            Training dataset.
        """
        cat_cols = (self.known_categorical_columns +
                    self.unknown_categorical_columns +
                    self.static_categorical_columns)

        if len(cat_cols) > 0:
            for cat_col in cat_cols:
                cat_names = list(set(data[cat_col].astype(str).values.tolist()))
                cat2idx = {self._unknown_category_name: 0}  # fall-back class for unseen data
                for i, c in enumerate(sorted(cat_names)):
                    cat2idx[c] = i+1

                self._cat_vocabs[cat_col] = cat2idx

    def _define_column_indices(self) -> None:
        """
        Get column indices so that you can pass them to the model
        during initialization. The column order in the transform data will be:
        1. target columns
        2. known numeric + unknown numeric (if any)
        3. known categorical + unknown categorical + static categorical (if any)
        4. static numeric (if any)
        """
        transformed_df_col_names = (self.target_columns + self.known_numeric_columns +
                                    self.unknown_numeric_columns + self.known_categorical_columns +
                                    self.unknown_categorical_columns + self.static_categorical_columns +
                                    self.static_numeric_columns)

        self._targ_idx = [transformed_df_col_names.index(c) for c in self.target_columns]
        self._known_num_idx = [transformed_df_col_names.index(c) for c in self.known_numeric_columns]
        self._unknown_num_idx = [transformed_df_col_names.index(c) for c in self.unknown_numeric_columns]
        self._known_cat_idx = [transformed_df_col_names.index(c) for c in self.known_categorical_columns]
        self._unknown_cat_idx = [transformed_df_col_names.index(c) for c in self.unknown_categorical_columns]
        self._static_num_idx = [transformed_df_col_names.index(c) for c in self.static_numeric_columns]
        self._static_cat_idx = [transformed_df_col_names.index(c) for c in self.static_categorical_columns]

        # calculate the size of the vocab in each case so that the model know
        # the size of embeddings to initialize
        if self.known_categorical_columns:
            for col_name in self.known_categorical_columns:
                self._known_cat_vocab_sz.append(len(self._cat_vocabs[col_name]))
        if self.unknown_categorical_columns:
            for col_name in self.unknown_categorical_columns:
                self._unknown_cat_vocab_sz.append(len(self._cat_vocabs[col_name]))
        if self.static_categorical_columns:
            for col_name in self.static_categorical_columns:
                self._static_cat_vocab_sz.append(len(self._cat_vocabs[col_name]))

    def _transform_and_batch(self, data: pd.DataFrame, shuffle: bool = False, dataset_tag: str = '') -> (
            Tuple)[np.ndarray, np.ndarray]:
        xs, ys = [], []
        targ_cols = self.target_columns
        num_cols = self.known_numeric_columns + self.unknown_numeric_columns
        cat_cols = self.known_categorical_columns + self.unknown_categorical_columns + self.static_categorical_columns
        stat_num = self.static_numeric_columns

        all_cols = targ_cols + num_cols + cat_cols + stat_num
        n_inputs = len(all_cols)
        # confirm that you don't have duplicates
        assert n_inputs == len(list(set(all_cols)))

        for example_id, df in tqdm(data.groupby(self._id_col), desc='Transforming values'):
            if df.shape[0] < self._ts_len:
                continue  # short examples are dropped

            try:
                scaler = self._targ_scaler_objs[example_id]
            except KeyError:  # skip examples without scaler
                continue
            targ_df = df[targ_cols].copy()
            x_full = scaler.transform(targ_df)

            scaler = self._num_scaler_objs.get(example_id)
            if scaler is not None and len(num_cols) > 0:
                num_df = df[num_cols].copy()
                num_scaled = scaler.transform(num_df)
                x_full = np.hstack((x_full, num_scaled))

            # transform categorical
            if len(cat_cols) > 0:
                for cat_col in cat_cols:
                    cat2id = self._cat_vocabs[cat_col]
                    cat_df = df[cat_col].astype(str).values.tolist()
                    ids_array = np.array([cat2id.get(c, cat2id[self._unknown_category_name]) for c in cat_df]).reshape(-1, 1)

                    x_full = np.hstack((x_full, ids_array))

            # assemble transformed values to a data frame
            if len(stat_num) > 0:
                stat_mat = df[stat_num].values
                x_full = np.hstack((x_full, stat_mat))

            # create xy pairs
            n_targets = len(targ_cols)
            n_rows = x_full.shape[0]
            start_idx = 0
            end_idx = self._ts_len
            while end_idx < n_rows:
                x = x_full[start_idx:end_idx, :]
                y = x_full[(start_idx + self._n_enc_len):end_idx, :n_targets]
                assert x.shape[0] == self._ts_len
                assert x.shape[1] == n_inputs
                xs.append(x)
                ys.append(y)

                start_idx += self._step_time_steps
                end_idx += self._step_time_steps

        xs_array = np.array(xs)
        ys_array = np.array(ys)

        if shuffle:
            indices = [i for i in range(len(xs))]
            random.shuffle(indices)
            xs_array = xs_array[indices]
            ys_array = ys_array[indices]

        if self._sample_sz is not None:
            print(f'Sampling {dataset_tag} dataset to {self._sample_sz} examples')
            xs_array = xs_array[:self._sample_sz, ...]
            ys_array = ys_array[:self._sample_sz, ...]

        return xs_array, ys_array

    def _make_features_dataset_filepath(self):
        out_fp = os.path.join(Paths.data_base_dir, self.name, self.features_filename)
        return out_fp

    @staticmethod
    def unzip_files(dl_fp: str, unzip_dir: str) -> None:
        if not os.path.exists(unzip_dir):
            os.makedirs(unzip_dir)
        print(f'Unzipping file: {dl_fp} to {unzip_dir}')
        pyunpack.Archive(dl_fp).extractall(unzip_dir)


@register_dataset
class ElectricityDataset(AbstractDataset):
    name = 'electricity'
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'
    filename = 'LD2011_2014.txt'
    features_filename = 'electricity_features.csv'
    _input_cols = [
        ('id', ValueTypes.EXAMPLE_ID),
        ('power_usage', ValueTypes.TARGET),
        ('hour', ValueTypes.KNOWN_NUMERIC),
        ('day_of_week', ValueTypes.KNOWN_CATEGORICAL),
        # ('day_of_week', ValueTypes.KNOWN_NUMERIC),
        ('hours_from_start', ValueTypes.KNOWN_NUMERIC),
        ('categorical_id', ValueTypes.STATIC_CATEGORICAL)
    ]

    def __init__(self,
                 ts_len: int,
                 n_enc_steps: int,
                 sample_sz: int | None = None,
                 unknown_category_name: str = 'other',
                 **kwargs):
        super().__init__(ts_len=ts_len,
                         n_enc_steps=n_enc_steps,
                         sample_sz=sample_sz,
                         unknown_category_name=unknown_category_name,
                         **kwargs)

    def _split(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        val_set_boundary = 1315
        test_set_boundary = 1339

        index = features_df['days_from_start']
        train_df = features_df[index < val_set_boundary].copy()
        val_df = features_df[(index >= val_set_boundary - 7) &
                             (index < test_set_boundary)].copy()
        test_df = features_df[(index >= test_set_boundary - 7)]

        return train_df, val_df, test_df

    def _create_features(self):
        print('Extracting features from raw dataset ...')
        fp = os.path.join(Paths.data_base_dir, self.name, self.unzip_dir, self.filename)
        data = pd.read_csv(fp, index_col=0, sep=';', decimal=',')
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        hourly_data = data.resample('1h').mean().replace(0, np.nan)
        earliest_time = hourly_data.index.min()
        client_codes = hourly_data.columns.values
        dfs_with_features = []
        for client_code in tqdm(client_codes, desc='Creating features ...'):
            cl_data = hourly_data[client_code].copy()

            start_date = min(cl_data.ffill().dropna().index)
            end_data = max(cl_data.bfill().dropna().index)

            valid_range = (cl_data.index >= start_date) & (cl_data.index <= end_data)
            cl_data = cl_data[valid_range].fillna(0.)

            cl_out_df = pd.DataFrame({'power_usage': cl_data})
            date = cl_out_df.index
            cl_out_df['hours_from_start'] = ((date - earliest_time).seconds / 60 / 60 +
                                             (date - earliest_time).days * 24)
            cl_out_df['days_from_start'] = (date - earliest_time).days
            cl_out_df['categorical_id'] = client_code
            cl_out_df['date'] = date
            cl_out_df['id'] = client_code
            cl_out_df['hour'] = date.hour
            cl_out_df['day'] = date.day
            cl_out_df['day_of_week'] = date.dayofweek
            cl_out_df['month'] = date.month
            dfs_with_features.append(cl_out_df)
        out_df = pd.concat(dfs_with_features, axis=0, join='outer') \
            .reset_index(drop=True)
        out_df['categorical_day_of_week'] = out_df['day_of_week'].copy()
        out_df['categorical_hour'] = out_df['hour'].copy()
        out_df = out_df[(out_df['days_from_start'] >= 1096) &
                        (out_df['days_from_start'] < 1346)].copy()
        out_fp = self._make_features_dataset_filepath()
        out_df.to_csv(out_fp)


@register_dataset
class TrafficDataset(AbstractDataset):
    name = 'traffic'
    unzip_dir = 'raw'
    filename = 'PEMS-SF'
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip'
    features_filename = 'traffic_features.csv'
    _input_cols = [
      ('id', ValueTypes.EXAMPLE_ID),
      ('hours_from_start', ValueTypes.KNOWN_NUMERIC),
      ('values', ValueTypes.TARGET),
      ('time_on_day', ValueTypes.KNOWN_NUMERIC),
      ('day_of_week', ValueTypes.KNOWN_CATEGORICAL),
      ('categorical_id', ValueTypes.STATIC_CATEGORICAL),
    ]

    def __init__(self,
                 ts_len: int,
                 n_enc_steps: int,
                 sample_sz: int | None = None,
                 unknown_category_name: str = 'other',
                 **kwargs):
        super().__init__(ts_len=ts_len,
                         n_enc_steps=n_enc_steps,
                         sample_sz=sample_sz,
                         unknown_category_name=unknown_category_name,
                         **kwargs)

    def _create_features(self):
        # a copy of the original implementation
        data_folder = os.path.join(self._make_dataset_dir(), self.unzip_dir)
        shuffle_order = np.array(self._read_single_list(data_folder, 'randperm')) - 1  # index from 0
        train_dayofweek = self._read_single_list(data_folder, 'PEMS_trainlabels')
        train_tensor = self._read_matrix(data_folder, 'PEMS_train')
        test_dayofweek = self._read_single_list(data_folder, 'PEMS_testlabels')
        test_tensor = self._read_matrix(data_folder, 'PEMS_test')

        # Inverse permutate shuffle order
        print('Shuffling')
        inverse_mapping = {
            new_location: previous_location
            for previous_location, new_location in enumerate(shuffle_order)
        }
        reverse_shuffle_order = np.array([
            inverse_mapping[new_location]
            for new_location, _ in enumerate(shuffle_order)
        ])

        # Group and re-oder based on permuation matrix
        print('Reodering')
        day_of_week = np.array(train_dayofweek + test_dayofweek)
        combined_tensor = np.array(train_tensor + test_tensor)

        day_of_week = day_of_week[reverse_shuffle_order]
        combined_tensor = combined_tensor[reverse_shuffle_order]

        # Put everything back into a dataframe
        print('Parsing as dataframe')
        labels = ['traj_{}'.format(i) for i in self._read_single_list(data_folder, 'stations_list')]

        hourly_list = []
        for day, day_matrix in enumerate(combined_tensor):

            # Hourly data
            hourly = pd.DataFrame(day_matrix.T, columns=labels)
            hourly['hour_on_day'] = [int(i / 6) for i in hourly.index
                                     ]  # sampled at 10 min intervals
            if hourly['hour_on_day'].max() > 23 or hourly['hour_on_day'].min() < 0:
                raise ValueError('Invalid hour! {}-{}'.format(
                    hourly['hour_on_day'].min(), hourly['hour_on_day'].max()))

            hourly = hourly.groupby('hour_on_day', as_index=True).mean()[labels]
            hourly['sensor_day'] = day
            hourly['time_on_day'] = hourly.index
            hourly['day_of_week'] = day_of_week[day]

            hourly_list.append(hourly)

        hourly_frame = pd.concat(hourly_list, axis=0, ignore_index=True, sort=False)

        # Flatten such that each entity uses one row in dataframe
        store_columns = [c for c in hourly_frame.columns if 'traj' in c]
        other_columns = [c for c in hourly_frame.columns if 'traj' not in c]
        flat_df = pd.DataFrame(
            columns=['values', 'prev_values', 'next_values'] + other_columns + ['id'])

        all_dfs = [flat_df]
        for store in store_columns:
            print('Processing {}'.format(store))

            sliced = hourly_frame[[store] + other_columns].copy()
            sliced.columns = ['values'] + other_columns
            sliced['id'] = int(store.replace('traj_', ''))

            # Sort by Sensor-date-time
            key = sliced['id'].apply(str) \
                  + sliced['sensor_day'].apply(lambda x: '_' + self._format_index_string(x)) \
                  + sliced['time_on_day'].apply(lambda x: '_' + self._format_index_string(x))
            sliced = sliced.set_index(key).sort_index()

            sliced['values'] = sliced['values'].ffill()
            sliced['prev_values'] = sliced['values'].shift(1)
            sliced['next_values'] = sliced['values'].shift(-1)

            # flat_df = flat_df.append(sliced.dropna(), ignore_index=True, sort=False)
            all_dfs.append(sliced.dropna())
        flat_df = pd.concat(all_dfs).reset_index(drop=True)

        # Filter to match range used by other academic papers
        index = flat_df['sensor_day']
        flat_df = flat_df[index < 173].copy()

        # Creating columns fo categorical inputs
        flat_df['categorical_id'] = flat_df['id'].copy()
        flat_df['hours_from_start'] = flat_df['time_on_day'] + flat_df['sensor_day'] * 24.
        flat_df['categorical_day_of_week'] = flat_df['day_of_week'].copy()
        flat_df['categorical_time_on_day'] = flat_df['time_on_day'].copy()

        out_fp = self._make_features_dataset_filepath()
        flat_df.to_csv(out_fp)

    @staticmethod
    def _format_index_string(x):
        """Returns formatted string for key."""

        if x < 10:
            return '00' + str(x)
        elif x < 100:
            return '0' + str(x)
        elif x < 1000:
            return str(x)

        raise ValueError('Invalid value of x {}'.format(x))

    @staticmethod
    def _process_list(s, variable_type=int, delimiter=None):
        """Parses a line in the PEMS format to a list."""
        if delimiter is None:
            l = [variable_type(i) for i in s.replace('[', '').replace(']', '').split()]
        else:
            l = [variable_type(i) for i in s.replace('[', '').replace(']', '').split(delimiter)]

        return l

    def _read_single_list(self, data_folder, filename):
        """Returns single list from a file in the PEMS-custom format."""
        with open(os.path.join(data_folder, filename), 'r') as dat:
            l = self._process_list(dat.readlines()[0])
        return l

    def _read_matrix(self, data_folder, filename):
            """Returns a matrix from a file in the PEMS-custom format."""
            array_list = []
            with open(os.path.join(data_folder, filename), 'r') as dat:

                lines = dat.readlines()
                for i, line in enumerate(lines):
                    if (i + 1) % 50 == 0:
                        print('Completed {} of {} rows for {}'.format(i + 1, len(lines),
                                                                      filename))

                    array = [
                        self._process_list(row_split, variable_type=float, delimiter=None)
                        for row_split in self._process_list(
                            line, variable_type=str, delimiter=';')
                    ]
                    array_list.append(array)

            return array_list

    def _split(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        valid_boundary = 151
        test_boundary = 166
        index = features_df['sensor_day']
        train_df = features_df.loc[index < valid_boundary]
        valid_df = features_df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
        test_df = features_df.loc[index >= test_boundary - 7]
        return train_df, valid_df, test_df


@register_dataset
class FavoritaDataset(AbstractDataset):
    name = 'favorita'

    unzip_dir = 'raw'

    filename = 'favorita-grocery-sales-forecasting'
    data_url = ''
    features_filename = 'favorita_features.csv'
    _id_col = 'traj_id'
    _input_cols = [
        ('traj_id', ValueTypes.EXAMPLE_ID),
        ('log_sales', ValueTypes.TARGET),
        ('onpromotion', ValueTypes.KNOWN_CATEGORICAL),
        ('transactions', ValueTypes.UNKNOWN_NUMERIC),
        ('oil', ValueTypes.UNKNOWN_NUMERIC),
        ('day_of_week', ValueTypes.KNOWN_CATEGORICAL),
        ('day_of_month', ValueTypes.KNOWN_CATEGORICAL),
        ('month', ValueTypes.KNOWN_CATEGORICAL),  # could also be numeric like the ones above
        ('national_hol', ValueTypes.KNOWN_CATEGORICAL),
        ('regional_hol', ValueTypes.KNOWN_CATEGORICAL),
        ('local_hol', ValueTypes.KNOWN_CATEGORICAL),
        ('open', ValueTypes.KNOWN_NUMERIC),
        ('item_nbr', ValueTypes.STATIC_CATEGORICAL),
        ('city', ValueTypes.STATIC_CATEGORICAL),
        ('state', ValueTypes.STATIC_CATEGORICAL),
        ('type', ValueTypes.STATIC_CATEGORICAL),
        ('cluster', ValueTypes.STATIC_CATEGORICAL),
        ('family', ValueTypes.STATIC_CATEGORICAL),
        ('class', ValueTypes.STATIC_CATEGORICAL),
        ('perishable', ValueTypes.STATIC_CATEGORICAL)
    ]

    def __init__(self,
                 ts_len: int,
                 n_enc_steps: int,
                 sample_sz: int | None = None,
                 unknown_category_name: str = 'other',
                 **kwargs):
        super().__init__(ts_len=ts_len,
                         n_enc_steps=n_enc_steps,
                         sample_sz=sample_sz,
                         unknown_category_name=unknown_category_name,
                         **kwargs)

    def _download_dataset(self, dataset_name: str, download_dir: str) -> None:
        raise ValueError(
            """
            Dataset should be downloaded manually from Kaggle
            (https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data)
            and be placed under ``data/favorita`` 
            """)

    def _create_features(self):
        # a copy of the original implementation with small modifications
        # Unpack individually zipped files
        download_dir = self._make_dataset_dir()
        data_folder = os.path.join(download_dir, self.unzip_dir)
        shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)

        for file in glob.glob(os.path.join(data_folder, '*.7z')):
            csv_file = file.replace('.7z', '').split('/')[-1]
            if not csv_file in os.listdir(data_folder):
                print(f'Unzipping {file}')
                shutil.unpack_archive(file, data_folder)

        # Extract only a subset of data to save/process for efficiency
        start_date = pd.to_datetime('2015-1-1')
        end_date = pd.to_datetime('2016-6-1')

        print('Regenerating data...')

        # load temporal data
        temporal = pd.read_csv(os.path.join(data_folder, 'train.csv'), index_col=0)
        store_info = pd.read_csv(os.path.join(data_folder, 'stores.csv'), index_col=0)
        oil = pd.read_csv(os.path.join(data_folder, 'oil.csv'), index_col=0).iloc[:, 0]
        holidays = pd.read_csv(os.path.join(data_folder, 'holidays_events.csv'))
        items = pd.read_csv(os.path.join(data_folder, 'items.csv'), index_col=0)
        transactions = pd.read_csv(os.path.join(data_folder, 'transactions.csv'))

        # Take first 6 months of data
        temporal['date'] = pd.to_datetime(temporal['date'])

        # Filter dates to reduce storage space requirements
        if start_date is not None:
            temporal = temporal[(temporal['date'] >= start_date)].copy()
        if end_date is not None:
            temporal = temporal[(temporal['date'] < end_date)].copy()

        dates = temporal['date'].unique()
        # Add trajectory identifier
        temporal['traj_id'] = temporal['store_nbr']\
            .apply(str) + '_' + temporal['item_nbr']\
            .apply(str)
        temporal['unique_id'] = temporal['traj_id'] + '_' + temporal['date']\
            .apply(str)

        # Remove all IDs with negative returns
        print('Removing returns data')
        min_returns = temporal['unit_sales'].groupby(temporal['traj_id']).min()
        valid_ids = set(min_returns[min_returns >= 0].index)
        selector = temporal['traj_id'].apply(lambda traj_id: traj_id in valid_ids)
        new_temporal = temporal[selector].copy()
        del temporal
        gc.collect()
        temporal = new_temporal
        temporal['open'] = 1

        # Resampling
        print('Resampling to regular grid')
        resampled_dfs = []
        nunique_traj_ids = temporal['traj_id'].nunique()  # 143644
        n_ids = int(nunique_traj_ids/2)  # use half of the cases

        for i, (traj_id, raw_sub_df) in tqdm(enumerate(temporal.groupby('traj_id'))):
            sub_df = raw_sub_df.set_index('date', drop=True).copy()
            sub_df = sub_df.resample('1d').last()
            sub_df['date'] = sub_df.index
            sub_df[['store_nbr', 'item_nbr', 'onpromotion']] \
                = sub_df[['store_nbr', 'item_nbr', 'onpromotion']].ffill()
            sub_df['open'] = sub_df['open'].fillna(0)  # flag where sales data is unknown
            sub_df['log_sales'] = np.log(sub_df['unit_sales'])

            resampled_dfs.append(sub_df.reset_index(drop=True))
            if i == n_ids:  # use a fraction of the dataset since it doesn't fit in memory
                break

        new_temporal = pd.concat(resampled_dfs, axis=0)
        del temporal
        gc.collect()
        temporal = new_temporal

        print('Adding oil')
        oil.name = 'oil'
        # oil.index = pd.to_datetime(oil.index)
        oil = oil.reset_index()
        oil['date'] = pd.to_datetime(oil['date'])
        idx_oil = oil.set_index('date')
        filt_oil = oil.loc[idx_oil.index.isin(dates), :].reset_index(drop=True)
        # temporal = temporal.join(oil.loc[dates].ffill(), on='date', how='left')
        # temporal = temporal.reset_index(drop=True)  # remove date from index
        temporal = temporal.merge(filt_oil.ffill(), on='date', how='left')
        temporal['oil'] = temporal['oil'].fillna(-1)

        print('Adding store info')
        temporal = temporal.join(store_info, on='store_nbr', how='left')

        print('Adding item info')
        temporal = temporal.join(items, on='item_nbr', how='left')

        transactions['date'] = pd.to_datetime(transactions['date'])
        temporal = temporal.merge(
            transactions,
            left_on=['date', 'store_nbr'],
            right_on=['date', 'store_nbr'],
            how='left')
        temporal['transactions'] = temporal['transactions'].fillna(-1)

        # Additional date info
        temporal['day_of_week'] = pd.to_datetime(temporal['date'].values).dayofweek
        temporal['day_of_month'] = pd.to_datetime(temporal['date'].values).day
        temporal['month'] = pd.to_datetime(temporal['date'].values).month

        # Add holiday info
        print('Adding holidays')
        holiday_subset = holidays[holidays['transferred'].apply(
            lambda x: not x)].copy()
        holiday_subset.columns = [
            s if s != 'type' else 'holiday_type' for s in holiday_subset.columns
        ]
        holiday_subset['date'] = pd.to_datetime(holiday_subset['date'])
        local_holidays = holiday_subset[holiday_subset['locale'] == 'Local']
        regional_holidays = holiday_subset[holiday_subset['locale'] == 'Regional']
        national_holidays = holiday_subset[holiday_subset['locale'] == 'National']

        temporal['national_hol'] = temporal.merge(
            national_holidays, left_on=['date'], right_on=['date'],
            how='left')['description'].fillna('')
        temporal['regional_hol'] = temporal.merge(
            regional_holidays,
            left_on=['state', 'date'],
            right_on=['locale_name', 'date'],
            how='left')['description'].fillna('')
        temporal['local_hol'] = temporal.merge(
            local_holidays,
            left_on=['city', 'date'],
            right_on=['locale_name', 'date'],
            how='left')['description'].fillna('')

        temporal.sort_values('unique_id', inplace=True)
        out_fp = self._make_features_dataset_filepath()
        temporal.to_csv(out_fp, index=False)

    def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print('Formatting train-valid-test splits.')
        valid_boundary = None

        if valid_boundary is None:
            valid_boundary = pd.to_datetime('2015-12-1')

        time_steps = self._ts_len
        lookback = self._ts_len - self._n_enc_len
        forecast_horizon = time_steps - lookback

        df['date'] = pd.to_datetime(df['date'])
        df_lists = {'train': [], 'valid': [], 'test': []}  # type: ignore
        for _, sliced in tqdm(df.groupby('traj_id')):
            index = sliced['date']
            train = sliced.loc[index < valid_boundary]
            train_len = len(train)
            valid_len = train_len + forecast_horizon
            valid = sliced.iloc[train_len - lookback:valid_len+14, :]
            test = sliced.iloc[valid_len + 14 - lookback:, :]

            sliced_map = {'train': train, 'valid': valid, 'test': test}

            for k in sliced_map:
                item = sliced_map[k]

                if len(item) >= time_steps:
                    df_lists[k].append(item)

        train_df = pd.concat(df_lists['train'], axis=0)
        val_df = pd.concat(df_lists['valid'], axis=0)
        test_df = pd.concat(df_lists['test'], axis=0)

        return train_df, val_df, test_df
