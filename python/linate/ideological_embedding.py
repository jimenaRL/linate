""" LINATE module: Compute (Latent) Ideological Embedding """

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

import os.path

import pandas as pd

import numpy as np

from scipy.sparse import csr_matrix, issparse

from importlib import import_module

from sklearn import utils


class IdeologicalEmbedding(BaseEstimator, TransformerMixin):

    """

    Parameters
    ----------
    n_latent_dimensions: int, default=2
        Number of ideological embedding dimensions

    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization
        of ideological embedding model.

    n_iter: int, default=2
        Number of iteration in SVD computation.

    check_input: bool, default=True
        Sklearn valid input check.

    engine: str, default='auto'
        The user can specify the ideological embedding
        computation library to use.

    in_degree_threshold: int or None, default=Non
        Nodes that are followed by less than this number
        (in the original graph) are removed from network.

    out_degree_threshold: int or None, default=None
        Nodes that follow less than this number
        (in the original graph) are removed the network.

    force_bipartite: bool, default=True

    standardize_mean: bool, default=True

    standardize_std: bool, default=False

    force_full_rank: bool, default=False

    """

    ideological_embedding_class = 'CA'

    # by default use the 'prince' module to compute the ideological embedding
    default_ideological_embedding_engines = ['sklearn', 'auto', 'fbpca']

    def __init__(
        self,
        n_latent_dimensions=2,
        n_iter=10,
        check_input=True,
        random_state=None,
        engine='auto',
        in_degree_threshold=None,
        out_degree_threshold=None,
        force_bipartite=True,
        standardize_mean=True,
        standardize_std=False,
        force_full_rank=False,
    ):

        self.random_state = random_state
        self.check_input = check_input
        self.n_latent_dimensions = n_latent_dimensions
        self.n_iter = n_iter

        self.in_degree_threshold = in_degree_threshold
        self.out_degree_threshold = out_degree_threshold

        self.force_bipartite = force_bipartite
        self.standardize_mean = standardize_mean
        self.standardize_std = standardize_std
        self.force_full_rank = force_full_rank

        self.engine = engine
        self.check_engin_import()

    def check_engin_import(self):
        """
        Try to load engine module.

        engine: str
            Engine name
        """

        module_name = 'prince'

        if self.engine not in self.default_ideological_embedding_engines:
            module_name = self.engine

        try:
            self.ideological_embedding_module = import_module(module_name)
        except ModuleNotFoundError:
            raise ValueError(f"""{module_name} module is not installed.
            Please install it and make it visible to use it.""")

        print(f"Using Ideological Embedding engine '{module_name}'.")

    def fit(self, X):
        """
        Compute ..........

        Parameters
        ----------
        X: {array-like, sparse matrix or pandas dataframe}
        Data on which fit model.
        """

        X = X.copy()

        if self.force_full_rank:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    'To force full rank X needs to be a pandas dataframe...')

            if 'source' not in X.columns:
                raise ValueError(
                    "To force full rank X needs to have a 'source' column...")

            if 'target' not in X.columns:
                raise ValueError(
                    "To force full rank X needs to have a 'target' column...")

            # check if X is full rank
            follow_pattern_df = X \
                .sort_values('target') \
                .groupby('source') \
                .target.agg(' '.join)
            unique_follow_pattern = follow_pattern_df.drop_duplicates(
                inplace=False)
            unique_follow_pattern_idx = unique_follow_pattern.index.tolist()
            is_original_full_rank = False
            if len(X.index.tolist()) == len(unique_follow_pattern_idx):
                is_original_full_rank = True
            else:
                # original_X = X.copy()
                X = X[X['source'].isin(unique_follow_pattern_idx)]
            follow_pattern_df = follow_pattern_df \
                .to_frame() \
                .reset_index()

        if isinstance(X, pd.DataFrame):
            X = self.__check_input_and_convert_to_matrix(X)

        # check input
        if self.check_input:
            utils.check_array(X, accept_sparse=True)

        # set source and targer entity numbers
        if issparse(X):
            self.source_entity_no_ = X.get_shape()[0]
            self.target_entity_no_ = X.get_shape()[1]
        else:
            self.source_entity_no_ = X.shape[0]
            self.target_entity_no_ = X.shape[1]

        # and generate row and column IDs if needed
        try:
            len(self.column_ids_)
        except AttributeError:
            self.column_ids_ = np.empty(self.target_entity_no_, dtype=object)
            for indx in range(self.target_entity_no_):
                self.column_ids_[indx] = 'target_' + str(indx)
            self.row_ids_ = np.empty(self.source_entity_no_, dtype=object)
            for indx in range(self.source_entity_no_):
                self.row_ids_[indx] = 'source_' + str(indx)

        # compute number of ideological embedding dimensions to keep
        n_latent_dimensions_tmp = self.source_entity_no_
        if n_latent_dimensions_tmp > self.target_entity_no_:
            n_latent_dimensions_tmp = self.target_entity_no_
        if self.n_latent_dimensions < 0:
            self.employed_n_latent_dimensions = n_latent_dimensions_tmp
        else:
            if self.n_latent_dimensions > n_latent_dimensions_tmp:
                self.employed_n_latent_dimensions = n_latent_dimensions_tmp
            else:
                self.employed_n_latent_dimensions = self.n_latent_dimensions

        # compute Ideological Embedding of a (sparse) matrix
        print('Computing Ideological Embedding...')
        ideological_embedding_class = getattr(
            self.ideological_embedding_module,
            self.ideological_embedding_class)
        self.ideological_embedding_model = None
        if self.engine in self.default_ideological_embedding_engines:
            self.ideological_embedding_model = ideological_embedding_class(
                n_components=self.employed_n_latent_dimensions,
                n_iter=self.n_iter,
                check_input=False,
                engine=self.engine,
                random_state=self.random_state
            )
            self.ideological_embedding_model.fit(X)
        else:
            self.ideological_embedding_model = ideological_embedding_class(
                n_components=self.employed_n_latent_dimensions)
            self.ideological_embedding_model.fit(X)

        # finally construct the results
        if self.engine in self.default_ideological_embedding_engines:
            self.ideological_embedding_source_latent_dimensions_ = \
                self.ideological_embedding_model.row_coordinates(X)
        else:
            self.ideological_embedding_source_latent_dimensions_ = \
                self.ideological_embedding_model.row_coordinates()

        if self.engine in self.default_ideological_embedding_engines:
            self.ideological_embedding_target_latent_dimensions_ = \
                self.ideological_embedding_model.column_coordinates(X)
        else:
            self.ideological_embedding_target_latent_dimensions_ = \
                self.ideological_embedding_model.column_coordinates()

        if self.standardize_mean:
            std_scaler = StandardScaler(
                with_mean=self.standardize_mean,
                with_std=self.standardize_std)
            std_scaler.fit(
                pd.concat([
                    self.ideological_embedding_source_latent_dimensions_,
                    self.ideological_embedding_target_latent_dimensions_
                    ],
                    axis=0))

            target_scaled_dim = pd.DataFrame(
                columns=self.ideological_embedding_target_latent_dimensions_.columns,
                data=std_scaler.transform(self.ideological_embedding_target_latent_dimensions_))
            self.ideological_embedding_target_latent_dimensions_ = \
                target_scaled_dim

            source_scaled_dim = pd.DataFrame(
                columns=self.ideological_embedding_source_latent_dimensions_.columns,
                data=std_scaler.transform(self.ideological_embedding_source_latent_dimensions_))
            self.ideological_embedding_source_latent_dimensions_ = source_scaled_dim

        column_names = self.ideological_embedding_source_latent_dimensions_.columns
        new_column_names = []
        for c in column_names:
            new_column_names.append('latent_dimension_' + str(c))
        self.ideological_embedding_source_latent_dimensions_.columns = new_column_names
        self.ideological_embedding_source_latent_dimensions_.index = self.row_ids_
        self.ideological_embedding_source_latent_dimensions_.index.name = 'source_id'

        if self.force_full_rank:
            if not is_original_full_rank:
                self.ideological_embedding_source_latent_dimensions_ = \
                    self.ideological_embedding_source_latent_dimensions_.reset_index()
                source_target_map_df = pd.merge(
                    self.ideological_embedding_source_latent_dimensions_,
                    follow_pattern_df,
                    left_on='source_id',
                    right_on='source',
                    how='left')
                source_target_map_df.drop(
                    columns='source',
                    inplace=True)

                self.ideological_embedding_source_latent_dimensions_ = pd.merge(
                    follow_pattern_df,
                    source_target_map_df,
                    on='target',
                    how='inner')
                self.ideological_embedding_source_latent_dimensions_.drop(
                    columns=['target', 'source_id'],
                    inplace=True)
                self.ideological_embedding_source_latent_dimensions_.rename(
                    columns={'source': 'source_id'},
                    inplace=True)
                self.ideological_embedding_source_latent_dimensions_.index.name = 'source_id'

        column_names = self.ideological_embedding_target_latent_dimensions_.columns
        new_column_names = []
        for c in column_names:
            new_column_names.append('latent_dimension_' + str(c))
        self.ideological_embedding_target_latent_dimensions_.columns = \
            new_column_names
        self.ideological_embedding_target_latent_dimensions_.index = \
            self.column_ids_
        self.ideological_embedding_target_latent_dimensions_.index.name = \
            'target_id'

        # list
        self.eigenvalues_ = self.ideological_embedding_model.eigenvalues_
        # numpy.float64 or None
        self.candidate_total_inertia_ = \
            self.ideological_embedding_model.total_inertia_
        # list or None
        self.candidate_explained_inertia_ = \
            self.ideological_embedding_model.explained_inertia_

        return self

    def transform(self, X):
        return (X)

    def get_params(self, deep=True):
        return {'random_state': self.random_state,
                'check_input': self.check_input,
                'n_latent_dimensions': self.n_latent_dimensions,
                'n_iter': self.n_iter,
                'engine': self.engine,
                'in_degree_threshold': self.in_degree_threshold,
                'out_degree_threshold': self.out_degree_threshold,
                'force_bipartite': self.force_bipartite,
                'standardize_mean': self.standardize_mean,
                'standardize_std': self.standardize_std}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y):
        return 1

    def compute_latent_embedding_distance(
        self,
        Y,
        use_target_ideological_embedding=True,
        ideological_dimension_mapping=None,
        error_aggregation_fun=None
    ):

        try:
            X = None
            if use_target_ideological_embedding:
                X = self.ideological_embedding_target_latent_dimensions_.copy()

                # sort to achieve correspondence with Y
                X = X.sort_values('target ID', ascending=True)
            else:
                X = self.ideological_embedding_source_latent_dimensions_.copy()

                # sort to achieve correspondence with Y
                X = X.sort_values('source ID', ascending=True)
        except AttributeError:
            raise AttributeError(
                'Ideological Embedding model has not been fitted.')

        if X.index.name is not None:
            X = X.reset_index()

        if isinstance(X, pd.DataFrame):

            if ideological_dimension_mapping is None:
                # delete first column which is the entity ID
                X = X.iloc[:, 1:]
            else:
                if 'X' in ideological_dimension_mapping.keys():
                    X = X[ideological_dimension_mapping['X']]
                else:
                    X = X.iloc[:, 1:]

            X = X.to_numpy()

        assert isinstance(X, np.ndarray)

        if isinstance(Y, pd.DataFrame):
            if 'entity' not in Y.columns:
                raise ValueError(
                    "Missing 'entity' column in benchmark dimension data.")

            # sort to achieve correspondence with Y
            Y = Y.sort_values('entity', ascending=True)

            if ideological_dimension_mapping is None:
                # delete first column which is the entity ID
                Y = Y.iloc[:, 1:]
            else:
                if 'Y' in ideological_dimension_mapping.keys():
                    Y = Y[ideological_dimension_mapping['Y']]
                else:
                    Y = Y.iloc[:, 1:]

            Y = Y.to_numpy()

        assert isinstance(Y, np.ndarray)

        # X and Y should have the same dimensions
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Dimension matrices should have the same shape.')
        if X.shape[1] != Y.shape[1]:
            raise ValueError('Dimension matrices should have the same shape.')

        # compute Euclidean norm of each row
        l2 = np.sqrt(np.power(X - Y, 2.0).sum(axis=1))

        # default RMSE variant
        ideological_dim_dist = np.sqrt(np.power(l2, 2).sum()) / float(len(l2))
        ideological_dim_dist /= float(len(l2))

        if error_aggregation_fun is not None:
            if error_aggregation_fun == 'MAE':
                ideological_dim_dist = np.absolute(l2).sum() / float(len(l2))
            else:  # user defined
                ideological_dim_dist = error_aggregation_fun(l2)

        return (ideological_dim_dist)

    def load_benchmark_ideological_dimensions_from_file(
        self,
        path,
        header_names=None
    ):

        # check that benchmark ideological dimensions file is provided
        if path is None:
            raise ValueError(
                'Benchmark ideological dimensions file name is not provided.')

        # check that benchmark ideological dimensions file exists
        if not os.path.isfile(path):
            raise ValueError(
                'Benchmark ideological dimensions file does not exist.')

        # handles files with or without header
        header_df = pd.read_csv(path, nrows=0)
        column_no = len(header_df.columns)
        if column_no < 2:
            raise ValueError(
                'Benchmark ideological dimensions file has only two columns.')

        # sanity checks in header
        if header_names is not None:
            if header_names['entity'] not in header_df.columns:
                raise ValueError(
                    'Benchmark ideological dimensions file has to have a '
                    + header_names['entity'] + ' column.')

        # load data
        input_df = None
        if header_names is None:
            input_df = pd.read_csv(path, header=None).rename(
                columns={0: 'entity'})
        else:
            input_df = pd.read_csv(path).rename(
                columns={header_names['entity']: 'entity'})

        if header_names is not None:
            if 'dimensions' in header_names.keys():
                cols = header_names['dimensions']
                cols.append('entity')
                input_df = input_df[cols]

        input_df['entity'] = input_df['entity'].astype(str)
        for c in input_df.columns:
            if c != 'entity':
                input_df[c] = input_df[c].astype(float)

        return (input_df)

    def load_input_from_file(
        self,
        path_to_network_data,
        header_names=None
    ):

        # check that a network file is provided
        if path_to_network_data is None:
            raise ValueError('Network file name is not provided.')

        # check network file exists
        if not os.path.isfile(path_to_network_data):
            raise ValueError('Network file does not exist.')

        # handles files with or without header
        header_df = pd.read_csv(path_to_network_data, nrows=0)
        column_no = len(header_df.columns)
        if column_no < 2:
            raise ValueError('Network file has to have at least two columns.')

        # sanity checks in header
        _source = header_names['source']
        _target = header_names['target']
        if header_names is not None:
            if _source not in header_df.columns:
                raise ValueError(
                    f'Network file has to have a {_source} column.')
            if _target not in header_df.columns:
                raise ValueError(
                    f'Network file has to have a {_target} column.')

        # load network data
        input_df = None
        if header_names is None:
            if column_no == 2:
                input_df = pd.read_csv(
                    path_to_network_data,
                    header=None,
                    dtype={0: str, 1: str}
                ).rename(columns={0: 'source', 1: 'target'})
            else:
                input_df = pd.read_csv(
                    path_to_network_data,
                    header=None,
                    dtype={0: str, 1: str}
                ).rename(columns={0: 'source', 1: 'target', 2: 'multiplicity'})
        else:
            if 'multiplicity' in header_names.keys():
                input_df = pd.read_csv(
                    path_to_network_data,
                    dtype={
                        header_names['source']: str,
                        header_names['target']: str
                    }).rename(
                        columns={
                            header_names['source']: 'source',
                            header_names['target']: 'target',
                            header_names['multiplicity']: 'multiplicity'})
            else:
                input_df = pd.read_csv(
                    path_to_network_data,
                    dtype={
                        header_names['source']: str,
                        header_names['target']: str
                    }).rename(
                        columns={
                            header_names['source']: 'source',
                            header_names['target']: 'target'})

        print('Load network ended.')

        return (input_df)

    def save_ideological_embedding_latent_dimensions(self, kind, path_to_file):
        attr_name = f'ideological_embedding_{kind}_latent_dimensions_'
        if not hasattr(self, attr_name):
            raise AttributeError(
                f'{kind} ideological embedding latent dimensions have \
                not been computed.')
        getattr(self, attr_name).to_csv(path_to_file)

    @property
    def total_inertia_(self):
        try:
            if self.engine in self.default_ideological_embedding_engines:
                return (self.candidate_total_inertia_)

            self.candidate_total_inertia_ = self.ideological_embedding_model.get_total_inertia()
            return (self.candidate_total_inertia_)
        except AttributeError:
            raise AttributeError('Ideological Embedding model not yet fitted.')

    @property
    def explained_inertia_(self):
        try:
            if self.engine in self.default_ideological_embedding_engines:
                return (self.candidate_explained_inertia_)

            self.candidate_total_inertia_ = self.ideological_embedding_model\
                .get_explained_inertia()
            return (self.candidate_total_inertia_)
        except AttributeError:
            raise AttributeError('Ideological Embedding model not yet fitted.')

    def __check_input_and_convert_to_matrix(self, input_df):
        # first perform validity checks over the input
        if not isinstance(input_df, pd.DataFrame):
            raise ValueError('Input should be a pandas dataframe.')

        if 'source' not in input_df.columns:
            raise ValueError('Input dataframe should have a source column.')

        if 'target' not in input_df.columns:
            raise ValueError('Input dataframe should have a target column.')

        # remove NAs from input data
        input_df.dropna(subset=['source', 'target'], inplace=True)

        # convert to 'str'
        input_df['source'] = input_df['source'].astype(str)
        input_df['target'] = input_df['target'].astype(str)

        # the file should either have repeated edges or a multiplicity column
        # but not both has_more_columns = True
        # if input_df.columns.size > 2 else False
        has_repeated_edges = True if input_df.duplicated(
            subset=['source', 'target']).sum() > 0 else False
        if ('multiplicity' in input_df.columns) and has_repeated_edges:
            raise ValueError(
                'There cannot be repeated edges AND a 3rd column \
                with edge multiplicities.')

        if 'multiplicity' in input_df.columns:
            # will fail if missing element, or cannot convert
            input_df['multiplicity'] = input_df['multiplicity'].astype(int)

        # checking if final network is bipartite:
        common_nodes_np = np.intersect1d(
            input_df['source'], input_df['target'])
        self.is_bipartite_ = common_nodes_np.size == 0
        if not self.is_bipartite_:
            if self.force_bipartite:
                input_df = input_df[~input_df['source'].isin(common_nodes_np)]
        print('Bipartite network:', self.is_bipartite_)

        # remove nodes with small degree if needed
        degree_per_target = None
        if self.in_degree_threshold is not None:
            degree_per_target = input_df.groupby('target').count()

        degree_per_source = None
        if self.out_degree_threshold is not None:
            degree_per_source = input_df.groupby('source').count()

        if degree_per_target is not None:
            if 'multiplicity' in degree_per_target.columns:
                degree_per_target.drop('multiplicity', axis=1, inplace=True)
            degree_per_target = degree_per_target[
                degree_per_target >= self.in_degree_threshold] \
                .dropna().reset_index()
            degree_per_target.drop('source', axis=1, inplace=True)
            input_df = pd.merge(
                input_df,
                degree_per_target,
                on=['target'],
                how='inner')

        if degree_per_source is not None:
            if 'multiplicity' in degree_per_source.columns:
                degree_per_source.drop(
                    'multiplicity',
                    axis=1,
                    inplace=True
                )
            degree_per_source = degree_per_source[
                degree_per_source >= self.out_degree_threshold] \
                .dropna().reset_index()
            degree_per_source.drop('target', axis=1, inplace=True)
            input_df = pd.merge(
                input_df,
                degree_per_source,
                on=['source'],
                how='inner'
            )

        # and then assemble the matrices to be fed to Ideological embedding
        ntwrk_df = input_df[['source', 'target']]

        n_i, r = ntwrk_df['target'].factorize()
        self.column_ids_ = r.values
        n_j, c = ntwrk_df['source'].factorize()
        assert len(n_i) == len(n_j)
        self.row_ids_ = c.values
        network_edge_no = len(n_i)
        n_in_j, tups = pd.factorize(list(zip(n_j, n_i)))
        # COO might be faster
        ntwrk_csr = csr_matrix((np.bincount(n_in_j), tuple(zip(*tups))))

        if self.engine in self.default_ideological_embedding_engines:
            ntwrk_np = ntwrk_csr.toarray()
            return (ntwrk_np)

        return (ntwrk_csr)
