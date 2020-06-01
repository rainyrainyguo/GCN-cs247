""" Experiment runner for the model with knowledge graph attached to interaction data """

from __future__ import division
from __future__ import print_function

import argparse
import datetime
import time

import tensorflow as tf
import sys
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
from zipfile import ZipFile
from io import StringIO
import shutil
import os.path

import pickle
import os
import h5py
from urllib.request import urlopen

from gcmc.model import RecommenderGAE, RecommenderSideInfoGAE
from gcmc.utils import construct_feed_dict


#data_utils.py
def data_iterator(data, batch_size):
    """
    A simple data iterator from https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
    :param data: list of numpy tensors that need to be randomly batched across their first dimension.
    :param batch_size: int, batch_size of data_iterator.
    Assumes same first dimension size of all numpy tensors.
    :return: iterator over batches of numpy tensors
    """
    # shuffle labels and features
    max_idx = len(data[0])
    idxs = np.arange(0, max_idx)
    np.random.shuffle(idxs)
    shuf_data = [dat[idxs] for dat in data]

    # Does not yield last remainder of size less than batch_size
    for i in range(max_idx//batch_size):
        data_batch = [dat[i*batch_size:(i+1)*batch_size] for dat in shuf_data]
        yield data_batch


def map_data(data):
    """
    Map data to proper indices in case they are not in a continues [0, N) range

    Parameters
    ----------
    data : np.int32 arrays

    Returns
    -------
    mapped_data : np.int32 arrays
    n : length of mapped_data

    """
    uniq = list(set(data))

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    #data = np.array(map(lambda x: id_dict[x], data))
    data = pd.Series(data).apply(lambda x: id_dict[x]).values
    n = len(uniq)

    return data, id_dict, n


def download_dataset(dataset, files, data_dir):
    """ Downloads dataset if files are not present. """

    if not np.all([os.path.isfile(data_dir + f) for f in files]):
        url = "http://files.grouplens.org/datasets/movielens/" + dataset.replace('_', '-') + '.zip'
        request = urlopen(url)

        print('Downloading %s dataset' % dataset)
        if dataset in ['ml_100k', 'ml_1m']:
            target_dir = 'data/' + dataset.replace('_', '-')
        elif dataset == 'ml_10m':
            target_dir = 'data/' + 'ml-10M100K'
        else:
            raise ValueError('Invalid dataset option %s' % dataset)

        with ZipFile(StringIO(request.read())) as zip_ref:
            zip_ref.extractall('data/')

        source = [target_dir + '/' + s for s in os.listdir(target_dir)]
        destination = data_dir+'/'
        for f in source:
            shutil.copy(f, destination)

        shutil.rmtree(target_dir)


def load_data(fname, seed=1234, verbose=True):
    """ Loads dataset and creates adjacency matrix
    and feature matrix

    Parameters
    ----------
    fname : str, dataset
    seed: int, dataset shuffling seed
    verbose: to print out statements or not

    Returns
    -------
    num_users : int
        Number of users and items respectively

    num_items : int

    u_nodes : np.int32 arrays
        User indices

    v_nodes : np.int32 array
        item (movie) indices

    ratings : np.float32 array
        User/item ratings s.t. ratings[k] is the rating given by user u_nodes[k] to
        item v_nodes[k]. Note that that the all pairs u_nodes[k]/v_nodes[k] are unique, but
        not necessarily all u_nodes[k] or all v_nodes[k] separately.

    u_features: np.float32 array, or None
        If present in dataset, contains the features of the users.

    v_features: np.float32 array, or None
        If present in dataset, contains the features of the users.

    seed: int,
        For datashuffling seed with pythons own random.shuffle, as in CF-NADE.

    """

    u_features = None
    v_features = None

    print('Loading dataset', fname)

    data_dir = 'data/' + fname

    if fname == 'ml_100k':

        # Check if files exist and download otherwise
        files = ['/u.data', '/u.item', '/u.user']

        download_dataset(fname, files, data_dir)

        sep = '\t'
        filename = data_dir + files[0]

        dtypes = {
            'u_nodes': np.int32, 'v_nodes': np.int32,
            'ratings': np.float32, 'timestamp': np.float64}

        data = pd.read_csv(
            filename, sep=sep, header=None,
            names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.as_matrix().tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
        ratings = ratings.astype(np.float64)

        # Movie features (genres)
        sep = r'|'
        movie_file = data_dir + files[1]
        movie_headers = ['movie id', 'movie title', 'release date', 'video release date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                               names=movie_headers, engine='python')

        genre_headers = movie_df.columns.values[6:]
        num_genres = genre_headers.shape[0]

        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, g_vec in zip(movie_df['movie id'].values.tolist(), movie_df[genre_headers].values.tolist()):
            # Check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                v_features[v_dict[movie_id], :] = g_vec

        # User features

        sep = r'|'
        users_file = data_dir + files[2]
        users_headers = ['user id', 'age', 'gender', 'occupation', 'zip code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        occupation = set(users_df['occupation'].values.tolist())

        gender_dict = {'M': 0., 'F': 1.}
        occupation_dict = {f: i for i, f in enumerate(occupation, start=2)}

        num_feats = 2 + len(occupation_dict)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user id']
            if u_id in u_dict.keys():
                # age
                u_features[u_dict[u_id], 0] = row['age']
                # gender
                u_features[u_dict[u_id], 1] = gender_dict[row['gender']]
                # occupation
                u_features[u_dict[u_id], occupation_dict[row['occupation']]] = 1.

        u_features = sp.csr_matrix(u_features)
        v_features = sp.csr_matrix(v_features)
        

    if verbose:
        print('Number of users = %d' % num_users)
        print('Number of items = %d' % num_items)
        print('Number of links = %d' % ratings.shape[0])
        print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

    return num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings, u_features, v_features



# preprocessing.py
def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf

    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    if feat_norm.nnz == 0:
        print('ERROR: normalized adjacency matrix has only zero entries!!!!!')
        exit

    return feat_norm


def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out


def preprocess_user_item_features(u_features, v_features):
    """
    Creates one big feature matrix out of user features and item features.
    Stacks item features under the user features.
    """

    zero_csr_u = sp.csr_matrix((u_features.shape[0], v_features.shape[1]), dtype=u_features.dtype)
    zero_csr_v = sp.csr_matrix((v_features.shape[0], u_features.shape[1]), dtype=v_features.dtype)

    u_features = sp.hstack([u_features, zero_csr_u], format='csr')
    v_features = sp.hstack([zero_csr_v, v_features], format='csr')

    return u_features, v_features


def globally_normalize_bipartite_adjacency(adjacencies, verbose=False, symmetric=True):
    """ Globally Normalizes set of bipartite adjacency matrices """

    if verbose:
        print('Symmetrically normalizing bipartite adj')
    # degree_u and degree_v are row and column sums of adj+I

    adj_tot = np.sum(adj for adj in adjacencies)
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf

    degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
    degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
    degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])

    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

    if symmetric:
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(degree_v_inv_sqrt_mat) for adj in adjacencies]

    else:
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]

    return adj_norm


def sparse_to_tuple(sparse_mx):
    """ change of format for sparse matrix. This format is used
    for the feed_dict where sparse matrices need to be linked to placeholders
    representing sparse matrices. """

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def load_ourdata(now_phase = 1, testing = True):
    train_path = '../../underexpose_train'  
    test_path = '../../underexpose_test'  
    recom_item = []  
    
    dtypes = {
    'u_nodes': np.int32, 'v_nodes': np.int32,
    'ratings': np.float32, 'timestamp': np.float64}

    data_train = pd.DataFrame()  
    data_test = pd.DataFrame()
    for c in range(now_phase + 1):  
        print('phase:', c)  
        click_train = pd.read_csv(train_path + '/underexpose_train_click-{}-real.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
        click_test = pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(c,c), header=None,  names=['user_id', 'item_id', 'time'])  
        #answer = pd.read_csv(test_path + '/underexpose_test_qtime_with_answer-{}.csv'.format(c), header=None, names=['user_id', 'item_id', 'time'])

        all_click = click_train.append(click_test)  
        data_train =data_train.append(all_click)  
        data_train = data_train.drop_duplicates(subset=['user_id','item_id','time'],keep='last')
        #data_train = data_train.sort_values('time')
        #data_test = data_test.append(answer)


    df = data_train[['user_id','item_id']]
    df2 = df.sort_values(by=['user_id','item_id']).drop_duplicates()
    df2['ratings'] = df.groupby(['user_id', 'item_id']).size().values
    data_train = df2.reset_index().drop(columns=['index'], axis=1)
    data_train.columns = ['u_nodes','v_nodes','ratings']
    
    n = int(len(data_train)*0.1)
    data_test = data_train[:n]
    data_train = data_train[n:]

    data_array_train = data_train.as_matrix().tolist()
    data_array_train = np.array(data_array_train)
    data_array_test = data_test.as_matrix().tolist()
    data_array_test = np.array(data_array_test)

    data_array = np.concatenate([data_array_train, data_array_test], axis=0)

    u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])

    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(u_nodes)):
        assert(labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number of test and validation edges, see cf-nade code

    num_train = data_array_train.shape[0]
    num_test = data_array_test.shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    for i in range(len(ratings)):
        assert(labels[idx_nonzero[i]] == rating_dict[ratings[i]])

    idx_nonzero_train = idx_nonzero[0:num_train+num_val]
    idx_nonzero_test = idx_nonzero[num_train+num_val:]

    pairs_nonzero_train = pairs_nonzero[0:num_train+num_val]
    pairs_nonzero_test = pairs_nonzero[num_train+num_val:]

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    assert(len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    class_values = np.sort(np.unique(ratings))


    item_df = pd.read_csv('../../underexpose_train/underexpose_item_feat.csv', header=None)

    item_df[0] = item_df[0].apply(lambda x: int(x))
    item_df[1] = item_df[1].apply(lambda x: float(x.strip('[')))
    item_df[128] = item_df[128].apply(lambda x: float(x.strip(']')))
    item_df[129] = item_df[129].apply(lambda x: float(x.strip('[')))
    item_df[256] = item_df[256].apply(lambda x: float(x.strip(']')))

    item_df = item_df.rename(columns = {0:'item_id'})

    for col in range(1, len(item_df.columns)):
        item_df[col] = item_df[col].values/item_df[col].values.max()

    def check(v_id):
        if v_id in v_dict.keys():
            return v_dict[v_id]
        return False

    num_feats = len(item_df.columns) - 1
    v_features = np.zeros((num_items, num_feats), dtype=np.float32)

    sorted_by_value = item_df.item_id.apply(lambda v_id: check(v_id)).sort_values(ascending=True)[60678:]
    dataframe_index = list(sorted_by_value.index)
    matrix_index = list(sorted_by_value.values)

    v_features[matrix_index] = item_df.iloc[dataframe_index][list(item_df.columns)[1:]].values

    user_df = pd.read_csv('../../underexpose_train/underexpose_user_feat.csv', header=None)

    user_df.columns = ['user_id', 'user_age_level', 'user_gender', 'user_city_level']
    user_df.fillna(0,inplace=True)

    user_age_dict = {f:i for i,f in enumerate(np.sort(user_df['user_age_level'].unique()))}
    user_gender_dict = {0:9, 'F':10, 'M':11}
    user_city_dict = {f:i for i,f in enumerate(np.sort(user_df['user_city_level'].unique()), start=12)}

    num_feats = len(user_age_dict) + len(user_gender_dict) + len(user_city_dict)

    u_features = np.zeros((num_users, num_feats), dtype=np.float32)
    for _, row in user_df.iterrows():
        u_id = row['user_id']
        if u_id in u_dict.keys():
            # age
            u_features[u_dict[u_id], user_age_dict[row['user_age_level']]] = 1
            # gender
            u_features[u_dict[u_id], user_gender_dict[row['user_gender']]] = 1
            # occupation
            u_features[u_dict[u_id], user_city_dict[row['user_city_level']]] = 1.

    u_features = sp.csr_matrix(u_features)
    v_features = sp.csr_matrix(v_features)

    print("User features shape: "+str(u_features.shape))
    print("Item features shape: "+str(v_features.shape))

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values


# args = {'dataset': 'ml_100k', 'learning_rate': 0.01, 'epochs': 1000, 'hidden': [500, 75], 'feat_hidden': 10, 'accumulation': 'stack', 'dropout': 0.7, 'num_basis_functions': 2, 'data_seed': 1234, 'summaries_dir': 'logs/2020-05-31_09:02:14.912270', 'norm_symmetric': False, 'features': True, 'write_summary': False, 'testing': True}

# Set random seed
# seed = 123 # use only for unit testing
seed = int(time.time())
np.random.seed(seed)
tf.set_random_seed(seed)


# Settings
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="ml_1m",
                choices=['ml_100k', 'ml_1m', 'ml_10m', 'douban', 'yahoo_music', 'flixster','taobao'],
                help="Dataset string.")

ap.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                help="Learning rate")

ap.add_argument("-e", "--epochs", type=int, default=2500,
                help="Number training epochs")

ap.add_argument("-nc", "--numclasses", type=int, default=3,
                help="Number of classes")

ap.add_argument("-ph", "--phase", type=int, default=2,
                help="Current phase")

ap.add_argument("-hi", "--hidden", type=int, nargs=2, default=[500, 75],
                help="Number hidden units in 1st and 2nd layer")

ap.add_argument("-fhi", "--feat_hidden", type=int, default=64,
                help="Number hidden units in the dense layer for features")

ap.add_argument("-ac", "--accumulation", type=str, default="sum", choices=['sum', 'stack'],
                help="Accumulation function: sum or stack.")

ap.add_argument("-do", "--dropout", type=float, default=0.7,
                help="Dropout fraction")

ap.add_argument("-nb", "--num_basis_functions", type=int, default=2,
                help="Number of basis functions for Mixture Model GCN.")

ap.add_argument("-ds", "--data_seed", type=int, default=1234,
                help="""Seed used to shuffle data in data_utils, taken from cf-nade (1234, 2341, 3412, 4123, 1324).
                     Only used for ml_1m and ml_10m datasets. """)

ap.add_argument("-sdir", "--summaries_dir", type=str, default='logs/' + str(datetime.datetime.now()).replace(' ', '_'),
                help="Directory for saving tensorflow summaries.")

# Boolean flags
fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-nsym', '--norm_symmetric', dest='norm_symmetric',
                help="Option to turn on symmetric global normalization", action='store_true')
fp.add_argument('-nleft', '--norm_left', dest='norm_symmetric',
                help="Option to turn on left global normalization", action='store_false')
ap.set_defaults(norm_symmetric=True)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-f', '--features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_true')
fp.add_argument('-no_f', '--no_features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_false')
ap.set_defaults(features=False)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-ws', '--write_summary', dest='write_summary',
                help="Option to turn on summary writing", action='store_true')
fp.add_argument('-no_ws', '--no_write_summary', dest='write_summary',
                help="Option to turn off summary writing", action='store_false')
ap.set_defaults(write_summary=False)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-t', '--testing', dest='testing',
                help="Option to turn on test set evaluation", action='store_true')
fp.add_argument('-v', '--validation', dest='testing',
                help="Option to only use validation set evaluation", action='store_false')
ap.set_defaults(testing=False)


args = vars(ap.parse_args())

print('Settings:')
print(args, '\n')



# Define parameters
DATASET = args['dataset']
DATASEED = args['data_seed']
NB_EPOCH = args['epochs']
DO = args['dropout']
HIDDEN = args['hidden']
FEATHIDDEN = args['feat_hidden']
BASES = args['num_basis_functions']
LR = args['learning_rate']
WRITESUMMARY = args['write_summary']
SUMMARIESDIR = args['summaries_dir']
FEATURES = args['features']
SYM = args['norm_symmetric']
TESTING = args['testing']
ACCUM = args['accumulation']
PHASE = args['phase']
NUMCLASSES = args['numclasses']

SELFCONNECTIONS = False
SPLITFROMFILE = True
VERBOSE = True


# Splitting dataset in training, validation and test set

# if DATASET == 'ml_1m' or DATASET == 'ml_10m':
#     if FEATURES:
#         datasplit_path = 'data/' + DATASET + '/withfeatures_split_seed' + str(DATASEED) + '.pickle'
#     else:
#         datasplit_path = 'data/' + DATASET + '/split_seed' + str(DATASEED) + '.pickle'
# elif FEATURES:
#     datasplit_path = 'data/' + DATASET + '/withfeatures.pickle'
# else:
#     datasplit_path = 'data/' + DATASET + '/nofeatures.pickle'

u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
    val_labels, val_u_indices, val_v_indices, test_labels, \
    test_u_indices, test_v_indices, class_values = load_ourdata(now_phase = PHASE, testing = TESTING)

num_users, num_items = adj_train.shape

num_side_features = 0

# feature loading
if not FEATURES:
    u_features = sp.identity(num_users, format='csr')
    v_features = sp.identity(num_items, format='csr')

    u_features, v_features = preprocess_user_item_features(u_features, v_features)

elif FEATURES and u_features is not None and v_features is not None:
    # use features as side information and node_id's as node input features

    print("Normalizing feature vectors...")
    u_features_side = normalize_features(u_features)
    v_features_side = normalize_features(v_features)

    u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)

    u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
    v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

    num_side_features = u_features_side.shape[1]

    # node id's for node input features
    id_csr_v = sp.identity(num_items, format='csr')
    id_csr_u = sp.identity(num_users, format='csr')

    u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)

else:
    raise ValueError('Features flag is set to true but no features are loaded from dataset ' + DATASET)


# global normalization
support = []
support_t = []
adj_train_int = sp.csr_matrix(adj_train, dtype=np.int32)

for i in range(NUMCLASSES):
    # build individual binary rating matrices (supports) for each rating
    support_unnormalized = sp.csr_matrix(adj_train_int == i + 1, dtype=np.float32)

    if support_unnormalized.nnz == 0 and DATASET != 'yahoo_music':
        # yahoo music has dataset split with not all ratings types present in training set.
        # this produces empty adjacency matrices for these ratings.
        sys.exit('ERROR: normalized bipartite adjacency matrix has only zero entries!!!!!')

    support_unnormalized_transpose = support_unnormalized.T
    support.append(support_unnormalized)
    support_t.append(support_unnormalized_transpose)


support = globally_normalize_bipartite_adjacency(support, symmetric=SYM)
support_t = globally_normalize_bipartite_adjacency(support_t, symmetric=SYM)

if SELFCONNECTIONS:
    support.append(sp.identity(u_features.shape[0], format='csr'))
    support_t.append(sp.identity(v_features.shape[0], format='csr'))

num_support = len(support)
support = sp.hstack(support, format='csr')
support_t = sp.hstack(support_t, format='csr')

if ACCUM == 'stack':
    div = HIDDEN[0] // num_support
    if HIDDEN[0] % num_support != 0:
        print("""\nWARNING: HIDDEN[0] (=%d) of stack layer is adjusted to %d such that
                  it can be evenly split in %d splits.\n""" % (HIDDEN[0], num_support * div, num_support))
    HIDDEN[0] = num_support * div

# Collect all user and item nodes for test set
test_u = list(set(test_u_indices))
test_v = list(set(test_v_indices))
test_u_dict = {n: i for i, n in enumerate(test_u)}
test_v_dict = {n: i for i, n in enumerate(test_v)}

test_u_indices = np.array([test_u_dict[o] for o in test_u_indices])
test_v_indices = np.array([test_v_dict[o] for o in test_v_indices])

test_support = support[np.array(test_u)]
test_support_t = support_t[np.array(test_v)]

# Collect all user and item nodes for validation set
val_u = list(set(val_u_indices))
val_v = list(set(val_v_indices))
val_u_dict = {n: i for i, n in enumerate(val_u)}
val_v_dict = {n: i for i, n in enumerate(val_v)}

val_u_indices = np.array([val_u_dict[o] for o in val_u_indices])
val_v_indices = np.array([val_v_dict[o] for o in val_v_indices])

val_support = support[np.array(val_u)]
val_support_t = support_t[np.array(val_v)]

# Collect all user and item nodes for train set
train_u = list(set(train_u_indices))
train_v = list(set(train_v_indices))
train_u_dict = {n: i for i, n in enumerate(train_u)}
train_v_dict = {n: i for i, n in enumerate(train_v)}

train_u_indices = np.array([train_u_dict[o] for o in train_u_indices])
train_v_indices = np.array([train_v_dict[o] for o in train_v_indices])

train_support = support[np.array(train_u)]
train_support_t = support_t[np.array(train_v)]

# features as side info
if FEATURES:
    test_u_features_side = u_features_side[np.array(test_u)]
    test_v_features_side = v_features_side[np.array(test_v)]

    val_u_features_side = u_features_side[np.array(val_u)]
    val_v_features_side = v_features_side[np.array(val_v)]

    train_u_features_side = u_features_side[np.array(train_u)]
    train_v_features_side = v_features_side[np.array(train_v)]

else:
    test_u_features_side = None
    test_v_features_side = None

    val_u_features_side = None
    val_v_features_side = None

    train_u_features_side = None
    train_v_features_side = None

placeholders = {
    'u_features': tf.sparse_placeholder(tf.float32, shape=np.array(u_features.shape, dtype=np.int64)),
    'v_features': tf.sparse_placeholder(tf.float32, shape=np.array(v_features.shape, dtype=np.int64)),
    'u_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'v_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'labels': tf.placeholder(tf.int32, shape=(None,)),

    'u_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),
    'v_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),

    'user_indices': tf.placeholder(tf.int32, shape=(None,)),
    'item_indices': tf.placeholder(tf.int32, shape=(None,)),

    'class_values': tf.placeholder(tf.float32, shape=class_values.shape),

    'dropout': tf.placeholder_with_default(0., shape=()),
    'weight_decay': tf.placeholder_with_default(0., shape=()),

    'support': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'support_t': tf.sparse_placeholder(tf.float32, shape=(None, None)),
}

# create model
if FEATURES:
    model = RecommenderSideInfoGAE(placeholders,
                                   input_dim=u_features.shape[1],
                                   feat_hidden_dim=FEATHIDDEN,
                                   num_classes=NUMCLASSES,
                                   num_support=num_support,
                                   self_connections=SELFCONNECTIONS,
                                   num_basis_functions=BASES,
                                   hidden=HIDDEN,
                                   num_users=num_users,
                                   num_items=num_items,
                                   accum=ACCUM,
                                   learning_rate=LR,
                                   num_side_features=num_side_features,
                                   logging=True)

# Convert sparse placeholders to tuples to construct feed_dict
test_support = sparse_to_tuple(test_support)
test_support_t = sparse_to_tuple(test_support_t)

val_support = sparse_to_tuple(val_support)
val_support_t = sparse_to_tuple(val_support_t)

train_support = sparse_to_tuple(train_support)
train_support_t = sparse_to_tuple(train_support_t)

u_features = sparse_to_tuple(u_features)
v_features = sparse_to_tuple(v_features)
assert u_features[2][1] == v_features[2][1], 'Number of features of users and items must be the same!'

num_features = u_features[2][1]
u_features_nonzero = u_features[1].shape[0]
v_features_nonzero = v_features[1].shape[0]

# Feed_dicts for validation and test set stay constant over different update steps
train_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                      v_features_nonzero, train_support, train_support_t,
                                      train_labels, train_u_indices, train_v_indices, class_values, DO,
                                      train_u_features_side, train_v_features_side)
# No dropout for validation and test runs
val_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                    v_features_nonzero, val_support, val_support_t,
                                    val_labels, val_u_indices, val_v_indices, class_values, 0.,
                                    val_u_features_side, val_v_features_side)

test_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                     v_features_nonzero, test_support, test_support_t,
                                     test_labels, test_u_indices, test_v_indices, class_values, 0.,
                                     test_u_features_side, test_v_features_side)


# Collect all variables to be logged into summary
merged_summary = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if WRITESUMMARY:
    train_summary_writer = tf.summary.FileWriter(SUMMARIESDIR + '/train', sess.graph)
    val_summary_writer = tf.summary.FileWriter(SUMMARIESDIR + '/val')
else:
    train_summary_writer = None
    val_summary_writer = None

best_val_score = np.inf
best_val_loss = np.inf
best_epoch = 0
wait = 0


print('Training...')

for epoch in range(NB_EPOCH):

    t = time.time()

    # Run single weight update
    # outs = sess.run([model.opt_op, model.loss, model.rmse], feed_dict=train_feed_dict)
    # with exponential moving averages
    outs = sess.run([model.training_op, model.loss, model.rmse], feed_dict=train_feed_dict)

    train_avg_loss = outs[1]
    train_rmse = outs[2]

    val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)

    if VERBOSE:
        print("[*] Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_avg_loss),
              "train_rmse=", "{:.5f}".format(train_rmse),
              "val_loss=", "{:.5f}".format(val_avg_loss),
              "val_rmse=", "{:.5f}".format(val_rmse),
              "\t\ttime=", "{:.5f}".format(time.time() - t))

    if val_rmse < best_val_score:
        best_val_score = val_rmse
        best_epoch = epoch

    if epoch % 20 == 0 and WRITESUMMARY:
        # Train set summary
        summary = sess.run(merged_summary, feed_dict=train_feed_dict)
        train_summary_writer.add_summary(summary, epoch)
        train_summary_writer.flush()

        # Validation set summary
        summary = sess.run(merged_summary, feed_dict=val_feed_dict)
        val_summary_writer.add_summary(summary, epoch)
        val_summary_writer.flush()

    if epoch % 100 == 0 and epoch > 300 and not TESTING and False:
        saver = tf.train.Saver()
        save_path = saver.save(sess, "tmp/%s_seed%d.ckpt" % (model.name, DATASEED), global_step=model.global_step)

        # load polyak averages
        variables_to_restore = model.variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, save_path)

        val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)

        print('polyak val loss = ', val_avg_loss)
        print('polyak val rmse = ', val_rmse)

        # Load back normal variables
        saver = tf.train.Saver()
        saver.restore(sess, save_path)
        
    if epoch % 100 == 0 and epoch > 300:
        saver = tf.train.Saver()
        save_path = saver.save(sess, "tmp/%s_seed%d.ckpt" % (model.name, DATASEED), global_step=model.global_step)        
        


# store model including exponential moving averages
saver = tf.train.Saver()
save_path = saver.save(sess, "tmp/%s.ckpt" % model.name, global_step=model.global_step)


if VERBOSE:
    print("\nOptimization Finished!")
    print('best validation score =', best_val_score, 'at iteration', best_epoch)


if TESTING:
    test_avg_loss, test_rmse = sess.run([model.loss, model.rmse], feed_dict=test_feed_dict)
    print('test loss = ', test_avg_loss)
    print('test rmse = ', test_rmse)

    # restore with polyak averages of parameters
    variables_to_restore = model.variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, save_path)

    test_avg_loss, test_rmse = sess.run([model.loss, model.rmse], feed_dict=test_feed_dict)
    print('polyak test loss = ', test_avg_loss)
    print('polyak test rmse = ', test_rmse)

else:
    # restore with polyak averages of parameters
    variables_to_restore = model.variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, save_path)

    val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)
    print('polyak val loss = ', val_avg_loss)
    print('polyak val rmse = ', val_rmse)

print('\nSETTINGS:\n')
for key, val in sorted(vars(ap.parse_args()).items()):
    print(key, val)

print('global seed = ', seed)

# For parsing results from file
results = vars(ap.parse_args()).copy()
results.update({'best_val_score': float(best_val_score), 'best_epoch': best_epoch})
print(json.dumps(results))

sess.close()
