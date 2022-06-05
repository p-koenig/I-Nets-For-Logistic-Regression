"""
Generate samples of synthetic data sets.
"""

# Authors: B. Thirion, G. Varoquaux, A. Gramfort, V. Michel, O. Grisel,
#          G. Louppe, J. Nothman
# License: BSD 3 clause

import numbers
import array
from collections.abc import Iterable

import numpy as np
from scipy import linalg
import scipy.sparse as sp

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as util_shuffle
from sklearn.utils.random import sample_without_replacement
from sklearn.preprocessing import MinMaxScaler


def get_distribution_data_from_string_sklearn(distribution_name, size, generator, n_informative, random_parameters=True, distrib_param_max=10, low=0, high=1):
    
    if random_parameters:
        value_1 = generator.uniform(0, 1)
        value_2 = generator.uniform(0, 1)   

        parameter_by_distribution = {
            'normal': {
                'loc': generator.uniform(0, distrib_param_max),
                'scale': generator.uniform(0, distrib_param_max),
            },
            'uniform': {
                'low': np.minimum(value_1, value_2),
                'high': np.maximum(value_1, value_2),
            },
            'gamma': {
                'shape': generator.uniform(0, distrib_param_max),
                'scale': generator.uniform(0, distrib_param_max),
            },        
            'exponential': {
                'scale': generator.uniform(0, distrib_param_max),
            },        
            'beta': {
                'a': generator.uniform(0, distrib_param_max),
                'b': generator.uniform(0, distrib_param_max),
            },
            'binomial': {
                'n': generator.uniform(1, 1000),
                'p': generator.uniform(0, 1),
            },
            'poisson': {
                'lam': np.random.uniform(0, distrib_param_max),
            },        

        }
    else:
        parameter_by_distribution = {
            'normal': {
                'loc': distrib_param_max//2,
                'scale': distrib_param_max//2,
            },
            'uniform': {
                'low': 0,
                'high': 1,
            },
            'gamma': {
                'shape': distrib_param_max//2,
                'scale': distrib_param_max//2,
            },        
            'exponential': {
                'scale': distrib_param_max//2,
            },        
            'beta': {
                'a': distrib_param_max//2,
                'b': distrib_param_max//2,
            },
            'binomial': {
                'n': size,
                'p': 0.5,
            },
            'poisson': {
                'lam': distrib_param_max//2,
            },        

        }           
        
    
    if distribution_name == 'normal':
        data_column = np.random.normal(parameter_by_distribution['normal']['loc'], parameter_by_distribution['normal']['scale'], size=size)
        A_column = np.random.normal(parameter_by_distribution['normal']['loc'], parameter_by_distribution['normal']['scale'], size=n_informative)
        distribution_parameter = {'normal': {'class_0': parameter_by_distribution['normal'], 'class_1': parameter_by_distribution['normal']}  }#parameter_by_distribution['normal']
    elif distribution_name == 'uniform':
        data_column = np.random.uniform(parameter_by_distribution['uniform']['low'], parameter_by_distribution['uniform']['high'], size=size)
        A_column = np.random.uniform(parameter_by_distribution['uniform']['low'], parameter_by_distribution['uniform']['high'], size=n_informative)
        distribution_parameter = {'uniform': {'class_0': parameter_by_distribution['uniform'], 'class_1': parameter_by_distribution['uniform']}  }#parameter_by_distribution['uniform']
    elif distribution_name == 'gamma':
        data_column = np.random.gamma(parameter_by_distribution['gamma']['shape'], parameter_by_distribution['gamma']['scale'], size=size)
        A_column = np.random.gamma(parameter_by_distribution['gamma']['shape'], parameter_by_distribution['gamma']['scale'], size=n_informative)
        distribution_parameter = {'gamma': {'class_0': parameter_by_distribution['gamma'], 'class_1': parameter_by_distribution['gamma']}  }#parameter_by_distribution['gamma']
    elif distribution_name == 'exponential':
        data_column = np.random.exponential(parameter_by_distribution['exponential']['scale'], size=size)
        A_column = np.random.exponential(parameter_by_distribution['exponential']['scale'], size=n_informative)
        distribution_parameter = {'exponential': {'class_0': parameter_by_distribution['exponential'], 'class_1': parameter_by_distribution['exponential']}  }#parameter_by_distribution['exponential']        
    elif distribution_name == 'beta':
        data_column = np.random.beta(parameter_by_distribution['beta']['a'], parameter_by_distribution['beta']['b'], size=size)
        A_column = np.random.beta(parameter_by_distribution['beta']['a'], parameter_by_distribution['beta']['b'], size=n_informative)
        distribution_parameter = {'beta': {'class_0': parameter_by_distribution['beta'], 'class_1': parameter_by_distribution['beta']}  }#parameter_by_distribution['beta']
    elif distribution_name == 'binomial':
        data_column = np.random.binomial(parameter_by_distribution['binomial']['n'], parameter_by_distribution['binomial']['p'], size=size)
        A_column = np.random.binomial(parameter_by_distribution['binomial']['n'], parameter_by_distribution['binomial']['p'], size=n_informative)
        distribution_parameter = {'binomial': {'class_0': parameter_by_distribution['binomial'], 'class_1': parameter_by_distribution['binomial']}  }#parameter_by_distribution['binomial']        
    elif distribution_name == 'poisson':
        data_column = np.random.poisson(parameter_by_distribution['poisson']['lam'], size=size)
        A_column = np.random.poisson(parameter_by_distribution['poisson']['lam'], size=n_informative)
        distribution_parameter = {'poisson': {'class_0': parameter_by_distribution['poisson'], 'class_1': parameter_by_distribution['poisson']}  }#parameter_by_distribution['poisson']        
    else:
        return None, None
    
    #print('NO NORMALIZE', data_column)
    scaler = MinMaxScaler(feature_range=(low, high))
    scaler.fit(data_column.reshape(-1, 1))
    data_column = scaler.transform(data_column.reshape(-1, 1)).ravel()
    #print('NORMALIZE', data_column)
    
    return data_column, A_column, distribution_parameter#, scaler

def _generate_hypercube(samples, dimensions, rng):
    """Returns distinct binary samples of length dimensions."""
    if dimensions > 30:
        return np.hstack(
            [
                rng.randint(2, size=(samples, dimensions - 30)),
                _generate_hypercube(samples, 30, rng),
            ]
        )
    out = sample_without_replacement(2 ** dimensions, samples, random_state=rng).astype(
        dtype=">u4", copy=False
    )
    out = np.unpackbits(out.view(">u1")).reshape((-1, 32))[:, -dimensions:]
    return out


def make_classification_distribution(
    n_samples=100,
    n_features=20,
    *,
    n_informative=2,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    distribution_list = ['uniform', 'normal', 'gamma', 'exponential', 'beta', 'binomial', 'poisson'], #['uniform'],#['normal'],#
    weights=None,
    flip_y=0.01,
    class_sep=1.0,
    hypercube=True,
    shift=0.0,
    scale=1.0,
    shuffle=True,
    random_state=None,
    random_parameters=True,
    distrib_param_max=10,
):
    """Generate a random n-class classification problem.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an ``n_informative``-dimensional hypercube with sides of
    length ``2*class_sep`` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Without shuffling, ``X`` horizontally stacks features in the following
    order: the primary ``n_informative`` features, followed by ``n_redundant``
    linear combinations of the informative features, followed by ``n_repeated``
    duplicates, drawn randomly with replacement from the informative and
    redundant features. The remaining features are filled with random noise.
    Thus, without shuffling, all useful features are contained in the columns
    ``X[:, :n_informative + n_redundant + n_repeated]``.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=20
        The total number of features. These comprise ``n_informative``
        informative features, ``n_redundant`` redundant features,
        ``n_repeated`` duplicated features and
        ``n_features-n_informative-n_redundant-n_repeated`` useless features
        drawn at random.

    n_informative : int, default=2
        The number of informative features. Each class is composed of a number
        of gaussian clusters each located around the vertices of a hypercube
        in a subspace of dimension ``n_informative``. For each cluster,
        informative features are drawn independently from  N(0, 1) and then
        randomly linearly combined within each cluster in order to add
        covariance. The clusters are then placed on the vertices of the
        hypercube.

    n_redundant : int, default=2
        The number of redundant features. These features are generated as
        random linear combinations of the informative features.

    n_repeated : int, default=0
        The number of duplicated features, drawn randomly from the informative
        and the redundant features.

    n_classes : int, default=2
        The number of classes (or labels) of the classification problem.

    n_clusters_per_class : int, default=2
        The number of clusters per class.

    weights : array-like of shape (n_classes,) or (n_classes - 1,),\
              default=None
        The proportions of samples assigned to each class. If None, then
        classes are balanced. Note that if ``len(weights) == n_classes - 1``,
        then the last class weight is automatically inferred.
        More than ``n_samples`` samples may be returned if the sum of
        ``weights`` exceeds 1. Note that the actual class proportions will
        not exactly match ``weights`` when ``flip_y`` isn't 0.

    flip_y : float, default=0.01
        The fraction of samples whose class is assigned randomly. Larger
        values introduce noise in the labels and make the classification
        task harder. Note that the default setting flip_y > 0 might lead
        to less than ``n_classes`` in y in some cases.

    class_sep : float, default=1.0
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.

    hypercube : bool, default=True
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.

    shift : float, ndarray of shape (n_features,) or None, default=0.0
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].

    scale : float, ndarray of shape (n_features,) or None, default=1.0
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.

    shuffle : bool, default=True
        Shuffle the samples and the features.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels for class membership of each sample.

    Notes
    -----
    The algorithm is adapted from Guyon [1] and was designed to generate
    the "Madelon" dataset.

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.

    See Also
    --------
    make_blobs : Simplified variant.
    make_multilabel_classification : Unrelated generator for multilabel tasks.
    """
    generator = check_random_state(random_state)

    # Count features, clusters and samples
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError(
            "Number of informative, redundant and repeated "
            "features must sum to less than the number of total"
            " features"
        )
    # Use log2 to avoid overflow errors
    if n_informative < np.log2(n_classes * n_clusters_per_class):
        msg = "n_classes({}) * n_clusters_per_class({}) must be"
        msg += " smaller or equal 2**n_informative({})={}"
        raise ValueError(
            msg.format(
                n_classes, n_clusters_per_class, n_informative, 2 ** n_informative
            )
        )

    if weights is not None:
        if len(weights) not in [n_classes, n_classes - 1]:
            raise ValueError(
                "Weights specified but incompatible with number of classes."
            )
        if len(weights) == n_classes - 1:
            if isinstance(weights, list):
                weights = weights + [1.0 - sum(weights)]
            else:
                weights = np.resize(weights, n_classes)
                weights[-1] = 1.0 - sum(weights[:-1])
    else:
        weights = [1.0 / n_classes] * n_classes

    n_useless = n_features - n_informative - n_redundant - n_repeated
    n_clusters = n_classes * n_clusters_per_class

    # Distribute samples among clusters by weight
    n_samples_per_cluster = [
        int(n_samples * weights[k % n_classes] / n_clusters_per_class)
        for k in range(n_clusters)
    ]

    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1

    # Initialize X and y
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    # Build the polytope whose vertices become cluster centroids
    centroids = _generate_hypercube(n_clusters, n_informative, generator).astype(
        float, copy=False
    )
    centroids *= 2 * class_sep
    centroids -= class_sep
    if not hypercube:
        centroids *= generator.rand(n_clusters, 1)
        centroids *= generator.rand(1, n_informative)
    
    distribution_parameter_list = [None]*n_features
    #scaler_list = [None]*n_features

    if False:
        # Initially draw informative features from the standard normal
        X[:, :n_informative] = generator.randn(n_samples, n_informative)
        distribution_parameter_list = None
        #scaler_list = None
    else:
        A_rand = np.zeros(shape=(n_informative, n_informative))
        for i in range(n_features):
            if i < n_informative:
                distribution = generator.choice(distribution_list)
                #print(distribution)
                #X[:, i], distribution_parameter_list[i], scaler_list[i] = get_distribution_data_from_string_sklearn(distribution, n_samples, generator)
                X[:, i], A_rand[:, i], distribution_parameter_list[i] = get_distribution_data_from_string_sklearn(distribution, n_samples, generator, n_informative, random_parameters=random_parameters, distrib_param_max=distrib_param_max)
            else:
                distribution = generator.choice(distribution_list)
                #print(distribution)
                #X[:, i], distribution_parameter_list[i], scaler_list[i] = get_distribution_data_from_string_sklearn(distribution, n_samples, generator)
                _, _, distribution_parameter_list[i] = get_distribution_data_from_string_sklearn(distribution, n_samples, generator, n_informative, random_parameters=random_parameters, distrib_param_max=distrib_param_max)                
            
    # Create each cluster; a variant of make_blobs
    stop = 0
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + n_samples_per_cluster[k]
        y[start:stop] = k % n_classes  # assign labels
        X_k = X[start:stop, :n_informative]  # slice a view of the cluster
        if False:
            A = 2 * generator.rand(n_informative, n_informative) - 1
            X_k[...] = np.dot(X_k, A)  # introduce random covariance
        elif False:
            A = 2 * A_rand - 1
            X_k[...] = np.dot(X_k, A)  # introduce random covariance
            
        X_k += centroid  # shift the cluster to a vertex

    # Create redundant features
    if n_redundant > 0:
        B = 2 * generator.rand(n_informative, n_redundant) - 1
        X[:, n_informative : n_informative + n_redundant] = np.dot(
            X[:, :n_informative], B
        )

    # Repeat some features
    if n_repeated > 0:
        n = n_informative + n_redundant
        indices = ((n - 1) * generator.rand(n_repeated) + 0.5).astype(np.intp)
        X[:, n : n + n_repeated] = X[:, indices]

    # Fill useless features
    if n_useless > 0:
        if False:
            X[:, -n_useless:] = generator.randn(n_samples, n_useless)
        else:
            for i in range(n_features):
                if i >= n_informative + n_redundant + n_repeated:
                    distribution = generator.choice(distribution_list)
                    #print(distribution)
                    #X[:, i], distribution_parameter_list[i], scaler_list[i] = get_distribution_data_from_string_sklearn(distribution, n_samples, generator)
                    X[:, i], A_rand[:, i], distribution_parameter_list[i] = get_distribution_data_from_string_sklearn(distribution, n_samples, generator, n_informative, random_parameters=random_parameters, distrib_param_max=distrib_param_max)           
                else:
                    continue
                

    # Randomly replace labels
    if flip_y >= 0.0:
        flip_mask = generator.rand(n_samples) < flip_y
        y[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())

    # Randomly shift and scale
    if shift is None:
        shift = (2 * generator.rand(n_features) - 1) * class_sep
    X += shift

    if scale is None:
        scale = 1 + 100 * generator.rand(n_features)
    X *= scale

    if shuffle:
        # Randomly permute samples
        X, y = util_shuffle(X, y, random_state=generator)

        # Randomly permute features
        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]
        
    X_scaled = []
    for i, column in enumerate(X.T):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(column.reshape(-1, 1))
        #list_of_data_points[:,i] = scaler.transform(column.reshape(-1, 1)).ravel()
        X_scaled.append(scaler.transform(column.reshape(-1, 1)).ravel())
    X = np.array(X_scaled).T    

    return X, y, distribution_parameter_list#, scaler_list


def make_multilabel_classification(
    n_samples=100,
    n_features=20,
    *,
    n_classes=5,
    n_labels=2,
    length=50,
    allow_unlabeled=True,
    sparse=False,
    return_indicator="dense",
    return_distributions=False,
    random_state=None,
):
    """Generate a random multilabel classification problem.

    For each sample, the generative process is:
        - pick the number of labels: n ~ Poisson(n_labels)
        - n times, choose a class c: c ~ Multinomial(theta)
        - pick the document length: k ~ Poisson(length)
        - k times, choose a word: w ~ Multinomial(theta_c)

    In the above process, rejection sampling is used to make sure that
    n is never zero or more than `n_classes`, and that the document length
    is never zero. Likewise, we reject classes which have already been chosen.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=20
        The total number of features.

    n_classes : int, default=5
        The number of classes of the classification problem.

    n_labels : int, default=2
        The average number of labels per instance. More precisely, the number
        of labels per sample is drawn from a Poisson distribution with
        ``n_labels`` as its expected value, but samples are bounded (using
        rejection sampling) by ``n_classes``, and must be nonzero if
        ``allow_unlabeled`` is False.

    length : int, default=50
        The sum of the features (number of words if documents) is drawn from
        a Poisson distribution with this expected value.

    allow_unlabeled : bool, default=True
        If ``True``, some instances might not belong to any class.

    sparse : bool, default=False
        If ``True``, return a sparse feature matrix

        .. versionadded:: 0.17
           parameter to allow *sparse* output.

    return_indicator : {'dense', 'sparse'} or False, default='dense'
        If ``'dense'`` return ``Y`` in the dense binary indicator format. If
        ``'sparse'`` return ``Y`` in the sparse binary indicator format.
        ``False`` returns a list of lists of labels.

    return_distributions : bool, default=False
        If ``True``, return the prior class probability and conditional
        probabilities of features given classes, from which the data was
        drawn.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
        The label sets. Sparse matrix should be of CSR format.

    p_c : ndarray of shape (n_classes,)
        The probability of each class being drawn. Only returned if
        ``return_distributions=True``.

    p_w_c : ndarray of shape (n_features, n_classes)
        The probability of each feature being drawn given each class.
        Only returned if ``return_distributions=True``.

    """
    if n_classes < 1:
        raise ValueError(
            "'n_classes' should be an integer greater than 0. Got {} instead.".format(
                n_classes
            )
        )
    if length < 1:
        raise ValueError(
            "'length' should be an integer greater than 0. Got {} instead.".format(
                length
            )
        )

    generator = check_random_state(random_state)
    p_c = generator.rand(n_classes)
    p_c /= p_c.sum()
    cumulative_p_c = np.cumsum(p_c)
    p_w_c = generator.rand(n_features, n_classes)
    p_w_c /= np.sum(p_w_c, axis=0)

    def sample_example():
        _, n_classes = p_w_c.shape

        # pick a nonzero number of labels per document by rejection sampling
        y_size = n_classes + 1
        while (not allow_unlabeled and y_size == 0) or y_size > n_classes:
            y_size = generator.poisson(n_labels)

        # pick n classes
        y = set()
        while len(y) != y_size:
            # pick a class with probability P(c)
            c = np.searchsorted(cumulative_p_c, generator.rand(y_size - len(y)))
            y.update(c)
        y = list(y)

        # pick a non-zero document length by rejection sampling
        n_words = 0
        while n_words == 0:
            n_words = generator.poisson(length)

        # generate a document of length n_words
        if len(y) == 0:
            # if sample does not belong to any class, generate noise word
            words = generator.randint(n_features, size=n_words)
            return words, y

        # sample words with replacement from selected classes
        cumulative_p_w_sample = p_w_c.take(y, axis=1).sum(axis=1).cumsum()
        cumulative_p_w_sample /= cumulative_p_w_sample[-1]
        words = np.searchsorted(cumulative_p_w_sample, generator.rand(n_words))
        return words, y

    X_indices = array.array("i")
    X_indptr = array.array("i", [0])
    Y = []
    for i in range(n_samples):
        words, y = sample_example()
        X_indices.extend(words)
        X_indptr.append(len(X_indices))
        Y.append(y)
    X_data = np.ones(len(X_indices), dtype=np.float64)
    X = sp.csr_matrix((X_data, X_indices, X_indptr), shape=(n_samples, n_features))
    X.sum_duplicates()
    if not sparse:
        X = X.toarray()

    # return_indicator can be True due to backward compatibility
    if return_indicator in (True, "sparse", "dense"):
        lb = MultiLabelBinarizer(sparse_output=(return_indicator == "sparse"))
        Y = lb.fit([range(n_classes)]).transform(Y)
    elif return_indicator is not False:
        raise ValueError("return_indicator must be either 'sparse', 'dense' or False.")
    if return_distributions:
        return X, Y, p_c, p_w_c
    return X, Y


def make_sparse_uncorrelated(n_samples=100, n_features=10, *, random_state=None):
    """Generate a random regression problem with sparse uncorrelated design.

    This dataset is described in Celeux et al [1]. as::

        X ~ N(0, 1)
        y(X) = X[:, 0] + 2 * X[:, 1] - 2 * X[:, 2] - 1.5 * X[:, 3]

    Only the first 4 features are informative. The remaining features are
    useless.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=10
        The number of features.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The input samples.

    y : ndarray of shape (n_samples,)
        The output values.

    References
    ----------
    .. [1] G. Celeux, M. El Anbari, J.-M. Marin, C. P. Robert,
           "Regularization in regression: comparing Bayesian and frequentist
           methods in a poorly informative situation", 2009.
    """
    generator = check_random_state(random_state)

    X = generator.normal(loc=0, scale=1, size=(n_samples, n_features))
    y = generator.normal(
        loc=(X[:, 0] + 2 * X[:, 1] - 2 * X[:, 2] - 1.5 * X[:, 3]),
        scale=np.ones(n_samples),
    )

    return X, y


def _shuffle(data, random_state=None):
    generator = check_random_state(random_state)
    n_rows, n_cols = data.shape
    row_idx = generator.permutation(n_rows)
    col_idx = generator.permutation(n_cols)
    result = data[row_idx][:, col_idx]
    return result, row_idx, col_idx
