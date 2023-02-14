class KNN:
    '''
    K-Nearest Neighbor Classifier
    *Instance Attribs*:
    train_set <DataFrame>: training data
    X_test <DataFrame>: testing data
    y_train <Series/DataFrame>: output variable from training data
    k <int>: number of nearest neighbors considered
    distance_type <str>: euclidean, manhattan
    '''

    def __init__(self, train_set, X_test, y_train, k=3, distance_type='euclidean', standardize: bool = True):
        self.train_set = train_set
        self.X_test = X_test
        self.y_train = y_train
        self.k = k
        self.distance_type = distance_type
        self.standardize = standardize

    def euclidean(self, X_test):
        '''Euclidean distance function'''
        return lambda train_set: ((train_set['x_1'] - X_test['x_1'])**2 + (train_set['x_2'] - X_test['x_2'])**2)**0.5

    def manhattan(self, X_test):
        '''Manhattan distance function'''
        return lambda train_set: abs(train_set - X_test)

    def standardizer(self, train_set, X_test):
        '''Standardize data
        *Parameters*: train_set, X_test
        *Return*: std_train, std_test
        '''
        import numpy as np
        # cara menghitung:
        # (data - rata2)/stdev
        input_data = train_set[['x_1', 'x_2']]
        mean = np.mean(input_data, axis=0)
        stdev = np.std(input_data, axis=0)

        std_train = input_data.apply(lambda x: (x-mean)/stdev, axis=1)
        std_test = X_test.apply(lambda x: (x-mean)/stdev, axis=1)

        return std_train, std_test

    def calculate_distance(self, X_test):
        '''
        Calculate distance test data to all data points in train_set
        *Parameter*:
        X_test <DataFrame>
        *Return*: lambda function to be-.apply()-ed to train_set DataFrame
        '''
        return getattr(self, self.distance_type)(X_test)

    def majority_vote(self, train_set, X_test, y_train, k):
        '''
        Determine majority class of neighbor data points
        *Parameters*:
        train_set <DataFrame>: training data
        X_test <DataFrame>: testing data
        y_train <Series/DataFrame>: output variable from training data
        k <int>: number of nearest neighbors considered
        *Return*: list of nearest neighbors' index in DataFrame <list>
        '''
        idx, _ = self.neighbor_index(train_set, X_test, y_train, k)
        return idx.value_counts(normalize=True).index[0]

    def neighbor_index(self, train_set, X_test, y_train, k):
        '''Calculate nearest neighbor classes to the testing set and the index of it
        train_set <DataFrame>: training data
        X_test <DataFrame>: testing data
        y_train <Series/DataFrame>: output variable from training data
        k <int>: number of nearest neighbors considered
        *Return*:
        nearest_y: classes of neighbors <str(?)>
        idx_nn: list of index
        '''
        import numpy as np
        if self.standardize:
            train_set, X_test = self.standardizer(train_set, X_test)

        f_dist = self.calculate_distance(X_test)
        distances = train_set.apply(f_dist, axis=1)

        # cari k tetangga terdekat
        idx_nn = list(np.argsort(distances[0]))[:k]
        nearest_y = y_train.loc[idx_nn]
        return nearest_y, idx_nn

    def calculate_knn(self):
        '''
        Calculate majority vote and neighbor index
        *Return*:
        mv: majority vote <int>
        ni: indexes <list>
        '''
        mv = self.majority_vote(
            self.train_set, self.X_test, self.y_train, self.k)
        _, ni = self.neighbor_index(
            self.train_set, self.X_test, self.y_train, self.k)
        return mv, ni
