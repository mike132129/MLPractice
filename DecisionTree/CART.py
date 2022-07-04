import numpy as np


class Node:

    def __init__(self, gini, num_samples, num_samples_per_class,
                 predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:

    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _grow_tree(self, X, y, depth=0):

        num_samples_per_class = [
            np.sum(y == i) for i in range(self.n_classes_)
        ]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y):
        '''
        Find the best split for a node.
        Return: 
            best_idx: index of feature to be split, None if no split is found
            best_thr: threshold to use for the split, None if no split is found
        '''
        m = y.size
        if m <= 1:
            return None, None

        num_classes = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini
        best_gini = 1.0 - sum((n / m)**2 for n in num_classes)
        best_idx, best_thr = None, None

        # Loop all features
        for idx in range(self.n_features_):
            # decision stump
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0] * self.n_classes_
            num_right = num_classes.copy()

            # Loop all possible split position
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1 - sum(
                    (num_left[x] / i)**2 for x in range(self.n_classes_))
                gini_right = 1 - sum((num_right[x] / (m - i))**2
                                     for x in range(self.n_classes_))

                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _gini(self, y):
        m = y.size
        num_class = [np.sum(y == c) for c in range(self.n_classes_)]
        return 1.0 - sum((n / m)**2 for n in num_class)

    def _predict(self, x):
        node = self.tree_
        while node.left:
            if x[node.feature_index] < node.threshold:
                node = node.left

            else:
                node = node.right
        return node.predicted_class


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('wifi_localization.txt', delimiter='\t')
    data = df.to_numpy()
    X, y = data[:, :-1], data[:, -1] - 1
    DT = DecisionTreeClassifier()
    DT.fit(X, y)
    pred = DT.predict([[-70, 0, 0, 0, -40, 0, 0]])
    breakpoint()
