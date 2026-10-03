from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Impurity(ABC):
    @abstractmethod
    def base_impurity(self, y:np.ndarray)->float:
        """Unconditional impurity of a label array.

        Args:
            y: 1D array of class labels.

        Returns:
            Impurity value, 0.0 if `y` is empty or single-class.
        """
        pass


    def impurity_gain(self, column: np.ndarray, y: np.ndarray, base_impurity: float | None = None) -> float:
        """Impurity gain from splitting `column` at its median.

        Args:
            column: 1D array of feature values, aligned with `y`.
            y: 1D array of labels.
            base_impurity: Impurity of `y`, computed if not given.

        Returns:
            Gain from the split. 0.0 if the split is degenerate
            (e.g. a constant column).
        """
        if base_impurity is None:
            base_impurity=self.base_impurity(y)

        threshold = np.median(column)
        left_mask = column<=threshold
        length_left = left_mask.sum()
        length_right = len(column)-length_left

        return base_impurity-(length_left/len(column))*self.base_impurity(y[left_mask])-(length_right/len(column))*self.base_impurity(y[~left_mask])



class Entropy(Impurity):
    def base_impurity(self, y:np.ndarray)->float:
        """Shannon entropy: -sum(p * log2(p))."""
        if len(y)==0:
            return 0
        _, counts = np.unique_counts(y)
        p = counts / np.sum(counts)
        return -np.sum(p*np.log2(p))



class Gini(Impurity):
    def base_impurity(self, y:np.ndarray)->float:
        """Gini impurity: 1 - sum(p**2)."""
        if len(y)==0:
            return 0
        _, counts = np.unique_counts(y)
        p = counts/np.sum(counts)
        return 1-np.sum(p**2)



class DecisionTree:
    def __init__(self, criterion="entropy", max_depth=None):
        """Args:
            criterion: "entropy" or "gini".
            max_depth: Max tree depth, unlimited if None.

        Raises:
            ValueError: If `criterion` is invalid.
        """
        self.root = None
        self.max_depth = max_depth
        if criterion=="entropy":
            self.impurity=Entropy()
        elif criterion=="gini":
            self.impurity=Gini()
        else:
            raise ValueError("Criterion must be either entropy or gini")


    def fit(self, x: pd.DataFrame, y: pd.Series):
        """Fits the tree, splitting each node on the feature's own median.

        Args:
            x: Feature matrix.
            y: Target labels, aligned with `x`.
        """
        X = x.to_numpy()
        y = y.to_numpy()
        new_tree = Node(X, y, self.impurity, self.max_depth).split()
        self.root = new_tree


    def predict(self, x: pd.DataFrame)->np.ndarray:
        """Predicts labels for each row of `x`.

        Args:
            x: Feature matrix with the same columns as training data.

        Returns:
            Array of predicted labels.

        Raises:
            RuntimeError: If called before `fit`.
        """
        if self.root is None:
            raise RuntimeError("Tree not fitted to data.")

        return np.array([
            self.root.predict(row)
            for row in x.to_numpy()
        ])



class Node:
    """Internal or leaf-producing node, split on a single feature's median."""
    def __init__(self, X: np.ndarray, y: np.ndarray, impurity:Impurity, max_depth=None):
        self.X = X
        self.y = y
        self.depth_remaining = max_depth
        self.impurity=impurity
        self.feature = 0
        self.threshold = 0
        self.left = None
        self.right = None


    def predict(self, row: np.ndarray):
        """Routes `row` to a leaf and returns its label."""
        if row[self.feature] <= self.threshold:
            return self.left.predict(row)
        return self.right.predict(row)

    
    def split(self):
        """Recursively splits on the best feature's median, or returns a Leaf.

        Stops and returns a Leaf if max depth is reached, labels are
        already pure, features are identical, or no feature has positive
        impurity gain.
        """
        X = self.X
        y = self.y

        #Early leaves generation checks.
        values, counts = np.unique_counts(y)
        majority_label = values[np.argmax(counts)]
        identical_label = len(values)==1
        identical_features = np.all(X == X[0], axis=0).all()
        reached_max_depth = (self.depth_remaining is not None and self.depth_remaining <= 0)

        #Create leaves if any special conditions are met.
        if reached_max_depth or identical_label or identical_features:
            return Leaf(majority_label)

        #Get best feature column.
        base_impurity = self.impurity.base_impurity(y)
        gains = [self.impurity.impurity_gain(X[:,column], y, base_impurity) for column in range(self.X.shape[1])]
        best_column = int(np.argmax(gains))
        self.feature=best_column

        #If best split means no gain (all split into one side ), must resort to Leaf to prevent infinite recursion.
        if gains[self.feature]<=0:
            return Leaf(majority_label)

        #Create mask.
        column = X[:,self.feature]
        self.threshold = np.median(column)
        mask = column <= self.threshold

        #Split.
        next_depth = None if self.depth_remaining is None else self.depth_remaining-1
        self.left=Node(X[mask], y[mask], self.impurity, next_depth).split()
        self.right=Node(X[~mask], y[~mask], self.impurity, next_depth).split()
        return self



class Leaf:
    """Terminal node holding a single predicted label."""
    def __init__(self, label):
        self.label = label

    def predict(self, row: np.ndarray):
        return self.label

    def split(self):
        raise Exception("Leaf can't be split.")