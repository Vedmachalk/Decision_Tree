import numpy as np
from collections import Counter
import pandas

def gini_criterion(threshold, feature_vector, target_vector):
    R = len(feature_vector)
    R_l = (feature_vector < threshold).sum()
    R_r = (feature_vector >= threshold).sum()
    R_l_true = ((feature_vector < threshold) & (target_vector == 0)).sum()
    R_r_true = ((feature_vector >= threshold) & (target_vector == 1)).sum()
    gini = - R_l / R * (1 - R_l_true**2 / R_l**2 - (R_l - R_l_true)**2 / R_l**2) \
            - R_r / R * (1 - R_r_true**2 / R_r**2 - (R_r - R_r_true)**2 / R_r**2)
    return gini

gini_criterion = np.vectorize(gini_criterion) # use vectorize instead loops
gini_criterion.excluded.add(1)
gini_criterion.excluded.add(2)

def find_best_split(feature_vector, target_vector):

    if isinstance(feature_vector, pandas.DataFrame):
        feature_vector = feature_vector.values
    unique_sorted = np.unique(feature_vector)
    unique_sorted.sort()
    thresholds = (unique_sorted[:-1] + unique_sorted[1:]) / 2
    ginis = gini_criterion(thresholds, feature_vector, target_vector) #here we use thresholds and distinguish them with the  gini_criterion
    best_thr = thresholds[ginis.argmax()]
    best_gini = ginis[ginis.argmax()]

    return [thresholds, ginis, best_thr, best_gini]

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {} # The main node
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        # They should be equal
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        # Transform pandas dataset in values
        if isinstance(sub_X, pandas.DataFrame):
            sub_X = sub_X.values

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            # if we have only one feature in X then :
            if np.all(sub_X[:, feature] == sub_X[0, feature]):
                continue

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count

                ratio_items = np.array(list(ratio.items()))
                order = np.argsort(ratio_items[:, 1])
                categories_map = dict(zip(ratio_items[:, 0], order))
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(feature_vector) <= 3: # Stop if we have <= 3
                continue

            # if we have feature like these we stop, cause we can not make a decision
            if feature_type == "categorical" and np.all(ratio_items[:, 1] == ratio_items[0, 1]):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if (gini_best is None or gini > gini_best):
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold
                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError
        # DEBUG
        # print(f'BEST FEATURE is {feature_best}, gini = {gini_best} at threshold_best = {threshold_best}')
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return


        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])
        # print(f'node["left_child"]: {node["left_child"]} | node["right_child"]: {node["right_child"]}')

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']
        if 'threshold' in node.keys():
            if x[node['feature_split']] < node['threshold']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        elif 'categories_split' in node.keys():
            if x[node['feature_split']] in node['categories_split']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        else:
            raise ValueError


    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
