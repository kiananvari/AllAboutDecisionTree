import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


def gini_index(data: pd.DataFrame, feature, label, classes, threshold):
    total_gini = 0
    samples = len(data)

    sub_tree_labels_high = data[data[feature] >= threshold][label].value_counts()
    sub_tree_samples_high = len(data[data[feature] >= threshold])
    sum_gini_high = 0
    for c in classes:
        pi = sub_tree_labels_high.get(c, 0) / sub_tree_samples_high
        sum_gini_high += pi ** 2
    sum_gini_high = 1 - sum_gini_high
    sum_gini_high = sum_gini_high * sub_tree_samples_high
    total_gini += sum_gini_high / samples

    sub_tree_labels_low = data[data[feature] < threshold][label].value_counts()
    sub_tree_samples_low = len(data[data[feature] < threshold])
    sum_gini_low = 0
    for c in classes:
        pi = sub_tree_labels_low.get(c, 0) / sub_tree_samples_low
        sum_gini_low += pi ** 2
    sum_gini_low = 1 - sum_gini_low
    sum_gini_low = sum_gini_low * sub_tree_samples_low
    total_gini += sum_gini_low / samples

    return total_gini


def entropy(data: pd.DataFrame, feature, label, classes, threshold):
    total_entropy = 0
    samples = len(data)

    sub_tree_labels_high = data[data[feature] >= threshold][label].value_counts()
    sub_tree_samples_high = len(data[data[feature] >= threshold])
    sum_entropy_high = 0
    for c in classes:
        pi = sub_tree_labels_high.get(c, 0) / sub_tree_samples_high
        if pi != 0:
            sum_entropy_high -= pi * np.log(pi)
        # else: pi * np.log(pi) is 0 and there is no need to compute
    sum_entropy_high = sum_entropy_high * sub_tree_samples_high
    total_entropy += sum_entropy_high / samples

    sub_tree_labels_low = data[data[feature] < threshold][label].value_counts()
    sub_tree_samples_low = len(data[data[feature] < threshold])
    sum_entropy_low = 0
    for c in classes:
        pi = sub_tree_labels_low.get(c, 0) / sub_tree_samples_high
        if pi != 0:
            sum_entropy_low -= pi * np.log(pi)
        # else: pi * np.log(pi) is 0 and there is no need to compute
    sum_entropy_low = sum_entropy_low * sub_tree_samples_low
    total_entropy += sum_entropy_low / samples

    return total_entropy


def calculate_best_threshold(data: pd.DataFrame, feature, label, classes, purity_function):
    values = sorted(data[feature].unique())
    thresholds = [(values[i + 1] + values[i]) / 2 for i in range(len(values) - 1)]

    best_purity = 1
    best_th = 0
    for threshold in thresholds:
        purity = purity_function(data, feature, label, classes, threshold)
        if purity < best_purity:
            best_purity = purity
            best_th = threshold
    return best_purity, best_th


def calculate_best_split(data: pd.DataFrame, label, classes, purity_function):
    available_features = set(data.columns) - set([label])
    assert len(available_features) > 0

    best_purity = 1
    best_feature = None
    best_th = None
    for f in available_features:
        purity, th = calculate_best_threshold(data, f, label, classes, purity_function)
        if purity < best_purity:
            best_purity = purity
            best_feature = f
            best_th = th

    return best_feature, best_th


def split_data(data: pd.DataFrame, feature, threshold):
    sub_datas = {f">= {threshold}": data[data[feature] >= threshold],
                 f"< {threshold}": data[data[feature] < threshold]}

    return sub_datas


def make_decision_tree(data: pd.DataFrame,
                       label='label',
                       classes=None,
                       return_prob=True,
                       purity_function=gini_index,
                       min_sample_split=0,
                       max_depth=np.inf):
    available_features = set(data.columns) - set([label])
    if len(available_features) == 0 or \
            len(data[label].unique()) == 1 or \
            len(data) < min_sample_split or \
            max_depth == 0:
        if return_prob:
            return json.loads((data[f'{label}'].value_counts() / len(data)).to_json())
        else:
            return {f'{label}': data[label].value_counts().idxmax()}

    split_feature, split_th = calculate_best_split(data, label, classes, purity_function)
    sub_datas = split_data(data, split_feature, split_th)

    if len(sub_datas.keys()) == 1:
        if return_prob:
            return json.loads((data[f'{label}'].value_counts() / len(data)).to_json())
        else:
            return {f'{label}': data[label].value_counts().idxmax()}

    results = (split_feature, {})
    for key in sub_datas.keys():
        results[1][key] = make_decision_tree(sub_datas[key],
                                             label,
                                             classes,
                                             return_prob=return_prob,
                                             purity_function=purity_function,
                                             min_sample_split=min_sample_split,
                                             max_depth=max_depth - 1
                                             )

    return results


def explain_tree(tree_json, pre_sentence=''):
    if type(tree_json) == dict:
        if 'label' in tree_json.keys():
            pre_sentence += f' then, prediction is {tree_json["label"]}'
        else:
            pre_sentence += ' then, prediction is '
            for value in tree_json.keys():
                pre_sentence += f'{value} or '
            pre_sentence = pre_sentence[:-4]
        print(pre_sentence)
        return

    feature, splits = tree_json
    if len(pre_sentence) != 0:
        pre_sentence = pre_sentence + ' and '

    sentence = pre_sentence + f'if {feature} is '

    for key in splits.keys():
        explain_tree(splits[key], sentence + key)


def predict_tree(tree_json, label, data):
    if type(tree_json) == dict:
        if label in tree_json.keys():
            return pd.DataFrame(tree_json[label], index=data.index, columns=['predict'])
        else:
            return None

    feature, splits = tree_json
    # split.keys()[0] = '>= threshold'
    threshold = float(list(splits.keys())[0].split(' ')[1])
    sub_datas = split_data(data, feature, threshold)

    predictions = [predict_tree(splits[key], label, sub_datas[key]) for key in splits.keys()]
    predictions = predictions[0].append(predictions[1])
    predictions = predictions.sort_index()

    return predictions


def score_tree(tree_json, label, data):
    classes = df[label].unique()
    h = predict_tree(tree_json, label, data)
    report = classification_report(data[label], h, output_dict=True)
    report['weighted avg']['accuracy'] = report['accuracy']
    report['macro avg']['accuracy'] = sum([accuracy_score(data[label] == cls, h == cls) for cls in classes]) / len(classes)
    report = {'macro avg': report['macro avg'], 'micro avg': report['weighted avg']}
    return report


def post_pruning(tree_json,
                 label,
                 valid_df):
    if type(tree_json) == dict or len(valid_df) == 0:
        return tree_json

    tree_json = deepcopy(tree_json)
    base_score = score_tree(tree_json, label, valid_df)['micro avg']['f1-score']

    pruned_subtree = {f'{label}': valid_df[label].value_counts().idxmax()}
    pruned_score = score_tree(pruned_subtree, label, valid_df)['micro avg']['f1-score']

    if pruned_score > base_score:
        print('pruned...')
        return pruned_subtree
    else:
        feature, splits = tree_json
        threshold = float(list(splits.keys())[0].split(' ')[1])
        sub_datas = split_data(valid_df, feature, threshold)

        # tree_json[1] = branches
        for key in tree_json[1].keys():
            tree_json[1][key] = post_pruning(tree_json[1][key], label, sub_datas[key])

    return tree_json


def evaluate_model(data,
                   purity_function=gini_index,
                   min_sample_split=0,
                   max_depth=np.inf,
                   num_runs=10,
                   postpruning=False):
    classes = data['target'].unique()
    total_report = {'macro avg': {'precision': {'train': 0, 'valid': 0},
                                  'recall': {'train': 0, 'valid': 0},
                                  'f1-score': {'train': 0, 'valid': 0},
                                  'accuracy': {'train': 0, 'valid': 0}},
                    'micro avg': {'precision': {'train': 0, 'valid': 0},
                                  'recall': {'train': 0, 'valid': 0},
                                  'f1-score': {'train': 0, 'valid': 0},
                                  'accuracy': {'train': 0, 'valid': 0}}}
    for i in range(num_runs):
        # train test split without sklearn or converting to numpy
        msk = np.random.rand(len(data)) < 0.7
        train_df = data[msk]
        valid_df = data[~msk]

        tree = make_decision_tree(train_df, 'target', classes, return_prob=False,
                                  purity_function=purity_function,
                                  min_sample_split=min_sample_split,
                                  max_depth=max_depth)
        if postpruning:
            tree = post_pruning(tree, 'target', valid_df)

        report_train = score_tree(tree, 'target', train_df)
        report_valid = score_tree(tree, 'target', valid_df)
        for avg_type in total_report.keys():
            for metric in total_report[avg_type].keys():
                total_report[avg_type][metric]['train'] += report_train[avg_type][metric] / num_runs
                total_report[avg_type][metric]['valid'] += report_valid[avg_type][metric] / num_runs

    total_report['model config'] = {'purity_function': purity_function.__name__,
                                    'min_sample_split': min_sample_split,
                                    'max_depth': max_depth,
                                    'post-pruning': postpruning}
    return total_report


class Experiments:
    def __init__(self, data, num_runs=10):
        msk = np.random.rand(len(data)) < 0.9
        self.train_df = data[msk]
        self.test_df = data[~msk]

    def purity_function_experiment(self):

        report_gini = evaluate_model(self.train_df, purity_function=gini_index)
        report_entropy = evaluate_model(self.train_df, purity_function=entropy)

        print(report_gini)
        print(report_entropy)
        print('----------------------------------')

        plt.subplot(2, 4, 1)
        plt.title('Macro Accuracy')
        plt.bar(1, report_gini['macro avg']['accuracy']['valid'])
        plt.bar(2, report_entropy['macro avg']['accuracy']['valid'])
        plt.bar(4, report_gini['macro avg']['accuracy']['train'])
        plt.bar(5, report_entropy['macro avg']['accuracy']['train'])
        plt.xticks([1, 2, 4, 5], ['gini-v', 'entropy-v', 'gini-t', 'entropy-t'])
        plt.ylim(0.87, 1.03)

        plt.subplot(2, 4, 2)
        plt.title('Macro Precision')
        plt.bar(1, report_gini['macro avg']['precision']['valid'])
        plt.bar(2, report_entropy['macro avg']['precision']['valid'])
        plt.bar(4, report_gini['macro avg']['precision']['train'])
        plt.bar(5, report_entropy['macro avg']['precision']['train'])
        plt.xticks([1, 2, 4, 5], ['gini-v', 'entropy-v', 'gini-t', 'entropy-t'])
        plt.ylim(0.87, 1.03)

        plt.subplot(2, 4, 3)
        plt.title('Macro Recall')
        plt.bar(1, report_gini['macro avg']['recall']['valid'])
        plt.bar(2, report_entropy['macro avg']['recall']['valid'])
        plt.bar(4, report_gini['macro avg']['recall']['train'])
        plt.bar(5, report_entropy['macro avg']['recall']['train'])
        plt.xticks([1, 2, 4, 5], ['gini-v', 'entropy-v', 'gini-t', 'entropy-t'])
        plt.ylim(0.87, 1.03)

        plt.subplot(2, 4, 4)
        plt.title('Macro F1-score')
        plt.bar(1, report_gini['macro avg']['f1-score']['valid'])
        plt.bar(2, report_entropy['macro avg']['f1-score']['valid'])
        plt.bar(4, report_gini['macro avg']['f1-score']['train'])
        plt.bar(5, report_entropy['macro avg']['f1-score']['train'])
        plt.xticks([1, 2, 4, 5], ['gini-v', 'entropy-v', 'gini-t', 'entropy-t'])
        plt.ylim(0.87, 1.03)

        plt.subplot(2, 4, 5)
        plt.title('Micro Accuracy')
        plt.bar(1, report_gini['micro avg']['accuracy']['valid'])
        plt.bar(2, report_entropy['micro avg']['accuracy']['valid'])
        plt.bar(4, report_gini['micro avg']['accuracy']['train'])
        plt.bar(5, report_entropy['micro avg']['accuracy']['train'])
        plt.xticks([1, 2, 4, 5], ['gini-v', 'entropy-v', 'gini-t', 'entropy-t'])
        plt.ylim(0.87, 1.03)

        plt.subplot(2, 4, 6)
        plt.title('Micro Precision')
        plt.bar(1, report_gini['micro avg']['precision']['valid'])
        plt.bar(2, report_entropy['micro avg']['precision']['valid'])
        plt.bar(4, report_gini['micro avg']['precision']['train'])
        plt.bar(5, report_entropy['micro avg']['precision']['train'])
        plt.xticks([1, 2, 4, 5], ['gini-v', 'entropy-v', 'gini-t', 'entropy-t'])
        plt.ylim(0.87, 1.03)

        plt.subplot(2, 4, 7)
        plt.title('Micro Recall')
        plt.bar(1, report_gini['micro avg']['recall']['valid'])
        plt.bar(2, report_entropy['micro avg']['recall']['valid'])
        plt.bar(4, report_gini['micro avg']['recall']['train'])
        plt.bar(5, report_entropy['micro avg']['recall']['train'])
        plt.xticks([1, 2, 4, 5], ['gini-v', 'entropy-v', 'gini-t', 'entropy-t'])
        plt.ylim(0.87, 1.03)

        plt.subplot(2, 4, 8)
        plt.title('Micro F1-score')
        plt.bar(1, report_gini['micro avg']['f1-score']['valid'])
        plt.bar(2, report_entropy['micro avg']['f1-score']['valid'])
        plt.bar(4, report_gini['micro avg']['f1-score']['train'])
        plt.bar(5, report_entropy['micro avg']['f1-score']['train'])
        plt.xticks([1, 2, 4, 5], ['gini-v', 'entropy-v', 'gini-t', 'entropy-t'])
        plt.ylim(0.87, 1.03)

        plt.show()

        if report_gini['micro avg']['f1-score']['valid'] > report_entropy['macro avg']['f1-score']['valid']:
            return gini_index
        else:
            return entropy

    def max_depth_experiment(self, purity_function):
        reports_train = {'macro avg': {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
                         'micro avg': {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}}
        reports_valid = {'macro avg': {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
                         'micro avg': {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}}

        explore_range = [0, 21, 2]
        for max_depth in range(*explore_range):
            report = evaluate_model(self.train_df,
                                    purity_function=purity_function,
                                    max_depth=max_depth)
            print(report)


            for avg_type in reports_valid.keys():
                for metric in reports_valid[avg_type].keys():
                    reports_valid[avg_type][metric].append(report[avg_type][metric]['valid'])
                    reports_train[avg_type][metric].append(report[avg_type][metric]['train'])
        print('----------------------------------')

        plt.subplot(2, 4, 1)
        plt.title('Macro Accuracy')
        plt.plot(range(*explore_range), reports_valid['macro avg']['accuracy'], label='valid')
        plt.plot(range(*explore_range), reports_train['macro avg']['accuracy'], label='train')
        plt.legend()
        plt.grid()

        plt.subplot(2, 4, 2)
        plt.title('Macro Precision')
        plt.plot(range(*explore_range), reports_valid['macro avg']['precision'], label='valid')
        plt.plot(range(*explore_range), reports_train['macro avg']['precision'], label='train')
        plt.legend()
        plt.grid()

        plt.subplot(2, 4, 3)
        plt.title('Macro Recall')
        plt.plot(range(*explore_range), reports_valid['macro avg']['recall'], label='valid')
        plt.plot(range(*explore_range), reports_train['macro avg']['recall'], label='train')
        plt.legend()
        plt.grid()

        plt.subplot(2, 4, 4)
        plt.title('Macro F1-score')
        plt.plot(range(*explore_range), reports_valid['macro avg']['f1-score'], label='valid')
        plt.plot(range(*explore_range), reports_train['macro avg']['f1-score'], label='train')
        plt.legend()
        plt.grid()

        plt.subplot(2, 4, 5)
        plt.title('Micro Accuracy')
        plt.plot(range(*explore_range), reports_valid['micro avg']['accuracy'], label='valid')
        plt.plot(range(*explore_range), reports_train['micro avg']['accuracy'], label='train')
        plt.legend()
        plt.xlabel('Max-Depth')
        plt.grid()

        plt.subplot(2, 4, 6)
        plt.title('Micro Precision')
        plt.plot(range(*explore_range), reports_valid['micro avg']['precision'], label='valid')
        plt.plot(range(*explore_range), reports_train['micro avg']['precision'], label='train')
        plt.legend()
        plt.xlabel('Max-Depth')
        plt.grid()

        plt.subplot(2, 4, 7)
        plt.title('Micro Recall')
        plt.plot(range(*explore_range), reports_valid['micro avg']['recall'], label='valid')
        plt.plot(range(*explore_range), reports_train['micro avg']['recall'], label='train')
        plt.legend()
        plt.xlabel('Max-Depth')
        plt.grid()

        plt.subplot(2, 4, 8)
        plt.title('Micro F1-score')
        plt.plot(range(*explore_range), reports_valid['micro avg']['f1-score'], label='valid')
        plt.plot(range(*explore_range), reports_train['micro avg']['f1-score'], label='train')
        plt.legend()
        plt.xlabel('Max-Depth')
        plt.grid()

        plt.show()

        return list(range(*explore_range))[
            reports_valid['micro avg']['f1-score'].index(max(reports_valid['micro avg']['f1-score']))]

    def min_samples_split_experiment(self, purity_function, max_depth):
        reports_train = {'macro avg': {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
                         'micro avg': {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}}
        reports_valid = {'macro avg': {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
                         'micro avg': {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}}

        explore_range = [0, 500, 50]
        for min_sample_split in range(*explore_range):
            report = evaluate_model(self.train_df,
                                    purity_function=purity_function,
                                    max_depth=max_depth,
                                    min_sample_split=min_sample_split)
            print(report)


            for avg_type in reports_valid.keys():
                for metric in reports_valid[avg_type].keys():
                    reports_valid[avg_type][metric].append(report[avg_type][metric]['valid'])
                    reports_train[avg_type][metric].append(report[avg_type][metric]['train'])
        print('----------------------------------')

        plt.subplot(2, 4, 1)
        plt.title('Macro Accuracy')
        plt.plot(range(*explore_range), reports_valid['macro avg']['accuracy'], label='valid')
        plt.plot(range(*explore_range), reports_train['macro avg']['accuracy'], label='train')
        plt.legend()
        plt.grid()

        plt.subplot(2, 4, 2)
        plt.title('Macro Precision')
        plt.plot(range(*explore_range), reports_valid['macro avg']['precision'], label='valid')
        plt.plot(range(*explore_range), reports_train['macro avg']['precision'], label='train')
        plt.legend()
        plt.grid()

        plt.subplot(2, 4, 3)
        plt.title('Macro Recall')
        plt.plot(range(*explore_range), reports_valid['macro avg']['recall'], label='valid')
        plt.plot(range(*explore_range), reports_train['macro avg']['recall'], label='train')
        plt.legend()
        plt.grid()

        plt.subplot(2, 4, 4)
        plt.title('Macro F1-score')
        plt.plot(range(*explore_range), reports_valid['macro avg']['f1-score'], label='valid')
        plt.plot(range(*explore_range), reports_train['macro avg']['f1-score'], label='train')
        plt.legend()
        plt.grid()

        plt.subplot(2, 4, 5)
        plt.title('Micro Accuracy')
        plt.plot(range(*explore_range), reports_valid['micro avg']['accuracy'], label='valid')
        plt.plot(range(*explore_range), reports_train['micro avg']['accuracy'], label='train')
        plt.legend()
        plt.xlabel('Min sample split')
        plt.grid()

        plt.subplot(2, 4, 6)
        plt.title('Micro Precision')
        plt.plot(range(*explore_range), reports_valid['micro avg']['precision'], label='valid')
        plt.plot(range(*explore_range), reports_train['micro avg']['precision'], label='train')
        plt.legend()
        plt.xlabel('Min sample split')
        plt.grid()

        plt.subplot(2, 4, 7)
        plt.title('Micro Recall')
        plt.plot(range(*explore_range), reports_valid['micro avg']['recall'], label='valid')
        plt.plot(range(*explore_range), reports_train['micro avg']['recall'], label='train')
        plt.legend()
        plt.xlabel('Min sample split')
        plt.grid()

        plt.subplot(2, 4, 8)
        plt.title('Micro F1-score')
        plt.plot(range(*explore_range), reports_valid['micro avg']['f1-score'], label='valid')
        plt.plot(range(*explore_range), reports_train['micro avg']['f1-score'], label='train')
        plt.legend()
        plt.xlabel('Min sample split')
        plt.grid()

        plt.show()

        return list(range(*explore_range))[
            reports_valid['micro avg']['f1-score'].index(max(reports_valid['micro avg']['f1-score']))]

    def post_pruning_experiment(self, purity_function, max_depth, min_sample_split):
        report_normal = evaluate_model(self.train_df,
                                       purity_function=purity_function,
                                       max_depth=max_depth,
                                       min_sample_split=min_sample_split,
                                       postpruning=False)
        report_prune = evaluate_model(self.train_df,
                                      purity_function=purity_function,
                                      max_depth=max_depth,
                                      min_sample_split=min_sample_split,
                                      postpruning=True)

        print(report_normal)
        print(report_prune)
        print('----------------------------------')

        plt.subplot(2, 4, 1)
        plt.title('Macro Accuracy')
        plt.bar(1, report_normal['macro avg']['accuracy']['valid'])
        plt.bar(2, report_prune['macro avg']['accuracy']['valid'])
        plt.bar(4, report_normal['macro avg']['accuracy']['train'])
        plt.bar(5, report_prune['macro avg']['accuracy']['train'])
        plt.xticks([1, 2, 4, 5], ['Normal-v', 'Pruned-v', 'Normal-t', 'Pruned-t'])
        plt.ylim(0.98, 1.02)

        plt.subplot(2, 4, 2)
        plt.title('Macro Precision')
        plt.bar(1, report_normal['macro avg']['precision']['valid'])
        plt.bar(2, report_prune['macro avg']['precision']['valid'])
        plt.bar(4, report_normal['macro avg']['precision']['train'])
        plt.bar(5, report_prune['macro avg']['precision']['train'])
        plt.xticks([1, 2, 4, 5], ['Normal-v', 'Pruned-v', 'Normal-t', 'Pruned-t'])
        plt.ylim(0.98, 1.02)

        plt.subplot(2, 4, 3)
        plt.title('Macro Recall')
        plt.bar(1, report_normal['macro avg']['recall']['valid'])
        plt.bar(2, report_prune['macro avg']['recall']['valid'])
        plt.bar(4, report_normal['macro avg']['recall']['train'])
        plt.bar(5, report_prune['macro avg']['recall']['train'])
        plt.xticks([1, 2, 4, 5], ['Normal-v', 'Pruned-v', 'Normal-t', 'Pruned-t'])
        plt.ylim(0.98, 1.02)

        plt.subplot(2, 4, 4)
        plt.title('Macro F1-score')
        plt.bar(1, report_normal['macro avg']['f1-score']['valid'])
        plt.bar(2, report_prune['macro avg']['f1-score']['valid'])
        plt.bar(4, report_normal['macro avg']['f1-score']['train'])
        plt.bar(5, report_prune['macro avg']['f1-score']['train'])
        plt.xticks([1, 2, 4, 5], ['Normal-v', 'Pruned-v', 'Normal-t', 'Pruned-t'])
        plt.ylim(0.98, 1.02)

        plt.subplot(2, 4, 5)
        plt.title('Micro Accuracy')
        plt.bar(1, report_normal['micro avg']['accuracy']['valid'])
        plt.bar(2, report_prune['micro avg']['accuracy']['valid'])
        plt.bar(4, report_normal['micro avg']['accuracy']['train'])
        plt.bar(5, report_prune['micro avg']['accuracy']['train'])
        plt.xticks([1, 2, 4, 5], ['Normal-v', 'Pruned-v', 'Normal-t', 'Pruned-t'])
        plt.ylim(0.98, 1.02)

        plt.subplot(2, 4, 6)
        plt.title('Micro Precision')
        plt.bar(1, report_normal['micro avg']['precision']['valid'])
        plt.bar(2, report_prune['micro avg']['precision']['valid'])
        plt.bar(4, report_normal['micro avg']['precision']['train'])
        plt.bar(5, report_prune['micro avg']['precision']['train'])
        plt.xticks([1, 2, 4, 5], ['Normal-v', 'Pruned-v', 'Normal-t', 'Pruned-t'])
        plt.ylim(0.98, 1.02)

        plt.subplot(2, 4, 7)
        plt.title('Micro Recall')
        plt.bar(1, report_normal['micro avg']['recall']['valid'])
        plt.bar(2, report_prune['micro avg']['recall']['valid'])
        plt.bar(4, report_normal['micro avg']['recall']['train'])
        plt.bar(5, report_prune['micro avg']['recall']['train'])
        plt.xticks([1, 2, 4, 5], ['Normal-v', 'Pruned-v', 'Normal-t', 'Pruned-t'])
        plt.ylim(0.98, 1.02)

        plt.subplot(2, 4, 8)
        plt.title('Micro F1-score')
        plt.bar(1, report_normal['micro avg']['f1-score']['valid'])
        plt.bar(2, report_prune['micro avg']['f1-score']['valid'])
        plt.bar(4, report_normal['micro avg']['f1-score']['train'])
        plt.bar(5, report_prune['micro avg']['f1-score']['train'])
        plt.xticks([1, 2, 4, 5], ['Normal-v', 'Pruned-v', 'Normal-t', 'Pruned-t'])
        plt.ylim(0.98, 1.02)

        plt.show()

        if report_normal['micro avg']['f1-score']['valid'] > report_prune['macro avg']['f1-score']['valid']:
            return False
        else:
            return True

    def sklearn_experiment(self):
        dt = DecisionTreeClassifier(criterion='gini',
                                    max_depth=18)

        dt.fit(self.train_df.drop('target', axis=1), self.train_df['target'])
        h = dt.predict(self.test_df.drop('target', axis=1))

        classes = self.train_df['target'].unique()
        report = classification_report(self.test_df['target'], h, output_dict=True)
        report['macro avg']['accuracy'] = report['accuracy']
        report['weighted avg']['accuracy'] = sum([accuracy_score(self.test_df['target'] == cls, h == cls) for cls in classes]) / len(classes)
        report = {'macro avg': report['macro avg'], 'micro avg': report['weighted avg']}
        print('sklearn score:')
        print(report)

    def run_experiments(self):
        purity_function = self.purity_function_experiment()
        max_depth = self.max_depth_experiment(purity_function)
        min_sample_split = self.min_samples_split_experiment(purity_function, max_depth)
        postpruning = self.post_pruning_experiment(purity_function, max_depth, min_sample_split)

        if postpruning:
            msk = np.random.rand(len(self.train_df)) < 0.7
            train_df = self.train_df[msk]
            valid_df = self.train_df[~msk]

            tree = make_decision_tree(train_df, 'target', classes=self.train_df['target'].unique(),
                                      return_prob=False, purity_function=purity_function,
                                      max_depth=max_depth, min_sample_split=min_sample_split)
            tree = post_pruning(tree, 'target', valid_df)
        else:
            tree = make_decision_tree(self.train_df, 'target', classes=self.train_df['target'].unique(),
                                      return_prob=False, purity_function=purity_function,
                                      max_depth=max_depth, min_sample_split=min_sample_split)

        print('My score')
        print(score_tree(tree, 'target', self.test_df))

        self.sklearn_experiment()


df = pd.read_csv('HW1-Dataset.csv')
df = df.rename(columns={'Grade (target)': 'target'})

exp = Experiments(df)
exp.run_experiments()
