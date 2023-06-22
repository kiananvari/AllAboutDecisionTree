import pandas as pd
from pprint import pprint
import json


def gini_index(data: pd.DataFrame, feature, label, classes):
    total_gini = 0
    samples = len(data)
    for value in data[feature].unique():
        sub_tree_labels = data[data[feature] == value][label].value_counts()
        sub_tree_samples = len(data[data[feature] == value])
        sum_gini = 0
        for c in classes:
            pi = sub_tree_labels.get(c, 0) / sub_tree_samples
            sum_gini += pi ** 2
        sum_gini = 1 - sum_gini
        sum_gini = sum_gini * sub_tree_samples
        total_gini += sum_gini / samples

    return total_gini


def calculate_best_split(data: pd.DataFrame, label, classes):
    available_features = set(data.columns) - set([label])
    assert len(available_features) > 0

    best_gini = 1
    best_feature = None
    for f in available_features:
        gini = gini_index(data, f, label, classes)
        if gini < best_gini:
            best_gini = gini
            best_feature = f

    return best_feature


def split_data(data: pd.DataFrame, feature):
    sub_datas = {}
    for value in data[feature].unique():
        sub_tree_data = data[data[feature] == value]
        sub_tree_data = sub_tree_data.drop(feature, axis=1)
        sub_datas[value] = sub_tree_data

    return sub_datas


def make_decision_tree(data: pd.DataFrame, label='label', classes=None, return_prob=True):
    available_features = set(data.columns) - set([label])
    if len(available_features) == 0 or len(data[label].unique()) == 1:
        if return_prob:
            return json.loads((data['label'].value_counts() / len(data)).to_json())
        else:
            return {'label': data[label].value_counts().idxmax()}

    split_feature = calculate_best_split(data, label, classes)
    sub_datas = split_data(data, split_feature)

    if len(sub_datas.keys()) == 1:
        if return_prob:
            return json.loads((data['label'].value_counts() / len(data)).to_json())
        else:
            return {'label': data[label].value_counts().idxmax()}

    results = (split_feature, {})
    for key in sub_datas.keys():
        results[1][key] = make_decision_tree(sub_datas[key], label, classes)

    return results


def explain_tree(json_data, pre_sentence=''):
    if type(json_data) == dict:
        if 'label' in json_data.keys():
            pre_sentence += f' then, prediction is {json_data["label"]}'
        else:
            pre_sentence += ' then, prediction is '
            for value in json_data.keys():
                pre_sentence += f'{value} or '
            pre_sentence = pre_sentence[:-4]
        print(pre_sentence)
        return

    feature, splits = json_data
    if len(pre_sentence) != 0:
        pre_sentence = pre_sentence + ' and '

    sentence = pre_sentence + f'if {feature} is '

    for key in splits.keys():
        explain_tree(splits[key], sentence + key)

data = [
    {'weather': 'sun',  'cash': 'yes',  'exam': 'no',   'label': 'shop'},
    {'weather': 'sun',  'cash': 'yes',  'exam': 'yes',  'label': 'coffe'},
    {'weather': 'snow', 'cash': 'yes',  'exam': 'no',   'label': 'coffe'},
    {'weather': 'sun',  'cash': 'yes',  'exam': 'yes',  'label': 'hiking'},
    {'weather': 'rain', 'cash': 'yes',  'exam': 'yes',  'label': 'stay'},
    {'weather': 'sun',  'cash': 'no',   'exam': 'no',  'label': 'hiking'},
    {'weather': 'sun',  'cash': 'yes',  'exam': 'no',   'label': 'hiking'},
    {'weather': 'rain', 'cash': 'yes',  'exam': 'yes',  'label': 'coffe'},
    {'weather': 'snow', 'cash': 'yes',  'exam': 'yes',  'label': 'shop'},
    {'weather': 'snow', 'cash': 'no',   'exam': 'yes',  'label': 'stay'}
]

df = pd.DataFrame(data)

classes = df['label'].unique()
tree = make_decision_tree(df, 'label', classes)
pprint(tree)
explain_tree(tree)
