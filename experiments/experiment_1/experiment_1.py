import hashlib
import os
import re

import emoji
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, \
    EarlyStoppingCallback, set_seed

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def load_dataset():
    # load dataset
    df = pd.read_csv('../../dataset/GermEval21_Toxic_Train.csv', index_col=0)

    # set column and index names
    df.rename(columns={'comment_text': 'text',
                       'Sub1_Toxic': 'toxic',
                       'Sub2_Engaging': 'engaging',
                       'Sub3_FactClaiming': 'fact'}, inplace=True)
    df.index.rename('id', inplace=True)

    # remove duplicates
    df.drop_duplicates(inplace=True)

    # shuffle dataset randomly
    df = df.sample(frac=1, random_state=9).reset_index(drop=True)

    return df


def remove_in_word_whitespaces(comment):
    find = re.findall(r'(^| )(([a-zA-zäöüß] ){1,}[a-zA-zäöüß!?,.]([^a-zA-zäöüß]|$))', comment)
    if len(find) > 0:
        for match in find:
            found = match[0] + match[1]
            replacement = ' ' + re.sub(r' ', '', found) + ' '
            comment = comment.replace(found, replacement, 1)
    return comment


def demojize(comment):
    return emoji.demojize(comment, delimiters=(' <', '> '))


def clean_up_comments(df):
    # insert whitespaces before and after emojis so they are tokenized as separate tokens
    df['text'] = df['text'].apply(lambda t: demojize(t))
    df['text'] = df['text'].apply(lambda t: emoji.emojize(t, delimiters=('<', '>')))

    # convert terms like "a k t u e l l" to "aktuell"
    df['text'] = df['text'].apply(lambda t: remove_in_word_whitespaces(t))

    # trim mutliple whitespace characters
    df['text'] = df['text'].str.replace(r' {2,}', ' ', regex=True)

    # strip outer whitespaces
    df['text'] = df['text'].str.strip()
    return df


class GermEvalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (sigmoid(logits) >= 0.5) * 1
    return {'F1': f1_score(labels, predictions, average='macro')}


def plot_result(cross_val_scores, scoring, label):
    # compute means of the 4 scores
    means = [
        np.mean(cross_val_scores["test_Precision"]),
        np.mean(cross_val_scores["test_Recall"]),
        np.mean(cross_val_scores["test_F1"]),
        np.mean(cross_val_scores["test_AUC"])
    ]
    # compute the std of the 4 scores
    err = [
        np.std(cross_val_scores["test_Precision"]),
        np.std(cross_val_scores["test_Recall"]),
        np.std(cross_val_scores["test_F1"]),
        np.std(cross_val_scores["test_AUC"])
    ]
    width = 0.35

    # plot the bar plot
    plt.ylim([0, 1])
    plt.bar(np.arange(len(means)), means, width=width, yerr=err, align='center', capsize=10, alpha=0.75)
    plt.xticks(range(len(means)), scoring.keys(), rotation=0)
    plt.title("scores of the " + label.capitalize() + " classifier")
    plt.savefig('results/' + label + '.png')
    plt.close()


def plot_macro_scores(scores_toxic, scores_engaging, scores_fact, scoring):
    # compute means of the 4 scores
    means = [
        np.mean([
            np.mean(scores_toxic["test_Precision"]),
            np.mean(scores_engaging["test_Precision"]),
            np.mean(scores_fact["test_Precision"])
        ]),
        np.mean([
            np.mean(scores_toxic["test_Recall"]),
            np.mean(scores_engaging["test_Recall"]),
            np.mean(scores_fact["test_Recall"])
        ]),
        np.mean([
            np.mean(scores_toxic["test_F1"]),
            np.mean(scores_engaging["test_F1"]),
            np.mean(scores_fact["test_F1"])
        ]),
        np.mean([
            np.mean(scores_toxic["test_AUC"]),
            np.mean(scores_engaging["test_AUC"]),
            np.mean(scores_fact["test_AUC"])
        ])
    ]
    # compute the std of scores_toxic
    err = [
        np.mean([
            np.std(scores_toxic["test_Precision"]),
            np.std(scores_engaging["test_Precision"]),
            np.std(scores_fact["test_Precision"])
        ]),
        np.mean([
            np.std(scores_toxic["test_Recall"]),
            np.std(scores_engaging["test_Recall"]),
            np.std(scores_fact["test_Recall"])
        ]),
        np.mean([
            np.std(scores_toxic["test_F1"]),
            np.std(scores_engaging["test_F1"]),
            np.std(scores_fact["test_F1"])
        ]),
        np.mean([
            np.std(scores_toxic["test_AUC"]),
            np.std(scores_engaging["test_AUC"]),
            np.std(scores_fact["test_AUC"])
        ])
    ]

    width = 0.35

    # plot the bar plot
    plt.ylim([0, 1])
    plt.bar(np.arange(len(means)), means, width=width, yerr=err, align='center', capsize=10, alpha=0.75)
    plt.xticks(range(len(means)), scoring.keys(), rotation=0)
    plt.title("mean and standard deviation of scores\nover all classes")
    plt.savefig("results/macro.png")
    plt.close()

    # print macro scores
    print('Macro')
    print('Precision:', "%0.2f" % (means[0] * 100), '+-', "%0.2f" % (err[0] * 100))
    print('Recall:', "%0.2f" % (means[1] * 100), '+-', "%0.2f" % (err[1] * 100))
    print('F1:', "%0.2f" % (means[2] * 100), '+-', "%0.2f" % (err[2] * 100))
    print('AUC:', "%0.2f" % (means[3] * 100), '+-', "%0.2f" % (err[3] * 100))
    print()


def write_scores_to_file(scores, title):
    means = [
        np.mean(scores["test_Precision"]),
        np.mean(scores["test_Recall"]),
        np.mean(scores["test_F1"]),
        np.mean(scores["test_AUC"])
    ]
    err = [
        np.std(scores["test_Precision"]),
        np.std(scores["test_Recall"]),
        np.std(scores["test_F1"]),
        np.std(scores["test_AUC"])
    ]

    f = open(title + '.txt', "w")
    f.write(title)
    f.write('\nPrecision: ' + '%0.2f' % (means[0] * 100) + ' +- ' + '%0.2f' % (err[0] * 100))
    f.write('\nRecall: ' + '%0.2f' % (means[1] * 100) + ' +- ' + '%0.2f' % (err[1] * 100))
    f.write('\nF1: ' + '%0.2f' % (means[2] * 100) + ' +- ' + '%0.2f' % (err[2] * 100))
    f.write('\nAUC: ' + '%0.2f' % (means[3] * 100) + ' +- ' + '%0.2f' % (err[3] * 100))
    f.close()

    # print scores
    print(title)
    print('Precision:', "%0.2f" % (means[0] * 100), '+-', "%0.2f" % (err[0] * 100))
    print('Recall:', "%0.2f" % (means[1] * 100), '+-', "%0.2f" % (err[1] * 100))
    print('F1:', "%0.2f" % (means[2] * 100), '+-', "%0.2f" % (err[2] * 100))
    print('AUC:', "%0.2f" % (means[3] * 100), '+-', "%0.2f" % (err[3] * 100))
    print()


def get_hugging_face_name(name):
    if name == 'gbert':
        return 'deepset/gbert-large'
    if name == 'gelectra':
        return 'deepset/gelectra-large'
    if name == 'gottbert':
        return 'uklfr/gottbert-base'
    return ''


if __name__ == '__main__':
    # relevant inputs
    model_count = 50
    model_names = ['gelectra']
    # model_names = ['gbert', 'gelectra', 'gottbert']

    df = load_dataset()
    df = clean_up_comments(df)

    scores_toxic = {
        'test_Precision': np.array([]),
        'test_Recall': np.array([]),
        'test_F1': np.array([]),
        'test_AUC': np.array([]),
    }
    scores_engaging = {
        'test_Precision': np.array([]),
        'test_Recall': np.array([]),
        'test_F1': np.array([]),
        'test_AUC': np.array([]),
    }
    scores_fact = {
        'test_Precision': np.array([]),
        'test_Recall': np.array([]),
        'test_F1': np.array([]),
        'test_AUC': np.array([]),
    }
    scores_macro = {
        'test_Precision': np.array([]),
        'test_Recall': np.array([]),
        'test_F1': np.array([]),
        'test_AUC': np.array([]),
    }

    # parameters needed for threshold optimization
    best_thresholds = {'toxic': [], 'engaging': [], 'fact': []}
    thresholds = np.linspace(0, 1, 41)

    for i, (train_index, val_index) in enumerate(KFold().split(df)):
        df_train = df.loc[train_index]
        df_val = df.loc[val_index]
        y_val = df_val[["toxic", "engaging", "fact"]].to_numpy()
        predictions = []

        f1_toxic = {c: [] for c in thresholds}
        f1_engaging = {c: [] for c in thresholds}
        f1_fact = {c: [] for c in thresholds}

        for model_name in model_names:
            tokenizer = AutoTokenizer.from_pretrained(get_hugging_face_name(model_name))
            tokens_val = tokenizer(df_val['text'].tolist(), return_tensors='pt', padding='max_length', truncation=True,
                                   max_length=200)
            dataset_val = GermEvalDataset(tokens_val, y_val)

            for k in range(0, model_count):
                df_train_val = df_train.sample(frac=0.1, random_state=k)
                y_train_val = df_train_val[["toxic", "engaging", "fact"]].to_numpy()
                df_train_train = df_train.drop(df_train[df_train['text'].isin(df_train_val['text'])].index)

                tokens_train_train = tokenizer(df_train_train['text'].tolist(), return_tensors='pt',
                                               padding='max_length',
                                               truncation=True, max_length=200)
                tokens_train_val = tokenizer(df_train_val['text'].tolist(), return_tensors='pt', padding='max_length',
                                             truncation=True, max_length=200)

                dataset_train_train = GermEvalDataset(tokens_train_train,
                                                      df_train_train[["toxic", "engaging", "fact"]].to_numpy())
                dataset_train_val = GermEvalDataset(tokens_train_val,
                                                    df_train_val[["toxic", "engaging", "fact"]].to_numpy())

                hash = hashlib.sha256(pd.util.hash_pandas_object(df_train_train,
                                                                 index=True).values).hexdigest() + '_' + get_hugging_face_name(
                    model_name)[get_hugging_face_name(model_name).find('/') + 1:]

                training_args = TrainingArguments(f'{model_name}_trainer',
                                                  no_cuda=False,
                                                  metric_for_best_model='F1',
                                                  load_best_model_at_end=True,
                                                  num_train_epochs=10,
                                                  eval_steps=40,
                                                  # eval_steps=1,
                                                  evaluation_strategy='steps',
                                                  per_device_train_batch_size=24,
                                                  # per_device_train_batch_size=2,
                                                  seed=i * 100 + k,
                                                  learning_rate=5e-5,
                                                  warmup_ratio=0.3)

                model = None
                try:
                    model = AutoModelForSequenceClassification.from_pretrained('../models/' + hash,
                                                                               local_files_only=True,
                                                                               num_labels=3)
                    trainer = MultilabelTrainer(
                        model=model,
                        args=training_args,
                        train_dataset=dataset_train_train,
                        eval_dataset=dataset_train_val,
                        compute_metrics=compute_metrics,
                    )
                except EnvironmentError:
                    set_seed(training_args.seed)
                    model = AutoModelForSequenceClassification.from_pretrained(get_hugging_face_name(model_name),
                                                                               num_labels=3)
                    trainer = MultilabelTrainer(
                        model=model,
                        args=training_args,
                        train_dataset=dataset_train_train,
                        eval_dataset=dataset_train_val,
                        compute_metrics=compute_metrics,
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
                        # callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
                    )
                    trainer.train()
                    model.save_pretrained('../models/' + hash)

                pred = sigmoid(trainer.predict(dataset_val).predictions)
                if len(predictions) == 0:
                    predictions = pred
                else:
                    predictions = predictions + pred

                pred_train_val = sigmoid(trainer.predict(dataset_train_val).predictions)
                for th in thresholds:
                    y_pred = (pred_train_val >= th) * 1
                    f1_toxic[th].append(f1_score(y_train_val[:, 0], y_pred[:, 0]))
                    f1_engaging[th].append(f1_score(y_train_val[:, 1], y_pred[:, 1]))
                    f1_fact[th].append(f1_score(y_train_val[:, 2], y_pred[:, 2]))

        y_pred_proba = predictions / (model_count * len(model_names))

        f1_toxic_mean = {c: np.mean(f1) for c, f1 in f1_toxic.items()}
        f1_engaging_mean = {c: np.mean(f1) for c, f1 in f1_engaging.items()}
        f1_fact_mean = {c: np.mean(f1) for c, f1 in f1_fact.items()}
        best_thresholds['toxic'].append(max(f1_toxic_mean, key=lambda k: f1_toxic_mean[k]))
        best_thresholds['engaging'].append(max(f1_engaging_mean, key=lambda k: f1_engaging_mean[k]))
        best_thresholds['fact'].append(max(f1_fact_mean, key=lambda k: f1_fact_mean[k]))

        y_pred = (y_pred_proba >= [best_thresholds['toxic'][-1], best_thresholds['engaging'][-1],
                                   best_thresholds['fact'][-1]]) * 1

        # Set toxic scores
        scores_toxic['test_Precision'] = np.append(scores_toxic['test_Precision'],
                                                   precision_score(y_val[:, 0], y_pred[:, 0]))
        scores_toxic['test_Recall'] = np.append(scores_toxic['test_Recall'],
                                                recall_score(y_val[:, 0], y_pred[:, 0]))
        scores_toxic['test_F1'] = np.append(scores_toxic['test_F1'],
                                            f1_score(y_val[:, 0], y_pred[:, 0]))
        try:
            scores_toxic['test_AUC'] = np.append(scores_toxic['test_AUC'],
                                                 roc_auc_score(y_val[:, 0], y_pred_proba[:, 0]))
        except ValueError:
            scores_toxic['test_AUC'] = np.append(scores_toxic['test_AUC'], 0)

        # Set engaging scores
        scores_engaging['test_Precision'] = np.append(scores_engaging['test_Precision'],
                                                      precision_score(y_val[:, 1], y_pred[:, 1]))
        scores_engaging['test_Recall'] = np.append(scores_engaging['test_Recall'],
                                                   recall_score(y_val[:, 1], y_pred[:, 1]))
        scores_engaging['test_F1'] = np.append(scores_engaging['test_F1'],
                                               f1_score(y_val[:, 1], y_pred[:, 1]))
        try:
            scores_engaging['test_AUC'] = np.append(scores_engaging['test_AUC'],
                                                    roc_auc_score(y_val[:, 1], y_pred_proba[:, 1]))
        except ValueError:
            scores_engaging['test_AUC'] = np.append(scores_engaging['test_AUC'], 0)

        # Set fact scores
        scores_fact['test_Precision'] = np.append(scores_fact['test_Precision'],
                                                  precision_score(y_val[:, 2], y_pred[:, 2]))
        scores_fact['test_Recall'] = np.append(scores_fact['test_Recall'],
                                               recall_score(y_val[:, 2], y_pred[:, 2]))
        scores_fact['test_F1'] = np.append(scores_fact['test_F1'],
                                           f1_score(y_val[:, 2], y_pred[:, 2]))
        try:
            scores_fact['test_AUC'] = np.append(scores_fact['test_AUC'],
                                                roc_auc_score(y_val[:, 2], y_pred_proba[:, 2]))
        except ValueError:
            scores_fact['test_AUC'] = np.append(scores_fact['test_AUC'], 0)

        # Set macro scores
        scores_macro['test_Precision'] = np.append(scores_macro['test_Precision'],
                                                   precision_score(y_val, y_pred, average='macro'))
        scores_macro['test_Recall'] = np.append(scores_macro['test_Recall'],
                                                recall_score(y_val, y_pred, average='macro'))
        scores_macro['test_F1'] = np.append(scores_macro['test_F1'],
                                            f1_score(y_val, y_pred, average='macro'))
        try:
            scores_macro['test_AUC'] = np.append(scores_macro['test_AUC'],
                                                 roc_auc_score(y_val, y_pred_proba, average='macro'))
        except ValueError:
            scores_macro['test_AUC'] = np.append(scores_macro['test_AUC'], 0)

    sco = {'Precision': 'precision',
           'Recall': 'recall',
           'F1': 'f1',
           'AUC': 'roc_auc'}

    plot_result(scores_toxic, sco, 'toxic')
    write_scores_to_file(scores_toxic, 'toxic')
    plot_result(scores_engaging, sco, 'engaging')
    write_scores_to_file(scores_engaging, 'engaging')
    plot_result(scores_fact, sco, 'fact-claiming')
    write_scores_to_file(scores_fact, 'fact-claiming')
    plot_macro_scores(scores_macro, scores_macro, scores_macro, sco)
    write_scores_to_file(scores_macro, 'macro')
    with open('results/thresholds.txt', 'w') as f:
        for k, v in best_thresholds.items():
            f.write(f'{k} all: {v}\n')
            f.write(f'{k} mean: {np.mean(v)}\n')
