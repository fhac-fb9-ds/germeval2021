import hashlib
import os
import re

import emoji
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, \
    EarlyStoppingCallback, set_seed

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def load_datasets():
    # load dataset
    df_train = pd.read_csv('../../dataset/GermEval21_Toxic_Train.csv', index_col=0)
    df_test = pd.read_csv('../../dataset/GermEval21_Toxic_TestData.csv', index_col=0)

    # set column and index names
    df_train.rename(columns={'comment_text': 'text',
                             'Sub1_Toxic': 'toxic',
                             'Sub2_Engaging': 'engaging',
                             'Sub3_FactClaiming': 'fact'}, inplace=True)
    df_train.index.rename('id', inplace=True)

    df_test.rename(columns={'comment_text': 'text',
                            'Sub1_Toxic': 'toxic',
                            'Sub2_Engaging': 'engaging',
                            'Sub3_FactClaiming': 'fact'}, inplace=True)
    df_test.index.rename('id', inplace=True)

    # remove duplicates
    df_train.drop_duplicates(inplace=True)

    # shuffle dataset randomly
    df_train = df_train.sample(frac=1, random_state=9).reset_index(drop=True)

    return df_train, df_test


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


def get_hugging_face_name(name):
    if name == 'gbert':
        return 'deepset/gbert-large'
    if name == 'gelectra':
        return 'deepset/gelectra-large'
    if name == 'gottbert':
        return 'uklfr/gottbert-base'
    return ''


def compute_scores_for_threshold(trainer, dataset):
    s_t, s_e, s_f = np.array([]), np.array([]), np.array([])
    pred_proba = sigmoid(trainer.predict(dataset).predictions)
    for t in np.arange(0, 1.025, 0.025):
        pred = (pred_proba >= t) * 1
        s_t = np.append(s_t, f1_score(dataset.labels[:, 0], pred[:, 0]))
        s_e = np.append(s_e, f1_score(dataset.labels[:, 1], pred[:, 1]))
        s_f = np.append(s_f, f1_score(dataset.labels[:, 2], pred[:, 2]))
    s_t = s_t.reshape((len(s_t), 1))
    s_e = s_e.reshape((len(s_e), 1))
    s_f = s_f.reshape((len(s_f), 1))
    return s_t, s_e, s_f


if __name__ == '__main__':
    # relevant inputs
    model_count = 200
    model_names = ['gelectra', 'gbert']
    # model_names = ['gbert', 'gelectra', 'gottbert']

    df_train, df_test = load_datasets()
    df_train = clean_up_comments(df_train)
    df_test = clean_up_comments(df_test)
    y_test = df_test[["toxic", "engaging", "fact"]].to_numpy()

    predictions_test = []
    sco_thr_t = []
    sco_thr_e = []
    sco_thr_f = []

    for i, model_name in enumerate(model_names):
        tokenizer = AutoTokenizer.from_pretrained(get_hugging_face_name(model_name))
        tokens_test = tokenizer(df_test['text'].tolist(), return_tensors='pt', padding='max_length', truncation=True,
                                max_length=200)
        dataset_test = GermEvalDataset(tokens_test, y_test)

        for k in range(0, model_count):
            df_train_val = df_train.sample(frac=0.1, random_state=k)
            df_train_train = df_train.drop(df_train[df_train['text'].isin(df_train_val['text'])].index)

            tokens_train_train = tokenizer(df_train_train['text'].tolist(), return_tensors='pt',
                                           padding='max_length', truncation=True, max_length=200)
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
                )
                trainer.train()
                model.save_pretrained('../models/' + hash)

            t_t, t_e, t_f = compute_scores_for_threshold(trainer, dataset_train_val)
            if len(sco_thr_t) == 0:
                sco_thr_t = t_t
            else:
                sco_thr_t = np.hstack((sco_thr_t, t_t))
            if len(sco_thr_e) == 0:
                sco_thr_e = t_e
            else:
                sco_thr_e = np.hstack((sco_thr_e, t_e))
            if len(sco_thr_f) == 0:
                sco_thr_f = t_f
            else:
                sco_thr_f = np.hstack((sco_thr_f, t_f))

            pred = sigmoid(trainer.predict(dataset_test).predictions)
            if len(predictions_test) == 0:
                predictions_test = pred
            else:
                predictions_test = predictions_test + pred

    best_t_t = np.argmax(np.mean(sco_thr_t, axis=1)) * 0.025
    best_t_e = np.argmax(np.mean(sco_thr_e, axis=1)) * 0.025
    best_t_f = np.argmax(np.mean(sco_thr_f, axis=1)) * 0.025

    y_pred_proba = predictions_test / (model_count * len(model_names))
    y_pred = (y_pred_proba >= [best_t_t, best_t_e, best_t_f]) * 1

    df_test['Sub1_Toxic'] = y_pred[:, 0]
    df_test['Sub2_Engaging'] = y_pred[:, 1]
    df_test['Sub3_FactClaiming'] = y_pred[:, 2]
    df_test = df_test.drop(columns=['text', 'toxic', 'engaging', 'fact'])
    df_test.index.rename('comment_id', inplace=True)
    df_test.to_csv('results/answer.csv')

    with open('results/thresholds.txt', 'w') as f:
        f.write(f'optimal threshold for classifications of class toxic: {best_t_t}')
        f.write(f'optimal threshold for classifications of class engaging: {best_t_e}')
        f.write(f'optimal threshold for classifications of class claiming: {best_t_f}')

    with open('results/scores.txt', 'w') as f:
        f.write(f'F1 score for class toxic: {f1_score(y_test[:, 0], y_pred[:, 0])}')
        f.write(f'F1 score for class engaging: {f1_score(y_test[:, 1], y_pred[:, 1])}')
        f.write(f'F1 score for class fact-claiming: {f1_score(y_test[:, 2], y_pred[:, 2])}')
        f.write(f'macro F1 score: {f1_score(y_test, y_pred, average="macro")}')
