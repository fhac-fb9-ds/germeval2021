import hashlib
import os
import re

import emoji
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, \
    EarlyStoppingCallback, set_seed

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_dir = '../../experiments/models/'


def load_dataset():
    # load dataset
    df_train = pd.read_csv('../../dataset/GermEval21_Toxic_Train.csv', index_col=0)

    # set column and index names
    df_train.rename(columns={'comment_text': 'text',
                             'Sub1_Toxic': 'toxic',
                             'Sub2_Engaging': 'engaging',
                             'Sub3_FactClaiming': 'fact'}, inplace=True)
    df_train.index.rename('id', inplace=True)

    # remove duplicates
    df_train.drop_duplicates(inplace=True)

    # shuffle dataset randomly
    df_train = df_train.sample(frac=1, random_state=9).reset_index(drop=True)

    return df_train


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


def compute_metrics_multilabel(eval_pred):
    logits, labels = eval_pred
    predictions = (sigmoid(logits) >= 0.5) * 1
    return {'F1': calc_f1_score_germeval(labels, predictions)}


def compute_metrics_singlelabel(eval_pred):
    logits, labels = eval_pred
    softmax = torch.nn.Softmax(dim=1)
    predictions = np.argmax(softmax(torch.tensor(logits)), axis=-1).detach().cpu().numpy()
    return {'F1': calc_f1_score_germeval(labels, predictions)}


def get_hugging_face_name(name):
    if name == 'gbert':
        return 'deepset/gbert-large'
    if name == 'gelectra':
        return 'deepset/gelectra-large'
    if name == 'gottbert':
        return 'uklfr/gottbert-base'
    return ''


def get_gelectra_multilabel(df_train, df_test, size):
    tokenizer = AutoTokenizer.from_pretrained(get_hugging_face_name('gelectra'))
    tokens_test = tokenizer(df_test['text'].tolist(), return_tensors='pt', padding='max_length', truncation=True,
                            max_length=200)
    dataset_test = GermEvalDataset(tokens_test, np.zeros((len(df_test), 3)))
    predictions = np.array([])
    f1_thresholds = np.zeros((size, len(thresholds), 3))

    for k in range(0, size):
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
                                                         index=True).values).hexdigest() + '_gelectra-large'

        training_args = TrainingArguments(f'gelectra_trainer',
                                          no_cuda=False,
                                          metric_for_best_model='F1',
                                          load_best_model_at_end=True,
                                          num_train_epochs=10,
                                          eval_steps=40,
                                          per_device_train_batch_size=24,
                                          evaluation_strategy='steps',
                                          seed=k,
                                          learning_rate=5e-5,
                                          warmup_ratio=0.3)

        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir + hash,
                                                                       local_files_only=True,
                                                                       num_labels=3)
            trainer = MultilabelTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset_train_train,
                eval_dataset=dataset_train_val,
                compute_metrics=compute_metrics_multilabel,
            )
        except EnvironmentError:
            set_seed(training_args.seed)
            model = AutoModelForSequenceClassification.from_pretrained(get_hugging_face_name('gelectra'),
                                                                       num_labels=3)
            trainer = MultilabelTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset_train_train,
                eval_dataset=dataset_train_val,
                compute_metrics=compute_metrics_multilabel,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            trainer.train()
            model.save_pretrained(model_dir + hash)

        pred = np.array([sigmoid(trainer.predict(dataset_test).predictions)])
        if len(predictions) == 0:
            predictions = pred
        else:
            predictions = np.append(predictions, pred, axis=0)

        pred_train_val = sigmoid(trainer.predict(dataset_train_val).predictions)
        for i, th in enumerate(thresholds):
            y_pred = (pred_train_val >= th) * 1
            f1_thresholds[k, i, 0] = calc_f1_score_germeval(y_train_val[:, 0], y_pred[:, 0])
            f1_thresholds[k, i, 1] = calc_f1_score_germeval(y_train_val[:, 1], y_pred[:, 1])
            f1_thresholds[k, i, 2] = calc_f1_score_germeval(y_train_val[:, 2], y_pred[:, 2])

    return predictions, f1_thresholds


def get_gelectra_singlelabel(df_train, df_test, size, label):
    tokenizer = AutoTokenizer.from_pretrained(get_hugging_face_name('gelectra'))
    tokens_test = tokenizer(df_test['text'].tolist(), return_tensors='pt', padding='max_length', truncation=True,
                            max_length=200)
    dataset_test = GermEvalDataset(tokens_test, np.zeros(len(df_test), dtype=np.longlong))
    predictions = np.array([])

    for k in range(0, size):
        df_train_val = df_train.sample(frac=0.1, random_state=k)
        df_train_train = df_train.drop(df_train[df_train['text'].isin(df_train_val['text'])].index)

        tokens_train_train = tokenizer(df_train_train['text'].tolist(), return_tensors='pt',
                                       padding='max_length',
                                       truncation=True, max_length=200)
        tokens_train_val = tokenizer(df_train_val['text'].tolist(), return_tensors='pt', padding='max_length',
                                     truncation=True, max_length=200)

        dataset_train_train = GermEvalDataset(tokens_train_train,
                                              df_train_train[label].to_numpy())
        dataset_train_val = GermEvalDataset(tokens_train_val,
                                            df_train_val[label].to_numpy())

        hash = hashlib.sha256(pd.util.hash_pandas_object(df_train_train,
                                                         index=True).values).hexdigest() + '_gelectra-large_' + label

        training_args = TrainingArguments(f'gelectra_trainer',
                                          no_cuda=False,
                                          metric_for_best_model='F1',
                                          load_best_model_at_end=True,
                                          num_train_epochs=10,
                                          eval_steps=40,
                                          per_device_train_batch_size=24,
                                          evaluation_strategy='steps',
                                          seed=k,
                                          learning_rate=5e-5,
                                          warmup_ratio=0.3)

        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir + hash,
                                                                       local_files_only=True,
                                                                       num_labels=2)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset_train_train,
                eval_dataset=dataset_train_val,
                compute_metrics=compute_metrics_singlelabel,
            )
        except EnvironmentError:
            set_seed(training_args.seed)
            model = AutoModelForSequenceClassification.from_pretrained(get_hugging_face_name('gelectra'),
                                                                       num_labels=2)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset_train_train,
                eval_dataset=dataset_train_val,
                compute_metrics=compute_metrics_singlelabel,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            trainer.train()
            model.save_pretrained(model_dir + hash)

        logits = trainer.predict(dataset_test).predictions
        softmax = torch.nn.Softmax(dim=1)
        pred = np.array([softmax(torch.tensor(logits)).detach().cpu().numpy()])
        if len(predictions) == 0:
            predictions = pred
        else:
            predictions = np.append(predictions, pred, axis=0)

    return predictions


def get_gbert_multilabel(df_train, df_test, size):
    tokenizer = AutoTokenizer.from_pretrained(get_hugging_face_name('gbert'))
    tokens_test = tokenizer(df_test['text'].tolist(), return_tensors='pt', padding='max_length', truncation=True,
                            max_length=200)
    dataset_test = GermEvalDataset(tokens_test, np.zeros((len(df_test), 3)))
    predictions = np.array([])
    f1_thresholds = np.zeros((size, len(thresholds), 3))

    for k in range(0, size):
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
                                                         index=True).values).hexdigest() + '_gbert-large'

        training_args = TrainingArguments(f'gbert_trainer',
                                          no_cuda=False,
                                          metric_for_best_model='F1',
                                          load_best_model_at_end=True,
                                          num_train_epochs=10,
                                          eval_steps=40,
                                          per_device_train_batch_size=24,
                                          evaluation_strategy='steps',
                                          seed=k,
                                          learning_rate=5e-5,
                                          warmup_ratio=0.3)

        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir + hash,
                                                                       local_files_only=True,
                                                                       num_labels=3)
            trainer = MultilabelTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset_train_train,
                eval_dataset=dataset_train_val,
                compute_metrics=compute_metrics_multilabel,
            )
        except EnvironmentError:
            set_seed(training_args.seed)
            model = AutoModelForSequenceClassification.from_pretrained(get_hugging_face_name('gbert'),
                                                                       num_labels=3)
            trainer = MultilabelTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset_train_train,
                eval_dataset=dataset_train_val,
                compute_metrics=compute_metrics_multilabel,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            trainer.train()
            model.save_pretrained(model_dir + hash)

        pred = np.array([sigmoid(trainer.predict(dataset_test).predictions)])
        if len(predictions) == 0:
            predictions = pred
        else:
            predictions = np.append(predictions, pred, axis=0)

        pred_train_val = sigmoid(trainer.predict(dataset_train_val).predictions)
        for i, th in enumerate(thresholds):
            y_pred = (pred_train_val >= th) * 1
            f1_thresholds[k, i, 0] = calc_f1_score_germeval(y_train_val[:, 0], y_pred[:, 0])
            f1_thresholds[k, i, 1] = calc_f1_score_germeval(y_train_val[:, 1], y_pred[:, 1])
            f1_thresholds[k, i, 2] = calc_f1_score_germeval(y_train_val[:, 2], y_pred[:, 2])

    return predictions, f1_thresholds


def get_gbert_singlelabel(df_train, df_test, size, label):
    tokenizer = AutoTokenizer.from_pretrained(get_hugging_face_name('gbert'))
    tokens_test = tokenizer(df_test['text'].tolist(), return_tensors='pt', padding='max_length', truncation=True,
                            max_length=200)
    dataset_test = GermEvalDataset(tokens_test, np.zeros(len(df_test), dtype=np.longlong))
    predictions = np.array([])

    for k in range(0, size):
        df_train_val = df_train.sample(frac=0.1, random_state=k)
        df_train_train = df_train.drop(df_train[df_train['text'].isin(df_train_val['text'])].index)

        tokens_train_train = tokenizer(df_train_train['text'].tolist(), return_tensors='pt',
                                       padding='max_length',
                                       truncation=True, max_length=200)
        tokens_train_val = tokenizer(df_train_val['text'].tolist(), return_tensors='pt', padding='max_length',
                                     truncation=True, max_length=200)

        dataset_train_train = GermEvalDataset(tokens_train_train,
                                              df_train_train[label].to_numpy())
        dataset_train_val = GermEvalDataset(tokens_train_val,
                                            df_train_val[label].to_numpy())

        hash = hashlib.sha256(pd.util.hash_pandas_object(df_train_train,
                                                         index=True).values).hexdigest() + '_gbert-large_' + label

        training_args = TrainingArguments(f'gbert_trainer',
                                          no_cuda=False,
                                          metric_for_best_model='F1',
                                          load_best_model_at_end=True,
                                          num_train_epochs=10,
                                          eval_steps=40,
                                          per_device_train_batch_size=24,
                                          evaluation_strategy='steps',
                                          seed=k,
                                          learning_rate=5e-5,
                                          warmup_ratio=0.3)

        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir + hash,
                                                                       local_files_only=True,
                                                                       num_labels=2)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset_train_train,
                eval_dataset=dataset_train_val,
                compute_metrics=compute_metrics_singlelabel,
            )
        except EnvironmentError:
            set_seed(training_args.seed)
            model = AutoModelForSequenceClassification.from_pretrained(get_hugging_face_name('gbert'),
                                                                       num_labels=2)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset_train_train,
                eval_dataset=dataset_train_val,
                compute_metrics=compute_metrics_singlelabel,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            trainer.train()
            model.save_pretrained(model_dir + hash)

        logits = trainer.predict(dataset_test).predictions
        softmax = torch.nn.Softmax(dim=1)
        pred = np.array([softmax(torch.tensor(logits)).detach().cpu().numpy()])
        if len(predictions) == 0:
            predictions = pred
        else:
            predictions = np.append(predictions, pred, axis=0)

    return predictions


def calc_f1_score_germeval(ly_true, ly_pred):
    macro_f1 = 0
    if len(ly_true.shape) == 1:
        ly_true = ly_true[:, np.newaxis]
        ly_pred = ly_pred[:, np.newaxis]
    for i in range(ly_true.shape[1]):
        report = classification_report(ly_true[:, i], ly_pred[:, i], output_dict=True)
        precision_score = report['macro avg']['precision']
        recall_score = report['macro avg']['recall']
        lf1_score = 0
        if precision_score + recall_score > 0:
            lf1_score = 2 * precision_score * recall_score / (precision_score + recall_score)
        macro_f1 += lf1_score
    return macro_f1 / ly_true.shape[1]


if __name__ == '__main__':
    bootstrap_size = 1000
    max_size = 60
    pool_size = 100
    thresholds = np.linspace(0, 1, 41)

    df = load_dataset()
    df = clean_up_comments(df)
    mean_scores = np.empty((4, 5, 0)).tolist()
    std_scores = np.empty((4, 5, 0)).tolist()
    for fold, (train_index, test_index) in enumerate(KFold().split(df)):
        df_train = df.loc[train_index]
        df_test = df.loc[test_index]
        y_true = df_test[["toxic", "engaging", "fact"]].to_numpy()

#        gelectra50, gelectra50_thresholds = get_gelectra_multilabel(df_train, df_test, pool_size)
#        gbert50, gbert50_thresholds = get_gbert_multilabel(df_train, df_test, pool_size)
#        gelectra25 = gelectra50[:pool_size // 2]
#        gbert25 = gbert50[:pool_size // 2]
        gelectra25_toxic = get_gelectra_singlelabel(df_train, df_test, pool_size // 2, 'toxic')
        gbert25_toxic = get_gbert_singlelabel(df_train, df_test, pool_size // 2, 'toxic')
        gelectra25_engaging = get_gelectra_singlelabel(df_train, df_test, pool_size // 2, 'engaging')
        gbert25_engaging = get_gbert_singlelabel(df_train, df_test, pool_size // 2, 'engaging')
        gelectra25_fact = get_gelectra_singlelabel(df_train, df_test, pool_size // 2, 'fact')
        gbert25_fact = get_gbert_singlelabel(df_train, df_test, pool_size // 2, 'fact')

#        for i in range(1, max_size + 1):
#            np.random.seed(i)
#            idx = np.random.choice(len(gelectra50), size=(bootstrap_size, i))
#            idx_mapped = np.array([np.array([gelectra50[k] for k in j]) for j in idx])
#            ensembles = np.array([np.sum(j, axis=0) / len(j) for j in idx_mapped])
#            optimal_thresholds = [thresholds[np.argmax(np.mean(gelectra50_thresholds[j], axis=0), axis=0)] for j in idx]
#            f1s = np.array(
#                [calc_f1_score_germeval(y_true, (e >= t) * 1) for e, t in zip(ensembles, optimal_thresholds)])
#            mean_scores[0][fold].append(np.mean(f1s))
#            std_scores[0][fold].append(np.std(f1s))

#        for i in range(2, max_size + 1, 2):
#            np.random.seed(i)
#            idx_el = np.random.choice(len(gelectra25), size=(bootstrap_size, int(i / 2)))
#            idx_mapped_el = np.array([np.array([gelectra25[k] for k in j]) for j in idx_el])
#            idx_be = np.random.choice(len(gbert25), size=(bootstrap_size, int(i / 2)))
#            idx_mapped_be = np.array([np.array([gbert25[k] for k in j]) for j in idx_be])
#            idx_mapped = np.append(idx_mapped_el, idx_mapped_be, axis=1)
#            ensembles = np.array([np.sum(j, axis=0) / len(j) for j in idx_mapped])
#            combined_thresholds = np.append([gelectra50_thresholds[j] for j in idx_el],
#                                            [gbert50_thresholds[j] for j in idx_be], axis=1)
#            optimal_thresholds = [thresholds[np.argmax(np.mean(t, axis=0), axis=0)] for t in combined_thresholds]
#            f1s = np.array(
#                [calc_f1_score_germeval(y_true, (e >= t) * 1) for e, t in zip(ensembles, optimal_thresholds)])
#            mean_scores[1][fold].append(np.mean(f1s))
#            std_scores[1][fold].append(np.std(f1s))

#        for i in range(1, max_size + 1):
#            np.random.seed(i)
#            idx = np.random.choice(len(gbert50), size=(bootstrap_size, i))
#            idx_mapped = np.array([np.array([gbert50[k] for k in j]) for j in idx])
#            ensembles = np.array([np.sum(j, axis=0) / len(j) for j in idx_mapped])
#            optimal_thresholds = [thresholds[np.argmax(np.mean(gbert50_thresholds[j], axis=0), axis=0)] for j in idx]
#            f1s = np.array(
#                [calc_f1_score_germeval(y_true, (e >= t) * 1) for e, t in zip(ensembles, optimal_thresholds)])
#            mean_scores[2][fold].append(np.mean(f1s))
#            std_scores[2][fold].append(np.std(f1s))

        for i in range(2, max_size + 1, 2):
            np.random.seed(i)
            idx_el_to = np.random.choice(len(gelectra25_toxic), size=(bootstrap_size, int(i / 2)))
            idx_mapped_el_to = np.array([np.array([gelectra25_toxic[k] for k in j]) for j in idx_el_to])
            idx_be_to = np.random.choice(len(gbert25_toxic), size=(bootstrap_size, int(i / 2)))
            idx_mapped_be_to = np.array([np.array([gbert25_toxic[k] for k in j]) for j in idx_be_to])
            idx_mapped_to = np.append(idx_mapped_el_to, idx_mapped_be_to, axis=1)
            ensembles_to = np.array(
                [np.argmax(np.sum(j, axis=0) / len(j), axis=1).reshape((len(y_true), 1)) for j in idx_mapped_to])

            idx_el_en = np.random.choice(len(gelectra25_engaging), size=(bootstrap_size, int(i / 2)))
            idx_mapped_el_en = np.array([np.array([gelectra25_engaging[k] for k in j]) for j in idx_el_en])
            idx_be_en = np.random.choice(len(gbert25_engaging), size=(bootstrap_size, int(i / 2)))
            idx_mapped_be_en = np.array([np.array([gbert25_engaging[k] for k in j]) for j in idx_be_en])
            idx_mapped_en = np.append(idx_mapped_el_en, idx_mapped_be_en, axis=1)
            ensembles_en = np.array(
                [np.argmax(np.sum(j, axis=0) / len(j), axis=1).reshape((len(y_true), 1)) for j in idx_mapped_en])

            idx_el_fa = np.random.choice(len(gelectra25_fact), size=(bootstrap_size, int(i / 2)))
            idx_mapped_el_fa = np.array([np.array([gelectra25_fact[k] for k in j]) for j in idx_el_fa])
            idx_be_fa = np.random.choice(len(gbert25_fact), size=(bootstrap_size, int(i / 2)))
            idx_mapped_be_fa = np.array([np.array([gbert25_fact[k] for k in j]) for j in idx_be_fa])
            idx_mapped_fa = np.append(idx_mapped_el_fa, idx_mapped_be_fa, axis=1)
            ensembles_fa = np.array(
                [np.argmax(np.sum(j, axis=0) / len(j), axis=1).reshape((len(y_true), 1)) for j in idx_mapped_fa])

            ensembles = np.append(np.append(ensembles_to, ensembles_en, axis=2), ensembles_fa, axis=2)

            f1s = np.array([calc_f1_score_germeval(y_true, j) for j in ensembles])
            mean_scores[3][fold].append(np.mean(f1s))
            std_scores[3][fold].append(np.std(f1s))

    with open('scores/scores_b.txt', 'w') as f:
        f.write('setups:\n')
        f.write('1. GELECTRA\n')
        f.write('2. GELECTRA + GBERT\n')
        f.write('3. GBERT\n')
        f.write('4. GELECTRA + GBERT single-label\n\n')
        f.write('mean scores:\n')
        f.write(str(mean_scores) + '\n\n')
        f.write('std scores:\n')
        f.write(str(std_scores) + '\n')
