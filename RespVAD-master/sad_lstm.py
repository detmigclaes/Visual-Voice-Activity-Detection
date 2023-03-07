import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from pathlib import Path
from glob import glob
from natsort import natsorted
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, roc_auc_score, precision_recall_curve, \
    auc, accuracy_score
import plotly.graph_objects as go
import math

os.environ['KERAS_BACKEND'] = 'tensorflow'
WINDOW_SIZE = 100
PROJECT_ROOT = Path.cwd()
ANNOTATED_DATA_DIR = PROJECT_ROOT / f'Final Data/{WINDOW_SIZE}/annotated_data'
NO_PROCESSED_DATA_DIR = PROJECT_ROOT / f'Final Data/{WINDOW_SIZE}/non_overlapped'
O_PROCESSED_DATA_DIR = PROJECT_ROOT / f'Final Data/{WINDOW_SIZE}/overlapped'
SEQ_LEN = 1
BATCH_SIZE = 256
EPOCH = 105
SHUFFLE_FLAG = True


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def visualize(h, train_key, val_key, title, y_label, x_label, dir_name):
    plt.close('all')
    plt.plot(h.history[train_key])
    # plt.plot(h.history[val_key])
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    # plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(NO_PROCESSED_DATA_DIR / f'{dir_name}/lstm_sad_model_{train_key}.png')


def load_non_overlapped_train_data(base_dir):
    train_data = np.load(str(NO_PROCESSED_DATA_DIR / base_dir) + '/non_overlapping_seq_train_data.npy')
    x, y = train_data[:, :WINDOW_SIZE].reshape(-1, WINDOW_SIZE, SEQ_LEN), train_data[:, WINDOW_SIZE:].reshape(-1,
                                                                                                              WINDOW_SIZE,
                                                                                                              1)
    cw = {0: np.sum(y) / np.prod(y.shape), 1: 1 - np.sum(y) / np.prod(y.shape)}
    return x, y, cw


def load_non_overlapped_test_data(base_dir):
    test_data = np.load(str(NO_PROCESSED_DATA_DIR / base_dir) + '/non_overlapping_seq_test_data.npy')
    x, y = test_data[:, :WINDOW_SIZE].reshape(-1, WINDOW_SIZE, SEQ_LEN), test_data[:, WINDOW_SIZE:].reshape(-1,
                                                                                                            WINDOW_SIZE,
                                                                                                            1)
    cw = {0: np.sum(y) / np.prod(y.shape), 1: 1 - np.sum(y) / np.prod(y.shape)}
    return x, y, cw


def load_overlapped_train_data(base_dir):
    train_data = np.load(str(O_PROCESSED_DATA_DIR / base_dir) + '/overlapping_seq_train_data.npy')
    x, y = train_data[:, :WINDOW_SIZE].reshape(-1, WINDOW_SIZE, SEQ_LEN), train_data[:, WINDOW_SIZE:].reshape(-1,
                                                                                                              WINDOW_SIZE,
                                                                                                              1)
    cw = {0: np.sum(y) / np.prod(y.shape), 1: 1 - np.sum(y) / np.prod(y.shape)}
    return x, y, cw


def load_overlapped_test_data(base_dir):
    test_data_list = natsorted(glob(str(O_PROCESSED_DATA_DIR / base_dir) + '/test/*.npy', recursive=True))
    print(test_data_list)
    test_x_list = []
    test_y_list = []
    for i, data in enumerate(test_data_list):
        test_x_list.append(np.load(data)[:, :WINDOW_SIZE].reshape(-1, WINDOW_SIZE, SEQ_LEN))
        test_y_list.append(np.load(data)[:, WINDOW_SIZE:].reshape(-1, WINDOW_SIZE, 1))
    return test_x_list, test_y_list


def custom_loss(cw):
    def modified_bce_loss(target, output):
        weighted_bce_loss = -(cw[1] * target * K.log(output) + cw[0] * (1.0 - target) * K.log(1.0 - output))
        return weighted_bce_loss

    return modified_bce_loss


def build_model():
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential(name='seq-model')
    model.add(tf.keras.layers.Input(batch_shape=(None, WINDOW_SIZE, SEQ_LEN), name="input"))
    model.add(tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(units=128, return_sequences=True),
                                            merge_mode='concat'))
    model.add(tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(units=128, return_sequences=True),
                                            merge_mode='concat'))

    model.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(32, kernel_initializer=tf.keras.initializers.glorot_normal(),
                              activation=tf.nn.relu)))

    model.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.glorot_normal(),
                              activation='sigmoid')))
    return model


def compile_train_model(model, bs, n_epoch, dir_name, lr, OVERLAP_FLAG):
    if OVERLAP_FLAG:
        train_x, train_y, train_cw = load_overlapped_train_data(dir_name)
    else:
        train_x, train_y, train_cw = load_non_overlapped_train_data(dir_name)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=custom_loss(train_cw),
                  metrics=['acc', f1_m, precision_m, recall_m])
    model.summary()
    history = model.fit(x=train_x, y=train_y, batch_size=bs, shuffle=SHUFFLE_FLAG,
                        epochs=n_epoch
                        )
    return history


def lstm_sad_non_overlapped(bs, n_epoch, dir_name, exp_no):
    sad_model = build_model()
    h = compile_train_model(sad_model, bs, n_epoch, dir_name, 3e-4, False)

    sad_model.summary()
    # summarize history for accuracy
    visualize(h, 'acc', 'val_acc', 'Model Accuracy', 'Accuracy', 'epoch', dir_name)
    visualize(h, 'loss', 'val_loss', 'Model Loss', 'Loss', 'epoch', dir_name)
    visualize(h, 'f1_m', 'val_f1_m', 'Model F1', 'F1', 'epoch', dir_name)
    visualize(h, 'precision_m', 'val_precision_m', 'Model Precision', 'Accuracy', 'epoch', dir_name)
    visualize(h, 'recall_m', 'val_recall_m', 'Model Recall', 'Recall', 'epoch', dir_name)

    test_x, test_y, test_cw = load_non_overlapped_test_data(dir_name)

    scores = sad_model.evaluate(x=test_x, y=test_y, batch_size=bs)
    metrics = sad_model.metrics_names

    predicted_y = sad_model.predict(test_x)
    predicted_y = predicted_y.reshape((-1, 1))
    test_y = test_y.reshape((-1, 1))

    roc_auc = roc_auc_score(test_y, predicted_y)

    predicted_y = predicted_y.reshape((-1,))
    predicted_y[predicted_y <= 0.5] = 0
    predicted_y[predicted_y > 0.5] = 1

    f1_s = f1_score(test_y.reshape(-1, 1), predicted_y.reshape(-1, 1))
    p_s = precision_score(test_y.reshape(-1, 1), predicted_y.reshape(-1, 1))
    r_s = recall_score(test_y.reshape(-1, 1), predicted_y.reshape(-1, 1))
    acc_s = accuracy_score(test_y.reshape(-1, 1), predicted_y.reshape(-1, 1))

    # plt.close('all')
    # y1 = test_x.reshape((-1, ))
    # x = np.arange(len(y1))
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Signal'))
    # fig.add_trace(go.Scatter(x=x, y=0.7 * test_y.reshape((-1,)), mode='lines', name='GT'))
    # fig.add_trace(go.Scatter(x=x, y=0.5 * predicted_y, mode='lines', name='Pred'))
    # fig.show()

    print(
        f'ROC AUC: {roc_auc}, '
        f'Acc: {acc_s}, '
        f'F1: {f1_s}, '
        f'Precision: {p_s}, '
        f'Recall: {r_s}'
    )

    create_directory(str(NO_PROCESSED_DATA_DIR / dir_name) + '/trained_models/lstm_sad_non_overlapped')
    sad_model.save(str(NO_PROCESSED_DATA_DIR / dir_name) + f'/trained_models/lstm_sad_non_overlapped/model_{exp_no}.h5')

    return roc_auc, acc_s, f1_s, p_s, r_s


def lstm_sad_overlapped(bs, n_epoch, dir_name, exp_no):
    sad_model = build_model()
    h = compile_train_model(sad_model, bs, n_epoch, dir_name, 1e-4, True)

    sad_model.summary()
    # summarize history for accuracy
    # visualize(h, 'acc', 'val_acc', 'Model Accuracy', 'Accuracy', 'epoch', dir_name)
    # visualize(h, 'loss', 'val_loss', 'Model Loss', 'Loss', 'epoch', dir_name)
    # visualize(h, 'f1_m', 'val_f1_m', 'Model F1', 'F1', 'epoch', dir_name)
    # visualize(h, 'precision_m', 'val_precision_m', 'Model Precision', 'Accuracy', 'epoch', dir_name)
    # visualize(h, 'recall_m', 'val_recall_m', 'Model Recall', 'Recall', 'epoch', dir_name)

    if dir_name == 'data1':
        test_speakers = [i for i in range(1, 11, 1)]
    elif dir_name == 'data2':
        test_speakers = [i for i in range(11, 21, 1)]
    elif dir_name == 'data3':
        test_speakers = [i for i in range(21, 31, 1)]
    elif dir_name == 'data4':
        test_speakers = [i for i in range(31, 41, 1)]

    test_x, test_y = load_overlapped_test_data(dir_name)

    PREDICTION_DIR = O_PROCESSED_DATA_DIR / f'{dir_name}/predicted/sad_lstm/'
    create_directory(PREDICTION_DIR)

    for i, speaker in enumerate(test_speakers):
        sig = np.ndarray((test_x[i].shape[0], 2 * test_x[i].shape[1]))
        predicted_y = sad_model.predict(test_x[i])
        sig[:, 0:test_x[i].shape[1]] = test_x[i].reshape((test_x[i].shape[0], test_x[i].shape[1]))
        sig[:, test_x[i].shape[1]:] = predicted_y.reshape((test_x[i].shape[0], test_x[i].shape[1]))

        np.save(str(PREDICTION_DIR / f'prediction_sp_{speaker}.npy'), sig)
        print(f'prediction_sp_{speaker}')

        with open(str(ANNOTATED_DATA_DIR / f'pd{speaker}.csv')) as fp:
            df = pd.read_csv(fp, header=None)

        prediction_sp = np.load(str(PREDICTION_DIR / f'prediction_sp_{speaker}.npy'))[:, test_x[i].shape[1]:]

        print(prediction_sp.shape, len(df))

        binary_prediction = []
        predicted_prob = []

        for j in range(len(df)):
            if j < WINDOW_SIZE - 1:
                n_occur = j + 1
                index = n_occur - 1
                prob = 0
                for k in range(0, n_occur):
                    prob = prob + prediction_sp[k][index]
                    index = index - 1
                prob = prob / n_occur
            elif j > len(df) - WINDOW_SIZE:
                n_occur = len(df) - j
                index = WINDOW_SIZE - 1
                prob = 0
                for k in range(0, n_occur):
                    prob = prob + prediction_sp[j - WINDOW_SIZE + k][index]
                    index = index - 1
                prob = prob / n_occur
            else:
                n_occur = WINDOW_SIZE
                index = WINDOW_SIZE - 1
                prob = 0
                for k in range(0, n_occur):
                    prob = prob + prediction_sp[j - WINDOW_SIZE + k][index]
                    index = index - 1
                prob = prob / n_occur

            label = 1 if prob >= 0.5 else 0
            binary_prediction.append(label)
            predicted_prob.append(prob)

        df['Predicted Label'] = binary_prediction
        df['Predicted Probability'] = predicted_prob
        df.to_csv(str(PREDICTION_DIR / f'sp{speaker}_prediction.csv'), index=False)

        roc_auc = roc_auc_score(df[1].values.reshape(-1, 1), df["Predicted Probability"].values.reshape(-1, 1))

        print(
            f'Speaker ID: {speaker}, '
            f'ROC AUC: {roc_auc}, '
            f'F1: {f1_score(df[1].values.reshape(-1, 1), df["Predicted Label"].values.reshape(-1, 1))}, '
            f'Precision: {precision_score(df[1].values.reshape(-1, 1), df["Predicted Label"].values.reshape(-1, 1))}, '
            f'Recall: {recall_score(df[1].values.reshape(-1, 1), df["Predicted Label"].values.reshape(-1, 1))}'
        )

    for n, speaker in enumerate(test_speakers):
        with open(str(PREDICTION_DIR / f'sp{speaker}_prediction.csv')) as f:
            if n == 0:
                df = pd.read_csv(f)
            else:
                df = pd.concat([df, pd.read_csv(f)], ignore_index=True)

    roc_auc = roc_auc_score(df[df.columns[1]].values.reshape(-1, 1), df["Predicted Probability"].values.reshape(-1, 1))
    f1_s = f1_score(df[df.columns[1]].values.reshape(-1, 1), df["Predicted Label"].values.reshape(-1, 1))
    p_s = precision_score(df[df.columns[1]].values.reshape(-1, 1), df["Predicted Label"].values.reshape(-1, 1))
    r_s = recall_score(df[df.columns[1]].values.reshape(-1, 1), df["Predicted Label"].values.reshape(-1, 1))
    acc_s = accuracy_score(df[df.columns[1]].values.reshape(-1, 1), df["Predicted Label"].values.reshape(-1, 1))

    print('All Test Speakers')
    print(
        f'ROC AUC: {roc_auc}, '
        f'Acc: {acc_s}, '
        f'F1: {f1_s}, '
        f'Precision: {p_s}, '
        f'Recall: {r_s}'
    )

    create_directory(str(O_PROCESSED_DATA_DIR / dir_name) + '/trained_models/lstm_sad_overlapped')
    sad_model.save(str(O_PROCESSED_DATA_DIR / dir_name) + f'/trained_models/lstm_sad_overlapped/model_{exp_no}.h5')

    return roc_auc, acc_s, f1_s, p_s, r_s


num_exp = 10
for dataset in ['data1', 'data2', 'data3', 'data4']:
    df = pd.DataFrame({'Exp. No.': [i for i in range(num_exp)]})
    roc_auc_buf = []
    acc_buf = []
    f1_buf = []
    p_buf = []
    r_buf = []
    for step in range(num_exp):
        v, w, x, y, z = lstm_sad_non_overlapped(bs=BATCH_SIZE, n_epoch=EPOCH,
                                                dir_name=dataset, exp_no=step)
        roc_auc_buf.append(v)
        acc_buf.append(w)
        f1_buf.append(x)
        p_buf.append(y)
        r_buf.append(z)

    print(roc_auc_buf)
    print(acc_buf)
    print(f1_buf)
    print(p_buf)
    print(r_buf)

    df['ROC AUC'] = roc_auc_buf
    df['Accuracy'] = acc_buf
    df['F1'] = f1_buf
    df['Precision'] = p_buf
    df['Recall'] = r_buf
    df.to_csv(str(NO_PROCESSED_DATA_DIR / dataset) + '/lstm_sad_non_overlapped.csv')

num_exp = 10
for dataset in ['data1', 'data2', 'data3', 'data4']:
    df = pd.DataFrame({'Exp. No.': [i for i in range(num_exp)]})
    roc_auc_buf = []
    acc_buf = []
    f1_buf = []
    p_buf = []
    r_buf = []
    for step in range(num_exp):
        v, w, x, y, z = lstm_sad_overlapped(bs=BATCH_SIZE, n_epoch=5,
                                            dir_name=dataset, exp_no=step)
        roc_auc_buf.append(v)
        acc_buf.append(w)
        f1_buf.append(x)
        p_buf.append(y)
        r_buf.append(z)

    print(roc_auc_buf)
    print(acc_buf)
    print(f1_buf)
    print(p_buf)
    print(r_buf)

    df['ROC AUC'] = roc_auc_buf
    df['Accuracy'] = acc_buf
    df['F1'] = f1_buf
    df['Precision'] = p_buf
    df['Recall'] = r_buf
    df.to_csv(str(O_PROCESSED_DATA_DIR / dataset) + '/lstm_sad_overlapped.csv')
