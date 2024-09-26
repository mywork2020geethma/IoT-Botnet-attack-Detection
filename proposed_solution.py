#import required libraries

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from tensorflow import keras
import keras_tuner
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout,Activation,GRU,AveragePooling1D ,BatchNormalization, Reshape,Conv1D, MaxPooling1D,GlobalMaxPooling1D,Embedding

# create dataframes

device_info_dataframe=pd.read_csv("/kaggle/input/nbaiot-dataset/device_info.csv")
device_info_dataframe

#Extract features

features_data=pd.read_csv("/kaggle/input/nbaiot-dataset/features.csv")
features_data

# Create dataframes for different attack types

benign_df = pd.read_csv('../input/nbaiot-dataset/1.benign.csv')
m_a_df = pd.read_csv('../input/nbaiot-dataset/1.mirai.ack.csv')
m_sc_df = pd.read_csv('../input/nbaiot-dataset/1.mirai.scan.csv')
m_sy_df = pd.read_csv('../input/nbaiot-dataset/1.mirai.syn.csv')
m_u_df = pd.read_csv('../input/nbaiot-dataset/1.mirai.udp.csv')
m_u_p_df = pd.read_csv('../input/nbaiot-dataset/1.mirai.udpplain.csv')
g_c_df = pd.read_csv('../input/nbaiot-dataset/1.gafgyt.combo.csv')
g_j_df = pd.read_csv('../input/nbaiot-dataset/1.gafgyt.junk.csv')
g_s_df = pd.read_csv('../input/nbaiot-dataset/1.gafgyt.scan.csv')
g_t_df = pd.read_csv('../input/nbaiot-dataset/1.gafgyt.tcp.csv')
g_u_df = pd.read_csv('../input/nbaiot-dataset/1.gafgyt.udp.csv')
benign_df1 = pd.read_csv('../input/nbaiot-dataset/2.benign.csv')
m_a_df1 = pd.read_csv('../input/nbaiot-dataset/2.mirai.ack.csv')
m_sc_df1 = pd.read_csv('../input/nbaiot-dataset/2.mirai.scan.csv')
m_sy_df1 = pd.read_csv('../input/nbaiot-dataset/2.mirai.syn.csv')
m_u_df1 = pd.read_csv('../input/nbaiot-dataset/2.mirai.udp.csv')
m_u_p_df1 = pd.read_csv('../input/nbaiot-dataset/2.mirai.udpplain.csv')
g_c_df1 = pd.read_csv('../input/nbaiot-dataset/2.gafgyt.combo.csv')
g_j_df1 = pd.read_csv('../input/nbaiot-dataset/2.gafgyt.junk.csv')
g_s_df1 = pd.read_csv('../input/nbaiot-dataset/2.gafgyt.scan.csv')
g_t_df1 = pd.read_csv('../input/nbaiot-dataset/2.gafgyt.tcp.csv')
g_u_df1 = pd.read_csv('../input/nbaiot-dataset/2.gafgyt.udp.csv')

# labeling

benign_df['Attack_Class'] = 'benign     '
m_u_df['Attack_Class']    = 'mirai_udp'
m_a_df['Attack_Class']    = 'mirai_ack'
m_sc_df['Attack_Class']   = 'mirai_scan'
m_sy_df['Attack_Class']   = 'mirai_syn'
m_u_p_df['Attack_Class']  = 'mirai_udpplain'
g_c_df['Attack_Class']    = 'gafgyt_combo'
g_j_df['Attack_Class']    = 'gafgyt_junk'
g_s_df['Attack_Class']    = 'gafgyt_scan'
g_t_df['Attack_Class']    = 'gafgyt_tcp'
g_u_df['Attack_Class']    = 'gafgyt_udp'
benign_df1['Attack_Class'] = 'benign     '
m_u_df1['Attack_Class']    = 'mirai_udp'
m_a_df1['Attack_Class']    = 'mirai_ack'
m_sc_df1['Attack_Class']   = 'mirai_scan'
m_sy_df1['Attack_Class']   = 'mirai_syn'
m_u_p_df1['Attack_Class']  = 'mirai_udpplain'
g_c_df1['Attack_Class']    = 'gafgyt_combo'
g_j_df1['Attack_Class']    = 'gafgyt_junk'
g_s_df1['Attack_Class']    = 'gafgyt_scan'
g_t_df1['Attack_Class']    = 'gafgyt_tcp'
g_u_df1['Attack_Class']    = 'gafgyt_udp'

#concatanate dataframes

df = pd.concat([benign_df,
                m_u_df, m_a_df, m_sc_df,m_sy_df, m_u_p_df,
                g_c_df,g_j_df, g_s_df, g_t_df,g_u_df,
                benign_df1,m_u_df1, m_a_df1, m_sc_df1,m_sy_df1, m_u_p_df1,
                g_c_df1,g_j_df1, g_s_df1, g_t_df1,g_u_df1],
                axis=0, sort=False, ignore_index=True)

### Data Preprocessing ###

# remove duplicates

print(df.duplicated().sum(), "fully duplicate rows to remove")
df.drop_duplicates(inplace=True)
df.reset_index(inplace=True, drop=True)
df.shape

# Display the classes with the number of samples

class_counts = df["Attack_Class"].value_counts()
total_samples = class_counts.sum()

print("Serial\tAttack_Class\t\tNumber of Samples")
print("-----------------------------------")

serial_number = 1
for class_name, count in class_counts.items():
    print(f"{serial_number}\t{class_name}\t\t{count}")
    serial_number += 1
print("-----------------------------------")
print(f"Total Samples:                  {total_samples}")

# Splitting Features and Labels

label_columns = "Attack_Class"
feature_columns = list(df.columns)
feature_columns.remove(label_columns)
X = df[feature_columns]
y = df[label_columns]

# Encode the categorical labels into numerical format

cls_label_encoder = LabelEncoder()
y = cls_label_encoder.fit_transform(y)
unique_values = cls_label_encoder.classes_
print("Unique encoded values:", unique_values)

# Mapping Class Names to Indices
class_indices = dict(zip(unique_values, range(len(unique_values))))

print("Class Indices Mapping:", class_indices)

# Compute class weights

class_weights = {}
for class_name, count in class_counts.items():

    # Calculate weight as the inverse of the frequency of the class
    weight = total_samples / (len(class_counts) * count)
    if class_name in class_indices:
        class_index = class_indices[class_name]
        class_weights[class_index] = weight
    else:
        print(f"Warning: {class_name} not found in class_indices")

# Display class weights

print("Class Weights:")
for class_index, weight in class_weights.items():
    print(f"Class index {class_index} ({unique_values[class_index]}): {weight:.4f}")

# Split into 80% training + validation and 20% test

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=True,
    stratify=y
)

# Split the remaining 80% into 70% training and 30% validation

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25,  # 25% of the 80% (which is 20% of the total dataset)
    shuffle=True,
    stratify=y_train_val
)

# X_train: 70% of the original data
# X_val: 20% of the original data
# X_test: 10% of the original data

# Min-Max Scaling (Normalization)
# scales features to range between 0 and 1

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

# apply the same scaling parameters to X_train , X_test and X_val

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# converting from 2D arrays to 3D arrays with shape (samples, features, 1)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

# One-hot encoded

n_classes = len(np.unique(y))
y_train = to_categorical(y_train, num_classes=n_classes)
y_test = to_categorical(y_test, num_classes=n_classes)
y_val = to_categorical(y_val, num_classes=n_classes)

# extracts the shape of the input features for the model

input_shape = X_train.shape[1:]
print(input_shape)

### Proposed model ###

# Model Architecture

input_1 = Input (X_train.shape[1:],name='Inputlayer')

x = Conv1D(64, kernel_size=3, padding = 'same')(input_1)
x = MaxPool1D(3, strides = 2, padding = 'same')(x)
x = Conv1D(128,kernel_size=3,strides=1, padding='same',activation='relu')(x)
x = Conv1D(128, 3,strides=1, padding='same',activation='relu')(x)
x = MaxPooling1D(3, strides=1, padding='same')(x)
x = Conv1D(128, 2,strides=1, padding='same',activation='relu')(x)
x = Conv1D(128, 3,strides=1, padding='same',activation='relu')(x)
x = MaxPooling1D(3, strides=2, padding='same')(x)
x = Conv1D(128, 2,strides=1, padding='same',activation='relu')(x)
x = Conv1D(128, 3,strides=1, padding='same',activation='relu')(x)
x = MaxPooling1D(3, strides=2, padding='same')(x)
x = GRU(128, return_sequences=True,activation='relu')(x)
x = Dropout(0.4)(x)
x = GRU(32)(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output_layer = Dense(11, activation='softmax')(x)

model_cnn_gru = Model(inputs=input_1, outputs=output_layer)
model_cnn_gru.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model_cnn_gru.summary()

checkpoint_filepath = 'checkpoint.keras'

# Checkpoints

model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True,  # Save only the best model
    mode='min',           # Save the model with the minimum validation loss
    verbose=1
)

# Early stopping

stop_early = EarlyStopping(monitor='val_loss', patience=3)

# optimizer

optimizer = Adam(learning_rate=0.0001)

model_cnn_gru.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# learning rate scheduling

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1
)

# load checkpoint

model_cnn_gru = load_model('checkpoint.keras')

# resume training from checkpoint

history_cnn_gru = model_cnn_gru.fit(
     X_train,
     y_train,
     epochs=2,
     batch_size=32,
     validation_data=(X_val, y_val),
     verbose=1,
     callbacks=[stop_early, model_checkpoint, lr_scheduler],
     class_weight=class_weights
)

### performance evaluation ###

# Make predictions
y_pred_prob = model_cnn_gru.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate F1 Score
f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'weighted' for multi-class problems
print('F1 Score: ', f1)

# Calculate Precision
precision = precision_score(y_true, y_pred, average='weighted')  # Use 'weighted' for multi-class problems
print('Precision: ', precision)

# Calculate Recall
recall = recall_score(y_true, y_pred, average='weighted')  # Use 'weighted' for multi-class problems
print('Recall: ', recall)

# Calculate Accuracy
accuracy = accuracy_score(y_true, y_pred)
print('Accuracy: ', accuracy)