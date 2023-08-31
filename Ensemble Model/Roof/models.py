''''
Author: Daotan Yang

Date: 31/08/2023

Description: This file holds the model classe and the function used to interpolate the output on the validation set and output the results.

'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# DAE1, can handle classification and regression tasks at the same time, using leaky_relu activation function
class DAE(tf.keras.Model):
    def __init__(self, original_dim, categorical_dim, continuous_dim,  name="dae", **kwargs):
        super(DAE, self).__init__(name=name, **kwargs)
        self.encoder = tf.keras.Sequential([
            layers.Dense(original_dim, activation = 'leaky_relu'),
            layers.Dense(16, activation = 'leaky_relu'),
            layers.Dense(32, activation = 'leaky_relu'),
            layers.Dense(64, activation = 'leaky_relu'),
            layers.Dense(128, activation = 'leaky_relu'),
            # layers.Dense(256, activation = 'leaky_relu'),
        ])

        # An encoder that deals with continuous variables,
        # its goal is to restore the features of the fourth column and beyond
        self.decoder_continuous = tf.keras.Sequential([
            # layers.Dense(256, activation = 'leaky_relu'),
            layers.Dense(128, activation = 'leaky_relu'),
            layers.Dense(64, activation = 'leaky_relu'),
            layers.Dense(32, activation = 'leaky_relu'),
            layers.Dense(16, activation = 'leaky_relu'),
            layers.Dropout(0.2),
            layers.Dense(continuous_dim, activation = 'leaky_relu')
        ])

        # Decoder1 that deals with categorical variables,
        # its goal is to restore the target column: Quality_insulation_lower_floor
        self.decoder_categorical_1 = tf.keras.Sequential([
            # layers.Dense(256, activation = 'leaky_relu'),
            layers.Dense(128, activation = 'leaky_relu'),
            layers.Dense(64, activation = 'leaky_relu'),
            layers.Dense(32, activation = 'leaky_relu'),
            layers.Dense(16, activation = 'leaky_relu'),
            layers.Dense(8, activation = 'leaky_relu'),
            # layers.Dropout(0.2),
            layers.Dense(4, activation = 'softmax') # 4 classes for the first categorical variable
        ])

        # Decoder2 that deals with categorical variables,
        # its goal is to restore the second column: Quality_insulation_envelope
        self.decoder_categorical_2 = tf.keras.Sequential([
            # layers.Dense(256, activation = 'leaky_relu'),
            layers.Dense(128, activation = 'leaky_relu'),
            layers.Dense(64, activation = 'leaky_relu'),
            layers.Dense(32, activation = 'leaky_relu'),
            layers.Dense(16, activation = 'leaky_relu'),
            layers.Dense(8, activation = 'leaky_relu'),
            # layers.Dropout(0.2),
            layers.Dense(4, activation = 'softmax') # 4 classes for the second categorical variable
        ])

        # Decoder3 that deals with categorical variables,
        # its goal is to restore the third column:Roof_insulation_(0/1)
        self.decoder_categorical_3 = tf.keras.Sequential([
            # layers.Dense(256, activation = 'leaky_relu'),
            layers.Dense(128, activation = 'leaky_relu'),
            layers.Dense(64, activation = 'leaky_relu'),
            layers.Dense(32, activation = 'leaky_relu'),
            layers.Dense(16, activation = 'leaky_relu'),
            layers.Dense(4, activation = 'leaky_relu'),
            # layers.Dropout(0.2),
            layers.Dense(2, activation = 'softmax') # 2 classes for the third categorical variable
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded_continuous = self.decoder_continuous(encoded)
        decoded_categorical_1 = self.decoder_categorical_1(encoded)
        decoded_categorical_2 = self.decoder_categorical_2(encoded)
        decoded_categorical_3 = self.decoder_categorical_3(encoded)
        decoded = tf.concat([decoded_categorical_1, decoded_categorical_2, decoded_categorical_3, decoded_continuous], axis=1)
        return decoded


# DAE2, can handle classification and regression tasks at the same time, using swish activation function
class DAE2(tf.keras.Model):
    def __init__(self, original_dim, categorical_dim, continuous_dim, name="dae", **kwargs):
        super(DAE2, self).__init__(name=name, **kwargs)
        self.encoder = tf.keras.Sequential([
            layers.Dense(original_dim, activation = 'swish'),
            layers.Dense(16, activation = 'swish'),
            layers.Dense(32, activation = 'swish'),
            layers.Dense(64, activation = 'swish'),
            layers.Dense(128, activation = 'swish'),
            # layers.Dense(256, activation = 'swish'),
        ])

        # An encoder that deals with continuous variables,
        # its goal is to restore the features of the fourth column and beyond
        self.decoder_continuous = tf.keras.Sequential([
            # layers.Dense(256, activation = 'swish'),
            layers.Dense(128, activation = 'swish'),
            layers.Dense(64, activation = 'swish'),
            layers.Dense(32, activation = 'swish'),
            layers.Dense(16, activation = 'swish'),
            layers.Dropout(0.2),
            layers.Dense(continuous_dim, activation = 'swish')
        ])

        # Decoder1 that deals with categorical variables,
        # its goal is to restore the target column: Quality_insulation_lower_floor
        self.decoder_categorical_1 = tf.keras.Sequential([
            # layers.Dense(256, activation = 'swish'),
            layers.Dense(128, activation = 'swish'),
            layers.Dense(64, activation = 'swish'),
            layers.Dense(32, activation = 'swish'),
            layers.Dense(16, activation = 'swish'),
            layers.Dense(8, activation = 'swish'),
            # layers.Dropout(0.2),
            layers.Dense(4, activation = 'softmax') # 4 classes for the first categorical variable
        ])

        # Decoder2 that deals with categorical variables,
        # its goal is to restore the second column: Quality_insulation_envelope
        self.decoder_categorical_2 = tf.keras.Sequential([
            # layers.Dense(256, activation = 'swish'),
            layers.Dense(128, activation = 'swish'),
            layers.Dense(64, activation = 'swish'),
            layers.Dense(32, activation = 'swish'),
            layers.Dense(16, activation = 'swish'),
            layers.Dense(8, activation = 'swish'),
            # layers.Dropout(0.2),
            layers.Dense(4, activation = 'softmax') # 4 classes for the second categorical variable
        ])

        # Decoder3 that deals with categorical variables,
        # its goal is to restore the third column:Roof_insulation_(0/1)
        self.decoder_categorical_3 = tf.keras.Sequential([
            # layers.Dense(256, activation = 'swish'),
            layers.Dense(128, activation = 'swish'),
            layers.Dense(64, activation = 'swish'),
            layers.Dense(32, activation = 'swish'),
            layers.Dense(16, activation = 'swish'),
            layers.Dense(4, activation = 'swish'),
            # layers.Dropout(0.2),
            layers.Dense(2, activation = 'softmax') # 2 classes for the third categorical variable
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded_continuous = self.decoder_continuous(encoded)
        decoded_categorical_1 = self.decoder_categorical_1(encoded)
        decoded_categorical_2 = self.decoder_categorical_2(encoded)
        decoded_categorical_3 = self.decoder_categorical_3(encoded)
        decoded = tf.concat([decoded_categorical_1, decoded_categorical_2, decoded_categorical_3, decoded_continuous], axis=1)
        return decoded


# DAE3, can handle classification and regression tasks at the same time, using elu activation function
class DAE3(tf.keras.Model):
    def __init__(self, original_dim, categorical_dim, continuous_dim, name="dae", **kwargs):
        super(DAE3, self).__init__(name=name, **kwargs)
        self.encoder = tf.keras.Sequential([
            layers.Dense(original_dim, activation = 'elu'),
            layers.Dense(16, activation = 'elu'),
            layers.Dense(32, activation = 'elu'),
            layers.Dense(64, activation = 'elu'),
            layers.Dense(128, activation = 'elu'),
            # layers.Dense(256, activation = 'elu'),
        ])

        # An encoder that deals with continuous variables,
        # its goal is to restore the features of the fourth column and beyond
        self.decoder_continuous = tf.keras.Sequential([
            # layers.Dense(256, activation = 'elu'),
            layers.Dense(128, activation = 'elu'),
            layers.Dense(64, activation = 'elu'),
            layers.Dense(32, activation = 'elu'),
            layers.Dense(16, activation = 'elu'),
            layers.Dropout(0.2),
            layers.Dense(continuous_dim, activation = 'elu')
        ])

        # Decoder1 that deals with categorical variables,
        # its goal is to restore the target column: Quality_insulation_lower_floor
        self.decoder_categorical_1 = tf.keras.Sequential([
            # layers.Dense(256, activation = 'elu'),
            layers.Dense(128, activation = 'elu'),
            layers.Dense(64, activation = 'elu'),
            layers.Dense(32, activation = 'elu'),
            layers.Dense(16, activation = 'elu'),
            layers.Dense(8, activation = 'elu'),
            # layers.Dropout(0.2),
            layers.Dense(4, activation = 'softmax') # 4 classes for the first categorical variable
        ])

        # Decoder2 that deals with categorical variables,
        # its goal is to restore the second column: Quality_insulation_envelope
        self.decoder_categorical_2 = tf.keras.Sequential([
            # layers.Dense(256, activation = 'swish'),
            layers.Dense(128, activation = 'elu'),
            layers.Dense(64, activation = 'elu'),
            layers.Dense(32, activation = 'elu'),
            layers.Dense(16, activation = 'elu'),
            layers.Dense(8, activation = 'elu'),
            # layers.Dropout(0.2),
            layers.Dense(4, activation = 'softmax') # 4 classes for the second categorical variable
        ])

        # Decoder3 that deals with categorical variables,
        # its goal is to restore the third column:Roof_insulation_(0/1)
        self.decoder_categorical_3 = tf.keras.Sequential([
            # layers.Dense(256, activation = 'swish'),
            layers.Dense(128, activation = 'elu'),
            layers.Dense(64, activation = 'elu'),
            layers.Dense(32, activation = 'elu'),
            layers.Dense(16, activation = 'elu'),
            layers.Dense(4, activation = 'elu'),
            # layers.Dropout(0.2),
            layers.Dense(2, activation = 'softmax') # 2 classes for the third categorical variable
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded_continuous = self.decoder_continuous(encoded)
        decoded_categorical_1 = self.decoder_categorical_1(encoded)
        decoded_categorical_2 = self.decoder_categorical_2(encoded)
        decoded_categorical_3 = self.decoder_categorical_3(encoded)
        decoded = tf.concat([decoded_categorical_1, decoded_categorical_2, decoded_categorical_3, decoded_continuous], axis=1)
        return decoded


# Custom loss function, used to calculate the loss of discrete variables and continuous variables
def custom_loss(y_true, y_pred):
    # the first 3 columns of y_true and y_pred are the categorical variables,
    # and the rest are the continuous variables.
    y_true_categorical_1 = y_true[:, 0]
    y_true_categorical_2 = y_true[:, 1]
    y_true_categorical_3 = y_true[:, 2]

    y_pred_categorical_1 = y_pred[:, :4]
    y_pred_categorical_2 = y_pred[:, 4:8]
    y_pred_categorical_3 = y_pred[:, 8:10]
    y_true_continuous = y_true[:, 3:]
    y_pred_continuous = y_pred[:, 10:]

    # Use cross entropy for the categorical variables.
    categorical_loss_1 = tf.keras.losses.SparseCategoricalCrossentropy()(y_true_categorical_1, y_pred_categorical_1)
    categorical_loss_2 = tf.keras.losses.SparseCategoricalCrossentropy()(y_true_categorical_2, y_pred_categorical_2)
    categorical_loss_3 = tf.keras.losses.SparseCategoricalCrossentropy()(y_true_categorical_3, y_pred_categorical_3)

    # Use mean squared error for the continuous variables.
    continuous_loss = tf.keras.losses.MeanSquaredError()(y_true_continuous, y_pred_continuous)

    return categorical_loss_1 + categorical_loss_2 + categorical_loss_3 + continuous_loss


# The function of this function is to reshape the result output by DAE
def reshape_dae_output(dae_output):
    # the first 3 columns of y_true and y_pred are the categorical variables,
    # and the rest are the continuous variables.
    dae_output_categorical_1 = tf.argmax(dae_output[:, :4], 1)
    dae_output_categorical_2 = tf.argmax(dae_output[:, 4:8], 1)
    dae_output_categorical_3 = tf.argmax(dae_output[:, 8:10], 1)
    dae_output_continuous = dae_output[:, 10:]

    # Cast the categorical outputs to float32 and expand dimensions
    dae_output_categorical_1 = tf.expand_dims(tf.cast(dae_output_categorical_1, tf.float32), 1)
    dae_output_categorical_2 = tf.expand_dims(tf.cast(dae_output_categorical_2, tf.float32), 1)
    dae_output_categorical_3 = tf.expand_dims(tf.cast(dae_output_categorical_3, tf.float32), 1)

    reshaped_output = tf.concat([dae_output_categorical_1, dae_output_categorical_2, dae_output_categorical_3, dae_output_continuous], axis=1)

    return reshaped_output


# The function of this function is to reshape the result output by DAE
# Compared with the above function, it can output an additional mask,
# which will be used to select the value to be re-corrected
def reshape_dae_output_mask(dae_output):
    # Assuming the first 3 columns of y_true and y_pred are the categorical variables,
    # and the rest are the continuous variables.
    dae_output_categorical_1 = tf.argmax(dae_output[:, :4], 1)
    dae_output_categorical_2 = tf.argmax(dae_output[:, 4:8], 1)
    dae_output_categorical_3 = tf.argmax(dae_output[:, 8:10], 1)
    dae_output_continuous = dae_output[:, 10:]
    # print(dae_output[:, :4])

    # Find the maximum probability in dae_output_categorical_1
    max_prob = tf.reduce_max(dae_output[:, :4], axis=1)

    # Create a mask where max_prob is less than 0.4
    mask = tf.cast(tf.math.less(max_prob, 0.6), tf.bool)

    # Expand dimensions of mask
    mask = tf.expand_dims(mask, axis=1)

    # Cast the categorical outputs to float32 and expand dimensions
    dae_output_categorical_1 = tf.expand_dims(tf.cast(dae_output_categorical_1, tf.float32), 1)
    dae_output_categorical_2 = tf.expand_dims(tf.cast(dae_output_categorical_2, tf.float32), 1)
    dae_output_categorical_3 = tf.expand_dims(tf.cast(dae_output_categorical_3, tf.float32), 1)

    reshaped_output = tf.concat([dae_output_categorical_1, dae_output_categorical_2, dae_output_categorical_3, dae_output_continuous], axis=1)

    return reshaped_output, mask


# Interpolation process function. This function is used for non-ensemble methods.
def insert_and_impute(df, df_true, train_df, column_name, fraction, vae, num_iterations = 10):
    # Create a copy of df
    df_copy = df.copy()
    df_copy_np = df_copy.values

    # create a new array with the same shape as x_train_np but with all elements -1
    df_copy_np_minus_one = np.full(df_copy_np.shape, -1)

    # Create a mask, the shape of the mask is the same as x_train_np, p = masking_factor represents the probability of 1 on the mask
    masking_factor = 1 - fraction
    mask = np.random.binomial(n = 1, p = masking_factor, size = df_copy_np.shape)

    # Use the tf.where function to replace the value in x_train_np with -1 according to the mask to get the noise data
    df_copy_np_noisy = tf.where(mask, df_copy_np, df_copy_np_minus_one)
    df_copy_np_noisy = tf.cast(df_copy_np_noisy, tf.float32)

    # print(df_copy_np_noisy)

    # Convert dataframe to input format accepted by vae
    missing_data = tf.Variable(df_copy_np_noisy, dtype = tf.float32, trainable=True)
    true_data = tf.Variable(df_true.values.astype('float32'), trainable = True)

    # Convert numpy mask to tensorflow tensor
    mask = tf.convert_to_tensor(mask, dtype = tf.bool)
    mask = np.logical_not(mask)

    # Get the column index for the loss calculation
    column_index = df.columns.get_loc(column_name)

    # Iterate num_iterations rounds to impute missing values
    for i in range(num_iterations):
        # Get imputed data
        imputed_data = vae(missing_data)
        imputed_data = reshape_dae_output(imputed_data)

        # RMSE on specified column
        # loss = tf.sqrt(tf.reduce_mean(tf.square((imputed_data[:, column_index] - true_data[:, column_index])[mask[:, column_index]])))
        # MAE on specified column
        loss = tf.reduce_mean(tf.abs(imputed_data[:, column_index] - true_data[:, column_index])[mask[:, column_index]])
        print('Target column loss:', loss)

        loss = tf.reduce_mean(tf.abs(imputed_data - true_data)[mask])
        print('Loss:', loss)

        # Apply mask to imputed and true data
        imputed_data_masked = tf.boolean_mask(imputed_data[:, column_index], mask[:, column_index])
        true_data_masked = tf.boolean_mask(true_data[:, column_index], mask[:, column_index])

        # Create a mask for correct imputations
        accuracy_mask = (tf.abs(imputed_data_masked - true_data_masked) < 0.1)

        # Calculate accuracy by dividing the number of correct imputations by the total number of imputations
        accuracy = tf.reduce_mean(tf.cast(accuracy_mask, tf.float32))

        print('Imputation accuracy:', accuracy)

        # Fill missing parts of missing_data with imputed_data
        missing_data.assign(tf.where(mask, imputed_data, missing_data))

    df_copy[:] = missing_data.numpy()

    return df_copy, mask


# Interpolation process function. This function is used for ensemble methods.
def insert_and_impute_ensemble(df, df_true, train_df, column_name, fraction, dae1, dae2, dae3, num_iterations = 10):
    # Create a copy of df
    df_copy = df.copy()
    df_copy_np = df_copy.values

    # create a new array with the same shape as x_train_np but with all elements -1
    df_copy_np_minus_one = np.full(df_copy_np.shape, -1)

    # Create a mask, the shape of the mask is the same as x_train_np, p = masking_factor represents the probability of 1 on the mask
    masking_factor = 1 - fraction
    mask = np.random.binomial(n = 1, p = masking_factor, size = df_copy_np.shape)

    # Use the tf.where function to replace the value in x_train_np with -1 according to the mask to get the noise data
    df_copy_np_noisy = tf.where(mask, df_copy_np, df_copy_np_minus_one)
    df_copy_np_noisy = tf.cast(df_copy_np_noisy, tf.float32)

    # Convert dataframe to input format accepted by vae
    missing_data = tf.Variable(df_copy_np_noisy, dtype = tf.float32, trainable=True)
    true_data = tf.Variable(df_true.values.astype('float32'), trainable = True)

    # Convert numpy mask to tensorflow tensor
    mask = tf.convert_to_tensor(mask, dtype = tf.bool)
    mask = np.logical_not(mask)

    # Get the column index for the loss calculation
    column_index = df.columns.get_loc(column_name)

    # Iterate num_iterations rounds to impute missing values
    for i in range(num_iterations):
        # Get imputed data
        imputed_data1 = dae1(missing_data)
        imputed_data2 = dae2(missing_data)
        imputed_data3 = dae3(missing_data)

        # ensemble the 3 daes' output
        imputed_data = tf.reduce_mean([imputed_data1, imputed_data2, imputed_data3], axis=0)
        imputed_data = reshape_dae_output(imputed_data)

        # RMSE on specified column
        # loss = tf.sqrt(tf.reduce_mean(tf.square((imputed_data[:, column_index] - true_data[:, column_index])[mask[:, column_index]])))
        # MAE on specified column
        loss = tf.reduce_mean(tf.abs(imputed_data[:, column_index] - true_data[:, column_index])[mask[:, column_index]])
        print('Target column loss:', loss)

        loss = tf.reduce_mean(tf.abs(imputed_data - true_data)[mask])
        print('Loss:', loss)

        # Apply mask to imputed and true data
        imputed_data_masked = tf.boolean_mask(imputed_data[:, column_index], mask[:, column_index])
        true_data_masked = tf.boolean_mask(true_data[:, column_index], mask[:, column_index])

        # Create a mask for correct imputations
        # accuracy_mask = (tf.abs(imputed_data_masked - true_data_masked) < 0.1) | ((imputed_data_masked < -0.1) & (true_data_masked < 0))
        accuracy_mask = (tf.abs(imputed_data_masked - true_data_masked) < 0.1)

        # Calculate accuracy by dividing the number of correct imputations by the total number of imputations
        accuracy = tf.reduce_mean(tf.cast(accuracy_mask, tf.float32))

        print('Imputation accuracy:', accuracy)

        # Fill missing parts of missing_data with imputed_data
        missing_data.assign(tf.where(mask, imputed_data, missing_data))

    df_copy[:] = missing_data.numpy()

    return df_copy, mask

