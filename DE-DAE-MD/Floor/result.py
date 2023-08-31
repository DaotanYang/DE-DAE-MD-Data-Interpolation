''''
Author: Daotan Yang

Date: 31/08/2023

Description: This file is to contain the main function necessary to output the results of the project. The function is on line 45.
Lines 44-135 are the data preprocessing process.
Lines 136-190 through y are the model training and prediction processes.

If you want to run this function in your local environment, you only need to change two variables:
The first variable is path_dataset, on line 48, which you need to change to the path where your data is stored.

The second variable is save_path, on line 164, which is used to save an image of the resulting loss change. 
You need to change it to the path where you wish to save your results.

The functions used in this document are encapsulated in the utilities.py and models.py. 
You can review these two files if you wish to modify and optimise the functions. 
In both files, the functions are listed in the order in which they appear in the results file.

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

from sklearn.model_selection import train_test_split

from utilities import dataloader, encode_dropna, process_postal_code, output_dtype, output_missing_rate, process_year, remove_outliers
from utilities import output_distribution, output_heatmap, plot_loss
from models import DAE, custom_loss, insert_and_impute


if __name__ == "__main__":
    #---------------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------Data load and preprocessing-----------------------------------------------------
    #---------------------------------------------------------------------------------------------------------------------------
    path_dataset = '/content/gdrive/MyDrive/Colab Notebooks/DPE_dataset/dpe-v2-logements-existants.csv'
    
    # load data
    df_DPE_dataset = dataloader(path_dataset)

    # print basic information of the dataset
    print('The type of the features in DPE dataset: \n')
    output_dtype(df_DPE_dataset)

    print('The information about missing values in DPE data: \n')
    output_missing_rate(df_DPE_dataset)

    # reconstructe dataset: select target feature and other relevant features
    df_insulation_floor = df_DPE_dataset[['Quality_insulation_lower_floor',
                    'Quality_insulation_envelope',
                    'Roof_insulation_(0/1)',
                    'Year_construction',
                    # 'Living_area_building',
                    'Living_area_housing',
                    'Postal_code_(BAN)',
                    'Final_ECS_Consumption',
                    'Primary_5_usages_consumption',
                    'Losses_lower_floors',
                    ]]
    
    # print basic information of the reconstructed dataset
    print('The type of the features in insulation floor dataset: \n')
    output_dtype(df_insulation_floor)

    print('The information about missing values in insulation floor data: \n')
    output_missing_rate(df_insulation_floor)

    # preprocessing dataset
    # label encoding and drop NA
    df_insulation_floor_encoded_no_missing_values = encode_dropna(df_insulation_floor)
    # encode post_code
    df_insulation_floor_encoded_no_missing_values = process_postal_code(df_insulation_floor_encoded_no_missing_values)
    # processing "Year_construction"
    df_insulation_floor_encoded_no_missing_values = process_year(df_insulation_floor_encoded_no_missing_values)


    # remove outliers
    test_columns = [# 'Quality_insulation_lower_floor',
                    'Year_construction',
                    # 'Living_area_building',
                    'Living_area_housing',
                    'Postal_code_(BAN)',
                    'Final_ECS_Consumption',
                    'Primary_5_usages_consumption',
                    # 'Roof_insulation_(0/1)',
                    'Losses_lower_floors',
                    # 'Quality_insulation_envelope',
                    ]
    df_insulation_floor_encoded_no_missing_values_outliers = remove_outliers(df_insulation_floor_encoded_no_missing_values, test_columns)

    # scaling data to intervalsScaling data to intervals [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_insulation_floor_encoded_no_missing_values_outliers[[
                    # 'Quality_insulation_lower_floor',
                    'Year_construction',
                    # 'Living_area_building',
                    'Living_area_housing',
                    'Postal_code_(BAN)',
                    'Final_ECS_Consumption',
                    'Primary_5_usages_consumption',
                    # 'Roof_insulation_(0/1)',
                    'Losses_lower_floors',
                    # 'Quality_insulation_envelope',
                    ]] = scaler.fit_transform(df_insulation_floor_encoded_no_missing_values_outliers[[
                    # 'Quality_insulation_lower_floor',
                    'Year_construction',
                    # 'Living_area_building',
                    'Living_area_housing',
                    'Postal_code_(BAN)',
                    'Final_ECS_Consumption',
                    'Primary_5_usages_consumption',
                    # 'Roof_insulation_(0/1)',
                    'Losses_lower_floors',
                    # 'Quality_insulation_envelope',
                    ]])
    

    # print basic information of the preprocessed dataset
    output_distribution(df_insulation_floor_encoded_no_missing_values_outliers)
    output_heatmap(df_insulation_floor_encoded_no_missing_values_outliers)
    # end preprocessing


    #-------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------model training and prediction processes--------------------------------------
    #-------------------------------------------------------------------------------------------------------------------------
    
    dae_dataset = df_insulation_floor_encoded_no_missing_values_outliers
    target_column = 'Quality_insulation_lower_floor'

    # Divide the dataset into training and validation sets
    x_train, x_val = train_test_split(dae_dataset, test_size = 0.2, random_state = 1)
    x_true = x_val.copy()

    

    # Convert the dataset to np format
    x_train_np = x_train.values

    dae = DAE(original_dim = x_train_np.shape[1], categorical_dim = 3, continuous_dim = x_train_np.shape[1] - 3)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    dae.compile(optimizer, loss=custom_loss)

    # set up early stopping, but this has no effect when iterations_per_noise = 1.
    early_stop = EarlyStopping(monitor='loss', min_delta = 0.0005, patience=5)

    # multiple epochs with different sparsity noise
    num_epochs = 20
    iterations_per_noise = 20
    loss_history = []
    fig_title = 'Dimension-Enhanced DAE with Multi Decoder Loss'
    save_path = '/content/gdrive/MyDrive/Colab Notebooks/DPE_code/result/de_dae_md_result.png'

    for epoch in range(num_epochs // iterations_per_noise):
        # create a new array with the same shape as x_train_np but with all elements -1
        x_train_np_minus_one = np.full(x_train_np.shape, -1)

        # Create a mask, the shape of the mask is the same as x_train_np, p = masking_factor represents the probability of 1 on the mask
        masking_factor = 0.70       # Note that due to the replacement rules in tf.where later, the masking factor here should be 1-target value
        mask = np.random.binomial(n = 1, p = masking_factor, size = x_train_np.shape)

        # Use the tf.where function to replace the value in x_train_np with -1 according to the mask to get the noise data
        x_train_np_noisy = tf.where(mask == 1, x_train_np, x_train_np_minus_one)

        for iteration in range(iterations_per_noise):
            print(f"Noise Epoch {epoch + 1}/{num_epochs // iterations_per_noise}, Iteration {iteration + 1}/{iterations_per_noise}")
            # dae.fit(x_train_np_noisy, x_train_np, batch_size = 256, callbacks=[early_stop])
            history = dae.fit(x_train_np_noisy, x_train_np, batch_size = 256, callbacks=[early_stop], verbose=0)
            print("Train loss: ",history.history['loss'][0])
            loss_history.append(history.history['loss'][0])

    plot_loss(loss_history, save_path, fig_title)

    # Test the correct rate of the target column of DAE1 when the missing ratio is 20%.
    df_insulation_floor_imputed_dedaemd, mask = insert_and_impute(x_val, x_true, x_train, target_column, 0.2, dae, num_iterations = 1)

    # Test the correct rate of the target column of DAE1 when the missing ratio is 10%.
    df_insulation_floor_imputed_dedaemd, mask = insert_and_impute(x_val, x_true, x_train, target_column, 0.1, dae, num_iterations = 1)
