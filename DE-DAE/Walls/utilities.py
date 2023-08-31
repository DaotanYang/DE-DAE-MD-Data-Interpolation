''''
Author: Daotan Yang

Date: 31/08/2023

Description: To make it easier to understand the code, I've given an explanation of the function's function, 
inputs and outputs in the comments of each function. Also the more important statements or statements that 
are difficult to understand are commented. Because of the large number of functions in this file, 
I have designed an index to mark the location of each function in the file for your convenience.

Index:
Functions                       Location
dataloader                      Line 47
print_dtypes                    Line 120
check_missing_values            Line 135
encode_dropna                   Line 150
precess_postal_code             Line 169
round_to_nearest_decade         Line 191
process_year                    Line 204
output_dtype                    Line 218
output_missing_rate             Line 232
remove_outliers                 Line 251
output_distribution             Line 274
output_heatmap                  Line 304
count_and_plot_completed_data   Line 320
count_and_plot_missing_data     Line 353
plot_loss                       Line 394

'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def dataloader(path_dataset):
    '''
    Func:
        load dataset
    Input:
        path_dataset : string, save path for datasets
    Output:
        df_DPE_dataset : DataFrame

    '''
    df_DPE_dataset = pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/DPE_dataset/dpe-v2-logements-existants.csv',
                usecols = ['Conso_ECS_é_primaire',
                      'Qualité_isolation_plancher_bas',
                      'Qualité_isolation_murs',
                      'Qualité_isolation_menuiseries',
                      'Qualité_isolation_plancher_haut_toit_terrase',
                      'Conso_5_usages_é_primaire',
                      'Conso_ECS_é_finale',
                      'Type_énergie_principale_ECS',
                      'Conso_é_finale_générateur_ECS_n°1',
                      'Conso_é_finale_générateur_ECS_n°2',
                      'Isolation_toiture_(0/1)',
                      'Deperditions_planchers_bas',
                      'Qualité_isolation_enveloppe',
                      'Déperditions_murs',
                      'Déperditions_portes',
                      'Deperditions_baies_vitrées',
                      'Deperditions_planchers_hauts',
                      'Qualité_isolation_plancher_haut_comble_aménagé',
                      'Qualité_isolation_plancher_haut_comble_perdu',
                      'Année_construction',
                      'Surface_habitable_immeuble',
                      'Surface_habitable_logement',
                      'Code_postal_(BAN)',
                      ],
                sep = ',')

    # Column name mapping dictionary
    name_dict = {
        'Conso_ECS_é_primaire': 'Primary_ECS_Consumption',    
        'Qualité_isolation_plancher_bas': 'Quality_insulation_lower_floor', 
        'Qualité_isolation_murs': 'Quality_insulation_walls',   
        'Qualité_isolation_menuiseries': 'Quality_insulation_carpentry',    
        'Qualité_isolation_plancher_haut_toit_terrase': 'Quality_insulation_upper_floor_roof_terrace',  
        'Conso_5_usages_é_primaire': 'Primary_5_usages_consumption',    
        'Conso_ECS_é_finale': 'Final_ECS_Consumption',  
        'Type_énergie_principale_ECS': 'Type_main_energy_ECS',  
        'Conso_é_finale_générateur_ECS_n°1': 'Final_consumption_ECS_generator_no1',
        'Conso_é_finale_générateur_ECS_n°2': 'Final_consumption_ECS_generator_no2', 
        'Isolation_toiture_(0/1)': 'Roof_insulation_(0/1)',     
        'Deperditions_planchers_bas': 'Losses_lower_floors',
        'Qualité_isolation_enveloppe': 'Quality_insulation_envelope',   
        'Déperditions_murs': 'Losses_walls',    
        'Déperditions_portes': 'Losses_doors',  
        'Deperditions_baies_vitrées': 'Losses_glazed_bays',     
        'Deperditions_planchers_hauts': 'Losses_upper_floors',      
        'Qualité_isolation_plancher_haut_comble_aménagé': 'Quality_insulation_upper_floor_arranged_attic',  
        'Qualité_isolation_plancher_haut_comble_perdu': 'Quality_insulation_upper_floor_lost_attic',    
        'Année_construction': 'Year_construction',  
        'Surface_habitable_immeuble': 'Living_area_building',   
        'Surface_habitable_logement': 'Living_area_housing',    
        'Code_postal_(BAN)': 'Postal_code_(BAN)',  
    }

    # rename columns
    df_DPE_dataset.rename(columns = name_dict, inplace = True)

        

    return df_DPE_dataset


# output the data type of different featurers
def print_dtypes(df):
    '''
    Func:
        print the list of data type in dataframe
    Input:
        df : Dataframe
    Output:
        No return

    '''
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")


# output the missing ratio of featurers
def check_missing_values(df):
    '''
    Func:
        compute the missing ratio of features
    Input:
        df : Dataframe
    Output:
        feature: dictionary, store the missing ratio of features

    '''
    missing_values = df.isnull().mean()
    missing_features = missing_values[missing_values > 0].index.tolist()
    return {feature: missing_values[feature] for feature in missing_features}


def encode_dropna(dataset):
    '''
    Func:
        label encoding and drop NA
    Input:
        dataset : Dataframe
    Output:
        dataset_encoded_no_missing_values : Dataframe, label encoded and without missing values

    '''
    mapping = {'insuffisante': 0, 'moyenne': 1, 'bonne': 2, 'très bonne': 3}
    dataset_encoded = dataset.replace(mapping)

    # Get data without missing values
    dataset_encoded_no_missing_values = dataset_encoded.dropna()

    return dataset_encoded_no_missing_values


def process_postal_code(df):
    '''
    Func:
        encoding of postcodes according to French provinces
    Input:
        df : Dataframe
    Output:
        df : Dataframe

    '''
    # Convert the zip code column to a string, and use str.zfill() to complete five digits
    df['Postal_code_(BAN)'] = df['Postal_code_(BAN)'].astype('Int64').astype(str).str.zfill(5)

    # Create a new column to store the processed zip code
    df['Postal_code_(BAN)'] = df['Postal_code_(BAN)'].apply(lambda x: x[:2] if x[:2].isdigit() else 'NaN')

    # Convert the new column to float
    df['Postal_code_(BAN)'] = df['Postal_code_(BAN)'].apply(lambda x: float(x) if x != 'NaN' else float('nan'))

    return df


def round_to_nearest_decade(year):
    '''
    Func:
        Approximate year of construction to within 10 years (one decade)
    Input:
        year : int/float, year of construction
    Output:
        year : decade
    
    '''
    return math.floor(year / 10) * 10


def process_year(df):
    '''
    Func:
        Approximate year of construction to within 10 years (one decade)
    Input:
        df : Dataframe
    Output:
        df : Dataframe

    '''
    df['Year_construction'] = df['Year_construction'].apply(round_to_nearest_decade)
    return df


def output_dtype(dataset):
    '''
    Func:
        Output the names of all features in the dataset and the data types of the features
    Input:
        dataset : Dataframe
    Output:
        0

    '''
    print_dtypes(dataset)
    return 0


def output_missing_rate(dataset):
    '''
    Func:
        output missing rate of all features
    Input:
        dataset : Dataframe
    Output:
        return 0

    '''
    missing_results = check_missing_values(dataset)
    df_missing_results = pd.DataFrame(columns=['feature', 'missing_ratio'])
    df_missing_results['feature'] = missing_results.keys()
    df_missing_results['missing_ratio'] = missing_results.values()
    # print('The information about missing values in DPE data: \n')
    print(df_missing_results)
    return 0


def remove_outliers(df, columns):
    '''
    Func:
        Remove outliers for selected columns
    Input:
        df : Dataframe, dataset that requires outliers to be removed
        columns : Array of strings, columns where outliers need to be removed
    Output:
        df : Dataframe, dataset without outliers

    '''
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[column] > lower_bound) & (df[column] < upper_bound)]
    return df


def output_distribution(dataset):
    '''
    Func:
        output distribution of all features
    Input:
        dataset : Dataframe
    Output:
        return 0

    '''
    check_columns = list(dataset.columns.values)
    # check_class_balance(df_diabetic_data, check_columns)

    plt.figure(figsize = (20, 40))
    i = 1
    column = check_columns

    # plot the distributions
    for col in column:
        plt.subplot(10, 5, i)
        plt.title('The distribution of {}'.format(col))
        plt.xlabel('{}'.format(col))
        # plt.ylabel('Count')
        plt.hist(dataset[col])
        i += 1

    plt.subplots_adjust(wspace = 0.3, hspace = 0.5)
    return 0


def output_heatmap(dataset):
    '''
    Func:
        output heatmap of all features
    Input:
        dataset : Dataframe
    Output:
        return 0

    '''
    plt.figure(figsize = (10, 10))
    plt.title('Spearman Correlation of Features', size = 15)
    sns.heatmap(dataset.astype(float).corr(method = 'spearman'), linewidths = 0.1, vmax = 1.0, square = True, cmap = 'RdBu_r', linecolor = 'white', annot = True)
    return 0


def count_and_plot_completed_data(completed_data_numpy):
    '''
    Func:
        Count and plot the frequency of each value before imputation
    Input:
        df : Dataframe, no missing values.(Train set)
    Output:
        No Return

    '''
    # Count unique values in numpy array
    values, counts = np.unique(completed_data_numpy, return_counts = True)

    # Calculate frequencies
    frequencies = counts / counts.sum()
    percentages = 100 * frequencies

    # Create bar plot
    plt.figure(figsize = (10, 5))
    bars = plt.bar(values, counts, color = 'blue')

    plt.xlabel('Values')
    plt.ylabel('Counts')
    plt.title('Count of unique values in completed data')

    # Add percentage on top of each bar
    for bar, percentage in zip(bars, percentages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, round(percentage, 2), ha = 'center', va = 'bottom')

    plt.show()


def count_and_plot_missing_data(df, column_name, mask):
    '''
    Func:
        Count and plot the frequency of each value in imputed data.
    Input:
        df : Dataframe, imputed data.
        mask : The position of the missing value in the array.
        column_name : String, the name of the column to be plotted.
    Output:
        No Return

    '''
    # Get the column index
    column_index = df.columns.get_loc(column_name)

    # Apply mask to the specific column in the numpy array
    masked_data = df.values[mask, column_index]

    # Count unique values in numpy array
    values, counts = np.unique(masked_data, return_counts=True)

    # Calculate frequencies
    frequencies = counts / counts.sum()
    percentages = 100 * frequencies

    # Create bar plot
    plt.figure(figsize=(10,5))
    bars = plt.bar(values, counts, color='blue')

    plt.xlabel('Values')
    plt.ylabel('Counts')
    plt.title(f'Count of unique values in masked numpy array for {column_name}')

    # Add percentage on top of each bar
    for bar, percentage in zip(bars, percentages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(percentage, 2), ha='center', va='bottom')

    plt.show()


def plot_loss(history, save_path, fig_title):
    '''
    Func:
        plot an image of the variation of the loss function during the training process and save the image to a specified path
    Input:
        history : Array, stores the value of each round of the loss function
        save_path : String, storage path for loss function images
        fig_title : title of the figure
    Output:
        No return

    '''
    plt.plot(history)
    plt.title(fig_title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss'], loc='upper right')
    plt.savefig(save_path)
    plt.show()
