import pandas as pd
from tools import colors, loading_animation
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def check_len_colums(headers, csv_data):
    dic = {}
    index = 0
    for elt in headers:
        dic[elt] = csv_data[elt].tolist()
        if index == 0:
            size = len(dic[elt])
        else:
            try:
                if len(dic[elt]) != size:
                    raise AssertionError("invalid csv.")
            except AssertionError as error:
                print(colors.clr.fb.red, "Error:", error, colors.clr.reset)
                sys.exit(1)
        index += 1
    
    return dic


def insert_data_in_list(dic, headers):
    data_list = []
    index = 0
    for elt in dic[headers[0]]:
        try:
            index2 = 0
            new_list = []
            for elt in headers:
                if pd.isna(dic[headers[index2]][index]):
                    dic[headers[index2]][index] = None
                new_list.append(dic[headers[index2]][index])
                index2 += 1
            data_list.append(new_list)
        except Exception:
                print(colors.clr.fg.yellow, f"Warning: an error occured on line {index + 2}.", colors.clr.reset)
        index += 1
    
    return data_list, dic


def find_type_of_column(headers, data_list):
    columns_type = {}

    index = 0
    for elt in headers:
        columns_type[elt] = type(data_list[0][index])
        index += 1

    return columns_type


def apply_normalisation(headers, csv_data):

    data = csv_data
    # Sélectionner uniquement les colonnes numériques à standardiser
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    # Exclure la colonne spécifique de la standardisation
    column_to_exclude = 'id'
    numerical_columns = numerical_columns.drop(column_to_exclude)
    # Créer un objet StandardScaler
    scaler = MinMaxScaler()
    # Appliquer la standardisation sur les colonnes numériques (à l'exception de la colonne spécifique)
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    # Enregistrer les données standardisées dans un nouveau fichier CSV
    data.to_csv('datasets/nrmlst_data.csv', index=False)

    return data


def apply_standardisation(headers, csv_data):

    data = csv_data
    # Sélectionner uniquement les colonnes numériques à standardiser
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    # Exclure la colonne spécifique de la standardisation
    column_to_exclude = 'id'
    numerical_columns = numerical_columns.drop(column_to_exclude)
    # Créer un objet StandardScaler
    scaler = StandardScaler()
    # Appliquer la standardisation sur les colonnes numériques (à l'exception de la colonne spécifique)
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    # Enregistrer les données standardisées dans un nouveau fichier CSV
    data.to_csv('datasets/std_data.csv', index=False)

    return data
    

def parse(csv_data, path):
    """
    Parse data from csv file.
    """

    # start loading animation
    print("")
    loading_animation.start("Parsing " + path + "...")

    # get list of header csv
    headers = csv_data.columns.values.tolist()

    # function for create a dictionnary of data and check len of all column, error if different
    data_dic = check_len_colums(headers, csv_data)

    # function for create a list of data
    data_list, data_dic = insert_data_in_list(data_dic, headers)

    # function for find type of each column
    columns_type = find_type_of_column(headers, data_list)

    # function for get std numeric data
    data_std = apply_standardisation(headers, csv_data)

    # function for get nrmlst numeric data
    data_nrmlst = apply_normalisation(headers, csv_data)
    
    # create csv_objet with all informations and data
    csv_object = {}
    csv_object["headers"] = headers
    csv_object["data_brut"] = csv_data
    csv_object["data_list"] = data_list
    csv_object["data_dic"] = data_dic
    csv_object["data_std"] = data_std
    csv_object["data_nrmlst"] = data_nrmlst
    csv_object["columns_len"] = len(data_list)
    csv_object["columns_number"] = len(headers)
    csv_object["columns_type"] = columns_type

    # stop loading animation
    loading_animation.stop()

    print(colors.clr.fg.green, "Parse success.", colors.clr.reset)
    print("")
    print("")

    return(csv_object)

