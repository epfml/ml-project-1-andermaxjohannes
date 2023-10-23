"""Some helper functions for project 1."""
import csv
import numpy as np


def loadData(dataPath):
    ''' Loads data and returns it as masked numpy array. A masked array contains information about which values are invalid, ensuring methods like .mean() ignores the masked values
    Args:
        dataPath: The file path of the data
    Returns:
        data: (N,d) masked numpy array, where N is the number of samples, and d is the dimension of the x values, or 1 if the data in question are the labels
        header: (d,) array with the column names
    '''
    data = np.genfromtxt(dataPath, delimiter=',', skip_header=1, dtype=float, usemask=True) # Loading the data as a masked array (with usemask=True), skipping the header, and specifying that the values are floats
    header = np.genfromtxt(dataPath, delimiter=',', dtype=str, max_rows=1) # Loading the first row of the csv file, i.e. the header
    return data , header


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
