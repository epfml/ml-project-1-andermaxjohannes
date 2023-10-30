# CS433 Project 1
### Anders Dillerud Eriksen, Maximilian Grobbelaar, Johannes Kjær

## Implementations
The six required regression implementations can be found in implementations.py, along with all other implementations of different regression functions, data preprocessing, visualization, and general helper functions.

## Final predictions
The code used to produce our final AICrowd submission with id #243813 is in run.ipynb, including the fixed random seed used for the prediction. The code saves the predictions in the folder Predictions.

## Data preprocessing
As described in the code and in the report, the data preprocessing consists of first removing the features with less than 70% (or some other threshold) valid features. Then all the samples with more than five (or some other threshold) invalid entries were removed. These two steps ensure that the data we are working with is actually real data, and not just some filler values we have chosen.
Then we remove outliers that are more than ten, or some other threshold, standard deviations from the mean, so outliers won't have an outsized impact on the data.
The data is then standardized (the feature mean is subtracted and then the data is devided by the feature standard deviation).
A dummy variable (a column of ones) is subsequently added.
Then all remaining invalid values are replaced by zeros, which after the standardization is the mean.
Finally, the dataset is balanced by selecting all the positive samples, and equally many negative samples. In practice we select more negative samples than positive, to clue the model in on the real base rate. For our final prediction we used 2.5 times as many negative samples as positive ones.

## Where to place the data?
To run the code, the data must be in the folder Data, and must be called x_test.csv, x_train.csv, and y_train.csv respectively.

## Figures
In Figures some figures produced by some of our code is available.

## Other code
DataCleaning.ipynb was used primarily for exploring the data a bit.

testing.ipynb was used to quickly test our implementations, for which the height_weight_genders.csv dataset from the exercises was used. It should also be placed in the Data folder for the notebook to run.