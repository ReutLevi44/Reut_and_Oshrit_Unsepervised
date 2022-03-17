import pandas as pd


data = pd.read_csv('USCensus1990.data.csv', index_col=0)

external_variables_names = ['dAge', 'dHispanic', 'iYearwrk', 'iSex']

external_variables = data[external_variables_names]
data.drop(external_variables_names, axis=1, inplace=True)

data = pd.get_dummies(data.astype(str))

external_variables.to_csv('external_data.csv')
data.to_csv('one_hot_data.csv')

print('')
