import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_map = {}

    def fit(self, X: pd.Series, y=None):
        # Compute state frequency on training set
        self.freq_map = X.value_counts(normalize=False).to_dict()
        return self

    def transform(self, X: pd.Series):
        # Encode the states in the test data set according to the frequencies calculated in the training set.
        # If a state was not in the training set, we assign it a frequency of 1.
        freq_map_ = self.freq_map.copy()
        
        for x in X: 
            if x not in self.freq_map.keys():
                freq_map_[x] = 1

        return pd.DataFrame(X.map(freq_map_))
    
    def set_output(self, transform="pandas"):
        self._transform_output = transform
        return self
    
def frequency_transformer() -> ColumnTransformer:
    return ColumnTransformer(transformers=[('frequency_encoding',FrequencyEncoder(),'state')]
                             ,remainder='passthrough'
                             ,verbose_feature_names_out=False)