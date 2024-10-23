import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from scipy.stats import randint, loguniform
import dill
import os


DATASET_PATH = Path(__file__).parent.parent / "data" / "preprocessed_postings.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "models" / "best_model_script.pkl"
SCORE_PATH = Path(__file__).parent.parent / "out" / "score.txt"


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

def make_pipeline_with_model(model: BaseEstimator) -> Pipeline:
    pipeline = Pipeline(
        [
            ('frequency_tranformer',frequency_transformer()),
            ('pca',PCA()),
            ('model',model)
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def load_and_split_dataset(filepath: Path = DATASET_PATH) -> tuple:
    postings_data = pd.read_csv(filepath)
    X, y = postings_data.drop(columns=["standardized_salary"]), postings_data["standardized_salary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    return (X_train, X_test, y_train, y_test)


def train_model(output_path: Path = OUTPUT_PATH, scorepath: Path = SCORE_PATH) -> None:
    X_train, X_test, y_train, y_test = load_and_split_dataset()

    model_param = (HistGradientBoostingRegressor(), {
                                'pca__n_components': randint(100,250),
                                'model__learning_rate': loguniform(1e-3, 1),
                                'model__max_iter': randint(110,200), 
                                'model__max_leaf_nodes': randint(90,250),
                                'model__min_samples_leaf': randint(20,70)
                    })

    pipeline = make_pipeline_with_model(model_param[0])
    param = model_param[1]

    cv = RandomizedSearchCV(pipeline
                            ,param_distributions=param
                            ,n_iter=100
                            ,cv=10
                            ,scoring="neg_root_mean_squared_error"
                            ,refit=True
                            ,verbose=True)

    if not os.path.exists(output_path):
        print("Training in progress...")
        best_model = cv.fit(X_train, y_train)
        with open(output_path, "wb") as f:
            dill.dump(best_model, f)

        with open(scorepath, "w") as f:
            f.write("In sample error: ")
            f.write(str(root_mean_squared_error(y_train,best_model.predict(X_train))))
            f.write("\n")
            f.write("Out of sample error:")
            f.write(str(root_mean_squared_error(y_test,best_model.predict(X_test))))

if __name__ == "__main__":
    train_model()