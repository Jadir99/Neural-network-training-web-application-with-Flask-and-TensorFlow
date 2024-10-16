import importlib
import pandas as pd
def import_modules(model_name):
    # Construct the full module name
    module = importlib.import_module(f'tensorflow.{model_name}')
    return module



def train_model():
    df=pd.read_csv('static/uploads/champs.csv')
    df.head()