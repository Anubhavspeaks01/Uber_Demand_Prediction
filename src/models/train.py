import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn import set_config
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


set_config(transform_output="pandas")

# create a logger
logger = logging.getLogger("train_model")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def save_model(model, save_path):
    joblib.dump(model, save_path)
    
    
if __name__ == "__main__":
    # current path
    current_path = Path(__file__)
    # set the root path
    root_path = current_path.parent.parent.parent
    # data_path
    data_path = root_path / "data/processed/train.csv"
    
    # read the data
    df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read successfully")
    
    # set the datetime column as index
    df.set_index("tpep_pickup_datetime", inplace=True)
    
    # make X_train and y_train
    X_train = df.drop(columns=["total_pickups"])
    y_train = df["total_pickups"]
    
    # make the transformer
    encoder = ColumnTransformer([
        ("ohe", OneHotEncoder(drop="first",sparse_output=False), ["region","day_of_week"])
        ], remainder="passthrough", n_jobs=-1,force_int_remainder_cols=False)
        
    # fit the transformer
    encoder.fit(X_train)
    
    # save the transformer
    encoder_save_path = root_path / "models/encoder.joblib"
    joblib.dump(encoder, encoder_save_path)
    logger.info("Encoder saved successfully")
    
    # encode the training data
    X_train_encoded = encoder.fit_transform(X_train)
    logger.info("Data encoded successfully")
    
    # train the model
    lr = LinearRegression()

    # fit on the training data
    lr.fit(X_train_encoded, y_train)
    logger.info("Model trained successfully")
    
    # save the model
    model_save_path = root_path / "models/model.joblib"
    save_model(lr, model_save_path)
    logger.info("Model saved successfully")