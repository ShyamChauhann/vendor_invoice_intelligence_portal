import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split

def load_vendor_invoice_data(db_path:str):
    """
    load data from SQLite database
    """
    conn=sqlite3.connect(db_path)
    query="SELECT * FROM  vendor_invoice"
    df=pd.read_sql_query(query,conn)
    conn.close()
    return df

def prepare_features(df:pd.DataFrame):
    """
    Select features and target variables
    """
    X=df[["Dollars"]]
    Y=df["Freight"]
    return X,Y

def split_data(X,Y,test_size=0.2,random_state=42):
    """
    Split dataset in train and test sets
    """
    return train_test_split(
        X,Y,test_size=test_size,random_state=random_state
    )