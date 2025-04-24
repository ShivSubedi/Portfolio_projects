# the file serves as a module to loadign and pre-processing the dataset
import pandas as pd
import os

def load_data(filepath="../data/telco_churn.csv"):
    """
    Loads the Telco data sets and performs some basic cleaning of data
    Args: file path of the dataset (i.e. .csv file)
    Returns: pd.dataFrame i.e. a dataFrame, cleaned and ready for analysis
    """


    if not os.path.exists(filepath):
        raise FileNotFoundError ("Data couln't be loaded")
       
    print('Data loaded successfully')
    
    df_telco=pd.read_csv(filepath)

    #There could be several data pre-processing steps needed like: 
     # 1. stripping extra whitespaces in column names headers
     # 2. Converting certain columns to numeric
     # 3. drop rows with missing values etc.

    #Check for 1
    print(df_telco.columns.tolist()) #no whitespace observed, so this line is optional

    #Check for 2
    # Total charges needs to be converted to numeric values, as it is one of the variable to be used in predicting churn
    #errors='coerce' converts any invalid entries to NaN instead of raising an error
    df_telco['TotalCharges']=pd.to_numeric(df_telco['TotalCharges'], errors='coerce')
    print(df_telco['TotalCharges'].dtype) #previously returned as object, now as float

    #Check for 3
    df_telco=df_telco.dropna()
    # print(df.telco['customerID'].dtype)

    return df_telco


#LOCAL TEST FOR DATA LOADING
if __name__=="__main__":
    data=load_data()
    print(data.head(5))
    print(f"Total #of rows = {len(data)}")
    

