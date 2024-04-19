from abc import ABC, abstractmethod
import pandas as pd

class Model(ABC):
    """
    Abstract base class representing a machine learning model that processes
    a test dataframe with columns 'id' and 'problem' (text) and outputs a prediction
    dataframe with columns 'id' and 'output'.
    """
    
    @abstractmethod
    def predict(self, test_df: pd.DataFrame, n_samples=-1) -> pd.DataFrame:
        """
        Make predictions based on the problem text provided in the test dataframe.
        
        Parameters:
            test_df (pd.DataFrame): A dataframe containing at least two columns:
                                    'id' - An identifier for each entry.
                                    'problem' - The text data on which predictions need to be made.
                                    DO NOT MUTATE test_df
            n_samples: samples how many rows from test_df for predictions. For testing.
                       Default is -1, meaning all rows are used.
        
        Returns:
            pd.DataFrame: A dataframe that must contain the following three columns:
                          'id' - The identifier for each entry.
                          'problem' - same as 'problem' in test_df
                          'output' - The prediction for each problem.
        """
        pass

def mod1000(output):
    if output is not None:
        return int(output) % 1000
