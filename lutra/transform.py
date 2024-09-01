import numpy as np
import pandas as pd

np.random.seed(42)


class Transform:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def apply_hellinger(self):
        """Apply the hellinger tranformation to the dataframe sample wise"""
        df_sqrt = np.sqrt(self.df)
        row_norms = np.linalg.norm(df_sqrt, axis=1)
        df_normalized = df_sqrt.div(row_norms, axis=0)
        self.df = df_normalized
        return self.df
