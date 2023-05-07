import pandas as pd

from sklearn.tree import DecisionTreeRegressor


class Preprocessor:
    def __init__(self):
        self.fill_with_mean = dict()
        self.fill_with_tree = dict()

    def fit(self, train_path):
        # Reading dataset and dropping ID column.
        df = pd.read_csv(train_path)
        df = df.drop('recordid', axis=1)

        # Splitting into features and label.
        X = df.drop('In-hospital_death', axis=1)
        y = df['In-hospital_death']

        # Filling with mean those columns which have nan less than 50.
        for col in X.columns:
            if X[col].isna().sum() <= 50:
                # Saving columns and corresponding values for further transformation. 
                value = X[col].mean()
                self.fill_with_mean[col] = value
                
                # Filling with mean value of the column.
                X[col] = X[col].fillna(value)

        # Filling with DecisionTree prediciton those columns that have many nans.
        for col in X.columns:
            if X[col].isna().sum() > 0:
                # Taking as features those columns that don't have nans, and as label our current column.
                feat, label = X[~X[col].isna()].loc[:, self.fill_with_mean.keys()], X[~X[col].isna()].loc[:, col]

                # Fitting DecisionTree.
                model = DecisionTreeRegressor(max_depth=1, random_state=0)
                model.fit(feat, label)

                # Filling with model prediction.
                X.loc[X[col].isna(), col] = model.predict(X[X[col].isna()][self.fill_with_mean.keys()])

                # Saving columns and corresponding models for further transformation.
                self.fill_with_tree[col] = model
        
    def transform(self, test_path, test=True):
        # Reading test data from .csv file.
        df = pd.read_csv(test_path)

        # Filling some columns with mean.
        for col, value in self.fill_with_mean.items():
            df[col] = df[col].fillna(value)

        # Filling rest with corresponding model predictions.
        for col, model in self.fill_with_tree.items():
            df.loc[df[col].isna(), col] = model.predict(df[df[col].isna()][self.fill_with_mean.keys()])

        if not test: 
            return df.drop('In-hospital_death', axis=1), df['In-hospital_death']

        return df



