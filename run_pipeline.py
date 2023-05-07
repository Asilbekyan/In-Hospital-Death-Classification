import pandas as pd
import argparse
import joblib
import json
from Model import Model
from Preprocessor import Preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

class Pipeline:
    def __init__(self):
        self.model = Model()
        self.preprocessor = Preprocessor()

    def run(self, data_path, test=False):
        if test:
            self.preprocessor = joblib.load('preprocessor.pkl')
            self.model = joblib.load('model.pkl')

            X = self.preprocessor.transform(data_path, test=True)
            
            with open('predictions.json', 'w') as f:
                json.dump({'predict_probas': self.model.predict_proba(X).tolist(), 'threshold': self.model.threshold}, f)
        else:
            self.preprocessor.fit(data_path)
            X, y = self.preprocessor.transform(data_path, test=False)

            self.model.fit(X, y)
            
            joblib.dump(self.preprocessor, 'preprocessor.pkl')
            joblib.dump(self.model, 'model.pkl')


def main():
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help="Data path", required=True)
    parser.add_argument('--inference', type=bool, help="Model testing", required=False, default=False)
    args = parser.parse_args()

    # Creating and running pipeline    
    pipeline = Pipeline()
    pipeline.run(args.data_path, test=args.inference)

if __name__ == '__main__':
    main()
