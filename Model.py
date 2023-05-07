from sklearn.ensemble import HistGradientBoostingClassifier
class Model:
    def __init__(self, threshold=0.5):
        self.clf = None
        self.threshold=threshold
    
    def fit(self, X, y, random_state=0, class_weight='balanced', max_iter=120, l2_regularization=0.001):
        '''train our model'''
        
        clf = HistGradientBoostingClassifier(random_state=random_state, class_weight=class_weight, max_iter=max_iter, l2_regularization=l2_regularization)
        
        clf.fit(X, y)
        self.clf = clf

    def pred(self, X):
        '''making prediction in test data'''
        y_pred = self.clf.predict_proba(X)

        for i, pred in enumerate(y_pred):
            y_pred[i] = 1 if pred >= self.threshold else 0

        return y_pred

    def predict_proba(self, X):
        '''predict probabilities'''
        return self.clf.predict_proba(X)
    
    def score(self, pred, y):
        '''return accuracy'''
        
        return self.clf.score(pred, y)
