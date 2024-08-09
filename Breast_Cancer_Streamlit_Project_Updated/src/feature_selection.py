
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector

if __name__ == '__main__':
    df = pd.read_csv('../data/breast_cancer_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    X_new, selector = select_features(X, y)
    selected_features = X.columns[selector.get_support()]
    print('Selected features:', selected_features)
