
import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_and_prepare_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

if __name__ == '__main__':
    df = load_and_prepare_data()
    df.to_csv('../data/breast_cancer_data.csv', index=False)
    print('Data saved to ../data/breast_cancer_data.csv')
