import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, r2_score

def get_errors(true, predicted):
    print('Accuracy score: ', accuracy_score(true, predicted))
    print('Log loss: ', log_loss(true, predicted))
    print('R2: ', r2_score(true, predicted))
    print('Calibration: ', predicted.mean() / true.mean())

def export(file, path):
    df = pd.DataFrame(file)
    return df.to_csv(path, index=None, header=None)