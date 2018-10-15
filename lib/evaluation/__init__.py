import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, r2_score

def get_errors(true, predicted):
    print('Accuracy score: ', accuracy_score(true, predicted))
    print('Log loss: ', log_loss(true, predicted))
    print('R2: ', r2_score(true, predicted))
    print('Calibration: ', predicted.mean() / true.mean())
    return ('Done')


def export(self, path):
    self._build_df()
    return pd.DataFrame(self.error_results).to_csv(path, index=False)