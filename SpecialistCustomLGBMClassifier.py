## ----- Modelo Proposto --------


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels

class SpecialistCustomLGBMClassifier(BaseEstimator, ClassifierMixin):
    """
    Um wrapper customizado para o LGBMClassifier que encontra o limiar de decisão
    ideal usando validação cruzada interna para maximizar o F1-Score.
    """
    def __init__(self, base_model_params=None, n_splits=5):
        self.base_model_params = base_model_params if base_model_params is not None else {}
        self.n_splits = n_splits
        self.best_threshold_ = 0.5
        self.base_model_ = LGBMClassifier(**self.base_model_params)

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.base_model_ = LGBMClassifier(**self.base_model_params)
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.base_model_params.get('random_state'))
        out_of_fold_probas = []
        out_of_fold_true_labels = []
        print(f"\n[CustomFit] Iniciando busca de limiar com {self.n_splits} folds...")
        X_array = np.asarray(X)
        y_array = np.asarray(y)

        for train_idx, val_idx in cv.split(X_array, y_array):
            X_train_inner, X_val_inner = X_array[train_idx], X_array[val_idx]
            y_train_inner, y_val_inner = y_array[train_idx], y_array[val_idx]
            self.base_model_.fit(X_train_inner, y_train_inner)
            probas = self.base_model_.predict_proba(X_val_inner)[:, 1]
            out_of_fold_probas.extend(probas)
            out_of_fold_true_labels.extend(y_val_inner)

        best_f1 = 0
        if sum(out_of_fold_true_labels) > 0:
            for threshold in np.linspace(0.05, 0.95, 50):
                preds = (np.array(out_of_fold_probas) > threshold).astype(int)
                f1 = f1_score(out_of_fold_true_labels, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    self.best_threshold_ = threshold
        else:
            self.best_threshold_ = 0.5
            best_f1 = 0.0

        print(f"[CustomFit] Limiar encontrado {self.best_threshold_:.4f} (com F1-Score (crossValidation): {best_f1:.4f})")
        self.base_model_.fit(X_array, y_array)
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X_array = np.asarray(X)
        return self.base_model_.predict_proba(X_array)

    def predict(self, X):
        check_is_fitted(self)
        y_proba = self.predict_proba(X)[:, 1]
        return (y_proba > self.best_threshold_).astype(int)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        if deep:
            for key, value in self.base_model_params.items():
                 params[f'base_model_params__{key}'] = value
        return params

    def set_params(self, **params):
        if not params:
            return self
        base_params_to_set = {}
        for key in list(params.keys()):
            if key.startswith('classifier__base_model_params__'):
                base_param_name = key.split('__')[-1]
                base_params_to_set[base_param_name] = params.pop(key)
            elif key.startswith('base_model_params__'):
                 base_param_name = key.split('__')[-1]
                 base_params_to_set[base_param_name] = params.pop(key)
        self.base_model_params.update(base_params_to_set)
        super().set_params(**params)
        self.base_model_ = LGBMClassifier(**self.base_model_params)
        return self