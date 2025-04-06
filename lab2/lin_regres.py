import numpy as np
import pandas as pd
import random

class MyLineReg():
    def __init__(self,
                 weights=None,
                 n_iter=100,
                 learning_rate=0.1,
                 metric=None,
                 reg=None,
                 l1_coef=0,
                 l2_coef=0,
                 sgd_sample=None,
                 random_state=42,
                 normalize=False):
        """
        Параметры:
        ----------
        weights : np.array
            Начальные веса.
        n_iter : int
            Количество итераций.
        learning_rate : float or callable
            Скорость обучения (либо константа, либо функция от номера итерации).
        metric : str
            Метрика, которую будем считать и по которой можем отслеживать лучшую модель.
        reg : str
            Тип регуляризации ('l1', 'l2', 'elasticnet').
        l1_coef : float
            Коэффициент L1-регуляризации.
        l2_coef : float
            Коэффициент L2-регуляризации.
        sgd_sample : int or float
            Если float (0 < sgd_sample < 1), то берём процент выборки для стохастического шага.
            Если int (> 1), то берём ровно столько объектов.
            Если None, то используется весь X (градиентный спуск).
        random_state : int
            Фиксируем зерно для воспроизводимости.
        normalize : bool
            Признак, нужно ли нормализовать (стандартизировать) данные.
        """
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.normalize = normalize

        self.best_score = None
        self.best_weights = None


        self.history_errors = []
        self.history_metric = []


        self.means = None
        self.stds = None

    def __str__(self):
        return (f"MyLineReg class: n_iter={self.n_iter}, "
                f"learning_rate={self.learning_rate}, reg={self.reg}")

    def gradi(self, X, y, y_pred):
        """
        Считаем градиент функции потерь MSE
        с добавлением регуляризации (если нужно).
        """
        base_grad = -2 * np.dot(X.T, (y - y_pred)) / len(y)

        if self.reg == 'l1' and self.l1_coef != 0:
            return base_grad + self.l1_coef * np.sign(self.weights)
        elif self.reg == 'l2' and self.l2_coef != 0:
            return base_grad + 2 * self.l2_coef * self.weights
        elif self.reg == 'elasticnet' and self.l1_coef != 0 and self.l2_coef != 0:
            return base_grad + self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights
        else:
            return base_grad

    def _add_intercept_column(self, X):
        """
        Вспомогательный метод, чтобы не менять исходный DataFrame.
        Возвращает копию X с добавленным столбцом 'ones' слева.
        """
        X_ = X.copy()

        if 'ones' not in X_.columns:
            X_.insert(0, 'ones', 1)
        return X_

    def _normalize_fit(self, X):
        """
        Сохраняем средние и стандартные отклонения для каждой колонки (кроме 'ones').
        """
        self.means = X.mean()
        self.stds = X.std(ddof=0)


        if 'ones' in self.means:
            self.means['ones'] = 0.0
            self.stds['ones'] = 1.0

    def _normalize_transform(self, X):
        """
        Применяем нормализацию к X, используя ранее сохранённые self.means и self.stds.
        """
        X_ = (X - self.means) / self.stds
        return X_

    def fit(self, X, y, verbose=False):

        """
        Обучение модели методом (сто)градиентного спуска.
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)


        if isinstance(self.sgd_sample, float) and 0 < self.sgd_sample < 1:
            self.sgd_sample = round(X.shape[0] * self.sgd_sample)


        random.seed(self.random_state)
        np.random.seed(self.random_state)
        print("FIT: X columns before normalization:", X.columns)
        print("FIT: X shape before normalization:", X.shape)

        if self.normalize:
            self._normalize_fit(X)
            X = self._normalize_transform(X)

        print("FIT: X shape after normalization:", X.shape)
        X_ = self._add_intercept_column(X)
        print("FIT: X_ shape after add_intercept:", X_.shape)


        self.weights = np.ones(X_.shape[1])


        y_pred_full = np.dot(X_, self.weights)
        error_full = np.mean((y - y_pred_full) ** 2)
        self.best_score = float('inf') if self.metric in ['mse', 'rmse'] else -float('inf')
        self.best_weights = self.weights.copy()

        for i in range(1, self.n_iter + 1):
            if self.sgd_sample:
                sample_rows_idx = random.sample(range(X_.shape[0]), self.sgd_sample)
                X_sample = X_.iloc[sample_rows_idx]
                y_sample = y.iloc[sample_rows_idx]
            else:
                X_sample = X_
                y_sample = y


            y_pred_sample = np.dot(X_sample, self.weights)


            lr = self.learning_rate(i) if callable(self.learning_rate) else self.learning_rate

            grad = self.gradi(X_sample, y_sample, y_pred_sample)

            self.weights -= lr * grad

            y_pred_full = np.dot(X_, self.weights)
            error_full = np.mean((y - y_pred_full) ** 2)

            current_metric_value = None
            if self.metric:
                if self.metric == 'mae':
                    current_metric_value = np.mean(abs(y - y_pred_full))
                elif self.metric == 'mse':
                    current_metric_value = error_full
                elif self.metric == 'rmse':
                    current_metric_value = np.sqrt(error_full)
                elif self.metric == 'mape':
                    eps = 1e-10
                    current_metric_value = np.mean(
                        abs((y - y_pred_full) / (y + eps))) * 100
                elif self.metric == 'r2':
                    ss_res = np.sum((y - y_pred_full) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    current_metric_value = 1 - ss_res / ss_tot
                else:
                    raise ValueError(f"Unknown metric: {self.metric}")

                if self.metric == 'r2':
                    if current_metric_value > self.best_score:
                        self.best_score = current_metric_value
                        self.best_weights = self.weights.copy()
                else:
                    if current_metric_value < self.best_score:
                        self.best_score = current_metric_value
                        self.best_weights = self.weights.copy()

            self.history_errors.append(error_full)
            self.history_metric.append(current_metric_value)

            if verbose and i % verbose == 0:
                print(f'Итерация {i} | MSE: {error_full:.4f} | lr: {lr}')
                if self.metric and current_metric_value is not None:
                    print(f'{self.metric}: {current_metric_value:.4f}')

    def get_best_score(self):
        """
        Возвращает лучшее значение метрики, найденное во время обучения.
        """
        return self.best_score

    def get_coef(self):
        """
        Возвращает веса (кроме первого, отвечающего за intercept).
        """
        return self.weights[1:]

    def load_best_weights(self):
        """
        Позволяет явно загрузить лучшие веса, если хотим,
        чтобы модель сейчас начала предсказывать именно лучшим набором.
        """
        self.weights = self.best_weights.copy()

    def predict(self, X):
        """
        Предсказание для входной матрицы X.
        Учитываем, что, если была нормализация, надо применить те же преобразования.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.normalize and self.means is not None and self.stds is not None:
            X = self._normalize_transform(X)

        X_ = self._add_intercept_column(X)

        return np.dot(X_, self.weights)
