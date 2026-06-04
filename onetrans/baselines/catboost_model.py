import numpy as np
from catboost import CatBoostClassifier, Pool
import polars as pl

class CatBoostModel:
    def __init__(
            self,
            cat_features: list[str] | None,
            iterations=100,
            learning_rate=0.1,
            depth=8,
            loss_function='Logloss',
            verbose=10,
            random_seed=42,
            task_type="GPU",
    ):
        self.cat_features = cat_features
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function=loss_function,
            verbose=verbose,
            random_seed=random_seed,
            task_type=task_type,
        )

    def train_catboost(self, X_train: pl.DataFrame, y_train: np.array) -> None:
        train_pool = Pool(
            data=X_train.to_pandas(),
            label=y_train,
            cat_features=self.cat_features
        )
        self.model.fit(train_pool)

    def eval_catboost(self, X_test: pl.DataFrame) -> np.array:
        test_pool = Pool(
            data=X_test.to_pandas(),
            cat_features=self.cat_features
        )
        catboost_probs = self.model.predict_proba(test_pool)[:, 1]
        return catboost_probs

    def fit_predict(self, X_train: pl.DataFrame, y_train: np.array, X_test: pl.DataFrame) -> np.array:
        self.train_catboost(X_train, y_train)
        return self.eval_catboost(X_test)