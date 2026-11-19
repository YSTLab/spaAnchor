import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from typing import Optional, Callable
from functools import wraps


def regressor_decorator(model_func: Callable):

    @wraps(model_func)
    def configurator(preprocess: bool = True, **model_kwargs):

        def executor(
            source: np.ndarray, 
            target: np.ndarray, 
            predict: Optional[np.ndarray] = None,
            top_pcs: int = 50, 
            n_jobs: int = -1,
            **kwargs
        ) -> np.ndarray:

            predict = predict if (isinstance(predict, np.ndarray)) else source
            assert (source.shape[1] == predict.shape[1]), "ERROR: source and predict must have same feature size."

            if (preprocess):
                n_comps = min(top_pcs, source.shape[1], source.shape[0])
                preprocessor = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=n_comps, random_state=0))
                ])
                pp_source = preprocessor.fit_transform(source)
                pp_predict = preprocessor.transform(predict)
            else:
                pp_source = source
                pp_predict = predict

            def _fit_predict(y_chunk):
                return model_func(pp_source, y_chunk, pp_predict, **model_kwargs)

            if (target.shape[1] > 1e3):
                from tqdm import tqdm
                from joblib import Parallel, delayed
                return np.hstack(Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(_fit_predict)(target[:, 50*i: min(50*(i+1), target.shape[1])])
                    for i in tqdm(
                        range(np.ceil(target.shape[1]/50).astype(int)),
                        desc="INFO: Large panel regression"
                    )
                ))
            else:
                return _fit_predict(target)

        return executor
    return configurator


# Regressor functions
@regressor_decorator
def ridge_regressor(pp_source, y_chunk, pp_predict, **kwargs):
    from sklearn.linear_model import Ridge
    return Ridge(**kwargs).fit(pp_source, y_chunk).predict(pp_predict)


@regressor_decorator
def xgb_regressor(pp_source, y_chunk, pp_predict, **kwargs):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("XGBoost is not installed. Please install it with `pip install xgboost`.")
    model = XGBRegressor(**kwargs)
    return model.fit(pp_source, y_chunk).predict(pp_predict)


@regressor_decorator
def mlp_regressor(pp_source, y_chunk, pp_predict, **kwargs):
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(**kwargs)
    return model.fit(pp_source, y_chunk).predict(pp_predict)


@regressor_decorator
def rf_regressor(pp_source, y_chunk, pp_predict, **kwargs):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(**kwargs)
    return model.fit(pp_source, y_chunk).predict(pp_predict)


@regressor_decorator
def svm_regressor(pp_source, y_chunk, pp_predict, **kwargs):
    from sklearn.svm import SVR
    model = MultiOutputRegressor(SVR(**kwargs))
    return model.fit(pp_source, y_chunk).predict(pp_predict)


@regressor_decorator
def lightgbm_regressor(pp_source, y_chunk, pp_predict, **kwargs):
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError("LightGBM is not installed. Please install it with `pip install lightgbm`.")
    model = MultiOutputRegressor(LGBMRegressor(**kwargs))
    return model.fit(pp_source, y_chunk).predict(pp_predict)


# Classification functions
@regressor_decorator
def ridge_classifier(pp_source, y_chunk, pp_predict, **kwargs):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty="l2", **kwargs)
    return model.fit(pp_source, np.argmax(y_chunk, axis=1)).predict_proba(pp_predict)


@regressor_decorator
def xgb_classifier(pp_source, y_chunk, pp_predict, **kwargs):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("XGBoost is not installed. Please install it with `pip install xgboost`.")
    model = XGBClassifier(**kwargs)
    return model.fit(pp_source, np.argmax(y_chunk, axis=1)).predict_proba(pp_predict)


@regressor_decorator
def mlp_classifier(pp_source, y_chunk, pp_predict, **kwargs):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(**kwargs)
    return model.fit(pp_source, np.argmax(y_chunk, axis=1)).predict_proba(pp_predict)


@regressor_decorator
def rf_classifier(pp_source, y_chunk, pp_predict, **kwargs):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(**kwargs)
    return model.fit(pp_source, np.argmax(y_chunk, axis=1)).predict_proba(pp_predict)


@regressor_decorator
def svm_classifier(pp_source, y_chunk, pp_predict, **kwargs):
    from sklearn.svm import SVC
    model = SVC(probability=True, **kwargs)
    return model.fit(pp_source, np.argmax(y_chunk, axis=1)).predict_proba(pp_predict)


@regressor_decorator
def lightgbm_classifier(pp_source, y_chunk, pp_predict, **kwargs):
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError("LightGBM is not installed. Please install it with `pip install lightgbm`.")
    model = LGBMClassifier(**kwargs)
    return model.fit(pp_source, np.argmax(y_chunk, axis=1)).predict_proba(pp_predict)

