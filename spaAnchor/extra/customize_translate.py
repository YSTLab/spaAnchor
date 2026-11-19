import spaAnchor as sa
import numpy as np
from typing import Optional


class custom_mlp(sa.HANN):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def feature_regression(
        self,
        source: np.ndarray, 
        target: np.ndarray, 
        predict: Optional[np.ndarray] = None,
        top_pcs: int = 50, 
        # Although this parameter not used in this function, 
        # this still need to be kept for compatibility with 
        # the original function signature.
        n_jobs: int = -1,
        **kwargs
    ) -> np.ndarray:
        
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.neural_network import MLPRegressor

        predict = predict if (isinstance(predict, np.ndarray)) else source
        assert (source.shape[1] == predict.shape[1]), "ERROR: source and predict must have same feature size."

        regressor = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(top_pcs, source.shape[1]), random_state=0)),
            ("mlp", MLPRegressor(random_state=0))
        ])
        return regressor.fit(source, target).predict(predict)

