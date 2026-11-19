from .core import HANN


class panel_expansion(HANN):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
 

class rna2protein(HANN):

    def __init__(
        self, 
        reduction_opt="harmony", 
        reduction_dim=20, 
        scale=True, 
        feature_key="protein", 
        regression_feat="X_pp",
        **kwargs
    ):
        super().__init__(
            reduction_opt=reduction_opt, 
            reduction_dim=reduction_dim,
            scale=scale, 
            feature_key=feature_key,
            regression_feat=regression_feat,
            **kwargs
        )


class rna2epig(rna2protein):

    def __init__(
        self, 
        feature_key="epigenomics", 
        reduction_dim=100, 
        **kwargs
    ):
        super().__init__(
            feature_key=feature_key, 
            reduction_dim=reduction_dim,
            **kwargs
        )


    def translate(
        self, 
        source_emb, 
        source_feat, 
        target_emb, 
        target_feat, 
        hnn_pairs, 
        index, 
        columns, 
        **kwargs
    ):
        import pandas as pd
        from ..utils.translation import KNN_regression
        from ..utils.attn_model import attn_translation
        from ..utils.utils import lsi
        
        impute_result = attn_translation(
            source_x=source_emb,
            source_y=lsi(source_feat, 50),  # Train by LSI-reduced features
            target_x=target_emb,
            raw_edge_index=hnn_pairs,
            # To activate the latent feature learn -> expression feature aggregate
            # mode, user should input the source feature as `t_source_y`, and the 
            # model will learn to predict the target feature by aggregating source 
            # features according to the attention weight.
            t_source_y=source_feat,
            device=self.device,
        )

        # Due to the extremely high dimension of the target feature, we use the 
        # KNN regression to do global prediction.
        impute_result = KNN_regression(impute_result, target_emb)
        return pd.DataFrame(impute_result, index=index, columns=columns)


class asymmetric_translation(HANN):

    def __init__(
        self, 
        preprocess=False, 
        reduction_opt="harmony", 
        imbalanced=True, 
        **kwargs
    ):
        super().__init__(
            preprocess=preprocess, 
            reduction_opt=reduction_opt, 
            imbalanced=imbalanced, 
            **kwargs
        )
