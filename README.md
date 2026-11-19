# spaAnchor

spaAnchor transfers molecular features between spatial omics slices using shared features as anchors. It combines cell-level and spatial niche representations, identifies cross-slice correspondences, transfers features through attention, and predicts features across the target slice using regression.

## Installation

spaAnchor works with AnnData objects and integrates with Scanpy workflows. It can run on either a CPU or a GPU; we recommend an NVIDIA GPU for large datasets.

```bash
git clone https://github.com/hannshu/spaAnchor
cd spaAnchor
python -m pip install '.[gpu]'
```

For a CPU installation, use `python -m pip install '.[cpu]'` instead. See the [installation documentation](https://spaanchor.readthedocs.io/en/latest/install.html) for detailed environment setup instructions and information about the HIPT checkpoint required for diagonal integration.

## Quick start

The example below performs panel expansion between two slices using prepared `.h5ad` files.

### Input data

Provide a list of AnnData objects, one per slice.

| Field | Requirement |
| --- | --- |
| `adata.X` | Cells/spots × molecular features. Use raw nonnegative counts with default preprocessing. |
| `adata.var_names` | Unique feature names; shared names identify anchor features across slices. |
| `adata.obs_names` | Unique cell/spot identifiers within each slice. |
| `adata.obsm["spatial"]` | Spatial coordinates in cell/spot order; the default graph uses Delaunay triangulation. |
| `adata.obsm[feature_key]` | Additional source features as a pandas DataFrame, indexed exactly by `adata.obs_names`, with feature names as columns. |

Molecular-anchor workflows require overlapping features between the slices being compared. Diagonal integration instead requires aligned histology images.

Default preprocessing filters out cells with no expressed genes, selects highly variable genes, normalizes the total count per cell to 10,000, and applies `log1p`. For already processed inputs, set `preprocess=False`. In the standard molecular-anchor workflow, set `filter_cell=False` to disable cell filtering. For other changes, see the guide to [customizing preprocessing](https://spaanchor.readthedocs.io/en/latest/customize/preprocess.html).

### Run panel expansion

Once your data is prepared, create a `sa.panel_expansion` model and pass the slices as a list. Pass copies to preserve your original data, since spaAnchor preprocesses the inputs and appends a slice suffix to cell identifiers.

```python
import scanpy as sc
import spaAnchor as sa

# Each slice has shared anchor genes plus its own measured gene panel.
adata0 = sc.read_h5ad("slice0.h5ad")
adata1 = sc.read_h5ad("slice1.h5ad")

model = sa.panel_expansion()
results = model([adata0.copy(), adata1.copy()])

# Genes transferred FROM slice 1 TO slice 0.
predicted = results[0]["X_from_1"]
predicted.to_csv("slice0_predicted_from_slice1.csv")
```

The output is a nested dictionary:

```text
results[target_slice]["<feature_key>_from_<source_slice>"] -> pandas.DataFrame
```

For example, `results[0]["X_from_1"]` contains predictions for slice 0 using slice 1 as the reference. Rows correspond to retained target cells, with slice suffixes such as `-0`; columns identify the predicted features.

## Tutorials and customization

The [spaAnchor documentation](https://spaanchor.readthedocs.io/en/latest/) provides installation instructions, tutorials for individual analysis tasks, and an API reference. Explore the examples below to prepare your data, run the models, and interpret their outputs, or follow the customization guide to adapt the pipeline to your own analysis.

- [Panel expansion](https://spaanchor.readthedocs.io/en/latest/panel_expansion.html)
    - [Multi-slice panel expansion](https://spaanchor.readthedocs.io/en/latest/multi_slice_panel_expansion.html)
- [Cross-omics translation](https://spaanchor.readthedocs.io/en/latest/omics_translation.html)
- [Diagonal integration](https://spaanchor.readthedocs.io/en/latest/diagonal_integration.html)
- [Asymmetric translation](https://spaanchor.readthedocs.io/en/latest/asymmetric_translation.html)
- [Customize spaAnchor](https://spaanchor.readthedocs.io/en/latest/customize.html)
- [HANN API reference](https://spaanchor.readthedocs.io/en/latest/HANN_class.html)

## Established applications and new possibilities

spaAnchor provides ready-to-use models for the following applications. These offer starting points for exploring additional spatial omics integration tasks through custom preprocessing, representations, and prediction models.

| Task | Model | Input anchor / prediction target |
| --- | --- | --- |
| Panel expansion | `sa.panel_expansion` | Shared molecular features / molecular panels measured only in other slices |
| RNA-to-protein translation | `sa.rna2protein` | Shared molecular features / proteins in `obsm["protein"]` |
| RNA-to-epigenomics translation | `sa.rna2epig` | Shared molecular features / features in `obsm["epigenomics"]` |
| Diagonal integration | `sa.diagonal_integration` | Aligned histology images / molecular panels measured in other slices |
| Asymmetric translation | `sa.asymmetric_translation` | Shared molecular features / multiple feature types from references |

`sa.HANN` exposes the common pipeline for customization. We welcome new applications of spaAnchor from the spatial omics community. If you adapt the framework to a new modality, dataset, or analysis task and would like to share your work, contributions of custom spaAnchor classes through [pull requests](https://github.com/ystlab/spaAnchor/pulls) are welcome. Your implementation may also be useful to other researchers. The [customization guide](https://spaanchor.readthedocs.io/en/latest/customize.html) provides examples for adapting the pipeline.
