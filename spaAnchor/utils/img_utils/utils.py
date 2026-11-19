import scanpy as sc
import numpy as np
import pandas as pd
from skimage.transform import resize
import spatialdata as sd
import spatialdata_io as sd_io
from spatialdata import (
    SpatialData,
    get_centroids,
    get_extent,
    join_spatialelement_table,
    to_circles,
)
from spatialdata._core.operations._utils import transform_to_data_extent
from spatialdata.models import get_table_keys, Image2DModel
from spatialdata.transformations.transformations import Affine
import tifffile
from typing import Optional


def load_visium_img(
    adata: sc.AnnData, 
    img: np.ndarray,
    library_id: str = "histology_img",
    img_max_size: Optional[int] = 15000
) -> sc.AnnData:

    if ("spatial" not in adata.uns):
        adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {"images": {}, "scalefactors": {}}

    if (img_max_size is None):
        downscale_rate = 1.0
    else:
        downscale_rate = img_max_size / max(img.shape[: 2])
        img = resize(img, np.floor(downscale_rate*np.array(img.shape[: 2])))

    adata.uns['spatial'][library_id]['images']['hires'] = img
    adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef'] = downscale_rate

    return adata


# modified from spatialdata_io.experimental.to_legacy_anndata
def to_legacy_anndata(
    sdata: SpatialData,
    coordinate_system: Optional[str] = None,
    table_name: Optional[str] = None,
    include_images: bool = True,
    img_max_size: int = 15000
) -> sc.AnnData:

    DOWNSCALED_HIRES_LENGTH = img_max_size
    DOWNSCALED_LOWRES_LENGTH = 600
    # check that coordinate system is valid
    css = sdata.coordinate_systems
    if coordinate_system is None:
        assert len(css) == 1, "The SpatialData object has more than one coordinate system. Please specify one."
        coordinate_system = css[0]
    else:
        assert (
            coordinate_system in css
        ), f"The SpatialData object does not have the coordinate system {coordinate_system}."
    sdata = sdata.filter_by_coordinate_system(coordinate_system)

    if table_name is None:
        assert len(sdata.tables) == 1, (
            "When the table_name is not specified, the SpatialData object must have exactly one table. The specified "
            f"sdata object, after being filtered for the coordinate system {coordinate_system}, has "
            f"{len(sdata.tables)} tables."
        )
        table_name = next(iter(sdata.tables))
    else:
        assert table_name in sdata.tables, f"The table {table_name} is not present in the SpatialData object."

    table = sdata[table_name]
    (
        region,
        _,
        instance_key,
    ) = get_table_keys(table)
    if not isinstance(region, list):
        region = [region]

    # the table needs to annotate exactly one Shapes element
    if len(region) != 1:
        raise ValueError(f"The table needs to annotate exactly one element. Found {len(region)}.")
    if region[0] not in sdata.shapes and region[0] not in sdata.labels:
        raise ValueError("The table needs to annotate a Shapes or Labels element, not Points.")
    element = sdata[region[0]]
    region_name = region[0]

    # convert polygons, multipolygons and labels to circles
    shapes = to_circles(element)
    circles_sdata = SpatialData(tables={table_name: table}, shapes={region_name: shapes})

    joined_elements, new_table = join_spatialelement_table(
        sdata=circles_sdata, spatial_element_names=region_name, table_name=table_name, how="inner", match_rows="left"
    )

    # the table after the filtering must not be empty
    assert len(new_table) > 0, (
        "The table does not annotate any geometry in the Shapes element. This could also be caused by a mismatch "
        "between the type of the indices of the geometries and the type of the INSTANCE_KEY column of the table."
    )

    sdata_pre_rasterize = SpatialData(
        tables={table_name: new_table}, shapes={region_name: joined_elements[region_name]}
    )

    # process the images
    downscaled_images_hires = {}
    downscaled_images_lowres = {}
    sdata_post_rasterize = None
    if not include_images:
        sdata_post_rasterize = sdata_pre_rasterize
    else:
        sdata_images = sdata.subset(element_names=list(sdata.images.keys()))
        for image_name, image in sdata_images.images.items():
            sdata_pre_rasterize[image_name] = image
        bb = get_extent(sdata_images, coordinate_system=coordinate_system)

        parameters_dict = {"x": "target_width", "y": "target_height", "z": "target_depth"}
        longest_side = max(bb, key=bb.get)
        parameter_hires = {parameters_dict[longest_side]: DOWNSCALED_HIRES_LENGTH}
        parameter_lowres = {parameters_dict[longest_side]: DOWNSCALED_LOWRES_LENGTH}
        downscaled_hires = transform_to_data_extent(
            sdata_pre_rasterize, coordinate_system=coordinate_system, maintain_positioning=False, **parameter_hires
        )
        downscaled_lowres = transform_to_data_extent(
            sdata_pre_rasterize, coordinate_system=coordinate_system, maintain_positioning=False, **parameter_lowres
        )
        for image_name in sdata_images.images.keys():
            downscaled_images_hires[image_name] = downscaled_hires[image_name]
            downscaled_images_lowres[image_name] = downscaled_lowres[image_name]

        if sdata_post_rasterize is None:
            sdata_post_rasterize = downscaled_hires

    adata = sdata_post_rasterize[table_name].copy()

    if len(downscaled_images_hires) > 0:
        for image_name in sdata_images.images.keys():
            if "spatial" not in adata.uns:
                adata.uns["spatial"] = {}
            adata.uns["spatial"][image_name] = {}
            adata.uns["spatial"][image_name]["images"] = {
                "hires": downscaled_images_hires[image_name].data.compute().transpose([1, 2, 0]),
                "lowres": downscaled_images_lowres[image_name].data.compute().transpose([1, 2, 0]),
            }
            try:
                adata.uns["spatial"][image_name]["scalefactors"] = {
                    "tissue_hires_scalef": 1.0,
                    "tissue_lowres_scalef": DOWNSCALED_LOWRES_LENGTH / DOWNSCALED_HIRES_LENGTH,
                    "spot_diameter_fullres": sdata_post_rasterize.shapes[region_name]["radius"].iloc[0] * 2,
                }
            except KeyError:
                pass

    adata.obsm["spatial"] = (
        get_centroids(sdata_post_rasterize[region_name], coordinate_system=coordinate_system).compute().values
    )
    return adata


def align_xenium_img(
    sdata: SpatialData,
    img: np.ndarray,
    align_matrix: np.ndarray,
    library_id: str = "histology_img",
    img_max_size: Optional[int] = None
) -> sc.AnnData:
    if (None == img_max_size):
        img_max_size = max(img.shape[:2])
    assert (img_max_size <= max(img.shape[:2])), "ERROR: Incorrect `downsample size`."

    sdata.images[library_id] = Image2DModel.parse(
        img, dims=("y", "x", "c"),
        transformations={
            "global": 
            Affine(align_matrix, input_axes=("x", "y"), output_axes=("x", "y"))
        }
    )
    return to_legacy_anndata(sdata, img_max_size=img_max_size, table_name="table")


def align_xenium_img_from_path(
    sdata_path: str,
    img_path: str,
    align_file_path: str,
    library_id: str = "histology_img",
    img_max_size: Optional[int] = 15000
) -> sc.AnnData:
    if (".zarr" == sdata_path[-5:]):
        sdata = sd.read_zarr(sdata_path)
    else:
        sdata = sd_io.xenium(sdata_path)
    img = tifffile.imread(img_path, level=0, aszarr=False)
    align_matrix = pd.read_csv(align_file_path, header=None).to_numpy()
    return align_xenium_img(sdata, img, align_matrix, library_id, img_max_size)
