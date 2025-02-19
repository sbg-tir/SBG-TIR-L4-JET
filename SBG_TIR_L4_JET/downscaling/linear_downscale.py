from rasters import Raster
import rasters as rt

DEFAULT_UPSAMPLING = "average"
DEFAULT_DOWNSAMPLING = "linear"

def bias_correct(
        coarse_image: Raster,
        fine_image: Raster,
        upsampling: str = "average",
        downsampling: str = "linear",
        return_bias: bool = False):
    fine_geometry = fine_image.geometry
    coarse_geometry = coarse_image.geometry
    upsampled = fine_image.to_geometry(coarse_geometry, resampling=upsampling)
    bias_coarse = upsampled - coarse_image
    bias_fine = bias_coarse.to_geometry(fine_geometry, resampling=downsampling)
    bias_corrected_fine = fine_image - bias_fine

    if return_bias:
        return bias_corrected_fine, bias_fine
    else:
        return bias_corrected_fine

def linear_downscale(
        coarse_image: Raster,
        fine_image: Raster,
        upsampling: str = "average",
        downsampling: str = "cubic",
        use_gap_filling: bool = False,
        apply_scale: bool = True,
        apply_bias: bool = True,
        return_scale_and_bias: bool = False) -> Raster:
    if upsampling is None:
        upsampling = DEFAULT_UPSAMPLING

    if downsampling is None:
        downsampling = DEFAULT_DOWNSAMPLING

    coarse_geometry = coarse_image.geometry
    fine_geometry = fine_image.geometry
    upsampled = fine_image.to_geometry(coarse_geometry, resampling=upsampling)

    if apply_scale:
        scale_coarse = coarse_image / upsampled
        scale_coarse = rt.where(coarse_image == 0, 0, scale_coarse)
        scale_coarse = rt.where(upsampled == 0, 0, scale_coarse)
        scale_fine = scale_coarse.to_geometry(fine_geometry, resampling=downsampling)
        scale_corrected_fine = fine_image * scale_fine
        fine_image = scale_corrected_fine
    else:
        scale_fine = fine_image * 0 + 1

    if apply_bias:
        upsampled = fine_image.to_geometry(coarse_geometry, resampling=upsampling)
        bias_coarse = upsampled - coarse_image
        bias_fine = bias_coarse.to_geometry(fine_geometry, resampling=downsampling)
        bias_corrected_fine = fine_image - bias_fine
        fine_image = bias_corrected_fine
    else:
        bias_fine = fine_image * 0

    if use_gap_filling:
        gap_fill = coarse_image.to_geometry(fine_geometry, resampling=downsampling)
        fine_image = fine_image.fill(gap_fill)

    if return_scale_and_bias:
        fine_image["scale"] = scale_fine
        fine_image["bias"] = bias_fine

    return fine_image