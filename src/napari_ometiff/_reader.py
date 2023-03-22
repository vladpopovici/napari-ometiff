"""
OME TIFF image reader.
"""
import numpy as np
import tifffile
import zarr
import dask.array as da
import pyometiff
from contextlib import redirect_stdout

from dask.cache import Cache
cache = Cache(4e9)  # Leverage 4 gigabytes of memory
cache.register()


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        path = path[0]

    if any([path.endswith(".ome.tif"), path.endswith(".ome.tiff"),
            path.endswith(".ome_tif"), path.endswith(".ome_tiff")]):
        return reader_function

    return None


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    path = path if isinstance(path, str) else path[0]

    tif = tifffile.TiffFile(str(path))
    if not tif.is_ome:
        raise RuntimeError("only OME TIFF files are accepted.")
    
    ome = pyometiff.OMETIFFReader(fpath=str(path))
    ome.omexml_string = tif.ome_metadata  # work-around a bug in pyometiff
    with redirect_stdout(None): # to avoid messages about not found keys
        info = ome.parse_metadata(tif.ome_metadata)
    
    # axes: identify singletons and remove from axes annotation
    axes = ''.join(
        [a for a in list(info['DimOrder']) if info['Size' + a] > 1]
    ).lower() # -> something like 'cyx'

    # axes of interest
    axes_order = {
        'c': list(axes).index('c'), 
        'y': list(axes).index('y'),
        'x': list(axes).index('x'),
        } # just X, Y, C

    tif = tifffile.imread(path, aszarr=True)
    if not tif.is_multiscales:
        raise RuntimeError("only pyramidal images are accepted.")

    if info['PhysicalSizeXUnit'] == 'µm':
        unit_multiplier_x = 1.0 
    elif info['PhysicalSizeXUnit'] == 'mm': 
        unit_multiplier_x = 1000.0
    else:
        unit_multiplier_x = 1.0
        raise RuntimeWarning('unknown unit for resolution (X)')
    
    if info['PhysicalSizeYUnit'] == 'µm':
        unit_multiplier_y = 1.0 
    elif info['PhysicalSizeYUnit'] == 'mm': 
        unit_multiplier_y = 1000.0
    else:
        unit_multiplier_y = 1.0
        raise RuntimeWarning('unknown unit for resolution (Y)')

    base_mpp_x = unit_multiplier_x * info['PhysicalSizeX']  # in microns per pixel
    base_mpp_y = unit_multiplier_y * info['PhysicalSizeY']

    pyramid = None
    with zarr.open(tif, mode='r') as z:
        n_levels = len(list(z.array_keys()))
        pyramid = [
            da.moveaxis( 
                da.from_zarr(z[i]), [axes_order['y'], axes_order['x'], axes_order['c']], [0, 1, 2] 
            ) 
            for i in range(n_levels)
            ]
    metadata = {
        'rgb': True,
        #'channel_axis': 2,
        'contrast_limits': (0, 255),
        'multiscale': True,
    }

    layer_type = "image"  

    return [(pyramid, metadata, layer_type)]
