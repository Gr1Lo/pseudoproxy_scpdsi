from osgeo.gdalconst import *
from osgeo import gdal, osr
import os
import numpy as np
import pandas as pd
import gc
from sklearn.cross_decomposition import PLSRegression

from sys import getsizeof

import rasterio as rs
from rasterio.warp import calculate_default_transform, reproject, Resampling
from matplotlib import pyplot
from pyproj import Proj, Transformer


def reproj_raster(orig_path, var_summer, var_name, lats_tmp, lons_tmp, swap_minus = False):
    st_year = var_summer.index.values.min()
    '''var_summer_s = var_summer.set_index(['year']+[coor0_tmp[0],coor0_tmp[1]])
    var_summer_s = var_summer_s.unstack([coor0_tmp[0],coor0_tmp[1]]).to_numpy()'''
    var_summer_s = var_summer.to_numpy()
    var_summer_s = var_summer_s.reshape([var_summer_s.shape[0],len(lats_tmp),len(lons_tmp)])
    var_summer_s = np.nan_to_num(var_summer_s, nan=-99)

    src = gdal.Open(orig_path)
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    rows = src.RasterYSize
    cols = src.RasterXSize

    driver = gdal.GetDriverByName('GTiff')
    n_path = os.path.dirname(orig_path) + '/' + 'temp.tif'
    outDs = driver.Create(n_path, cols, rows, var_summer_s.shape[0], GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    outDs.SetProjection( srs.ExportToWkt() )

    for i in range(1,var_summer_s.shape[0]+1):
        outBand = outDs.GetRasterBand(i)
        fliped = np.flip(var_summer_s[i-1], 0)
        outBand.WriteArray(fliped)
        outBand.FlushCache()
        outBand.SetNoDataValue(-99)

    outDs.SetGeoTransform(src.GetGeoTransform())
    outDs = None
    
    ulx = min(lons_tmp.values)
    uly = max(lats_tmp.values)
    lrx = max(lons_tmp.values)
    lry = min(lats_tmp.values)
    
    if swap_minus:
        l_arr = lons_tmp.values
        l_arr = np.where(l_arr<0, l_arr+360, l_arr)
        ulx = min(l_arr)
        lrx = max(l_arr)

    src = gdal.Translate(os.path.dirname(orig_path) + '/' + 'cropped.tif', n_path, 
                   projWin=[ulx, uly, lrx, lry])
    ds = None

    src = gdal.Open(os.path.dirname(orig_path) + '/' + 'cropped.tif')
    rows = src.RasterYSize
    cols = src.RasterXSize

    #NAD 1983 Equidistant Conic North America (https://epsg.io/102010)
    dst_crs = 'ESRI:102010'
    with rs.open(os.path.dirname(orig_path) + '/' + 'cropped.tif') as src:
      transform, width, height = calculate_default_transform(
          src.crs, dst_crs, src.width, src.height, *src.bounds)
      kwargs = src.meta.copy()
      kwargs.update({
          'crs': dst_crs,
          'transform': transform,
          'width': width,
          'height': height
      })

      output_raster = os.path.dirname(n_path) + '/' + 'temp_rep.tif'
      with rs.open(output_raster, 'w', **kwargs) as dst:
          for i in range(1, src.count + 1):
                reproject(
                    source=rs.band(src, i),
                    destination=rs.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

    src = gdal.Open(output_raster)
    rast_count = src.RasterCount
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    rows = src.RasterYSize
    cols = src.RasterXSize
    lrx = ulx + (rows * xres)
    lry = uly + (cols * yres)

    iBand1 = src.GetRasterBand(1)
    i_array1 = np.array(iBand1.ReadAsArray())
    tab_arr = np.empty([rast_count, (i_array1.shape[0])*(i_array1.shape[1]), 4])#, dtype='float16')
    #tab_arr = np.nan
    for i in range (1, rast_count+1):
        print(i)
        ind = 0
        iBand = src.GetRasterBand(i)
        i_array = np.array(iBand.ReadAsArray())
        i_array = np.round(i_array,2)

        '''coords = np.argwhere(i_array != -99)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)'''
        i_array = np.where(i_array==-99, np.nan, i_array)
        cou_cells = 0
        for rows in range(i_array.shape[0]):
          for cols in range(i_array.shape[1]):
            crop_x = ulx + (cols + 0.5) * xres
            crop_y = uly + (rows + 0.5) * yres
            m_val = i_array[rows,cols]
            tab_arr[i-1,cou_cells] = np.array([int(st_year+i-1), crop_y, crop_x, m_val])
            cou_cells += 1

        del i_array
        del iBand   
        gc.collect()

    #tab_arr = tab_arr[0,:,:]
    tab_arr = tab_arr.reshape(tab_arr.shape[0]*tab_arr.shape[1], tab_arr.shape[2])
    res_tab = pd.DataFrame(data=tab_arr, columns=['year', 'lat', 'lon',var_name])

    del tab_arr
    gc.collect()

    res_tab0 = res_tab[res_tab[var_name]==0]
    t_t = res_tab0.groupby(['lat','lon']).size().reset_index()

    #change all0 to nan
    for index_t, row_t in t_t.iterrows():
      check_lat, check_lon = row_t['lat'], row_t['lon']
      cell = res_tab[var_name][(res_tab['lat']==check_lat) & (res_tab['lon']==check_lon)]
      values = cell.values
      if all(v == 0 for v in values):
        res_tab.loc[(res_tab['lat']==check_lat) & (res_tab['lon']==check_lon), var_name] = np.nan

    return res_tab              


def reproj_points(df, lat_var = 'lat', lon_var = 'lon'):
  m_df = df.copy()
  transformer = Transformer.from_crs("EPSG:4326", 'ESRI:102010')

  for i, row in m_df.iterrows():
    x1,y1 = m_df.at[i,lat_var], m_df.at[i,lon_var]
    x2,y2 = transformer.transform(x1,y1)
    m_df.at[i, lon_var] = x2
    m_df.at[i, lat_var] = y2

  return m_df
