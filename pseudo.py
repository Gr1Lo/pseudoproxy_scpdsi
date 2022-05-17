
import os
import numpy as np
import pandas as pd
import xarray as xr
from pyEOF import *

t_df = pd.read_csv('lat_lon_table.txt')


def r_netCDF(f_path, min_lon = -145, min_lat = 14, max_lon = -52, max_lat = 71, swap = 0):
    '''
    Формирование таблицы по годам из netCDF с индексами scpdsi
    '''

    ds = xr.open_dataset(f_path)["scpdsi"]

    coor = [] 
    for key in ds.coords.keys():
      coor.append(key)

    #Выбор территории анализа
    if coor[1] == 'latitude':
      mask_lon = (ds.longitude >= min_lon) & (ds.longitude <= max_lon)
      mask_lat = (ds.latitude >= min_lat) & (ds.latitude <= max_lat)
    else:
      mask_lon = (ds.lon >= min_lon) & (ds.lon <= max_lon)
      mask_lat = (ds.lat >= min_lat) & (ds.lat <= max_lat)

    ds_n = ds.where(mask_lon & mask_lat, drop=True)

    nd = pd.to_datetime(ds_n.indexes['time'], errors = 'coerce')
    #datetimeindex = nd.to_datetimeindex()
    ds_n['time'] = nd

    df_nn = ds_n.to_dataframe().reset_index()

    #Используется информация только по летним месяцам
    print(type(df_nn['time'][0]))
    df_nn0 = df_nn[(df_nn['time'].dt.month < 9)&(df_nn['time'].dt.month > 5)]
    grouped_df = df_nn0.groupby([coor[1], coor[0] ,df_nn0['time'].dt.year])
    mean_df = grouped_df.mean()
    mean_df = mean_df.reset_index()

    mean_df['time_n'] = pd.to_datetime(mean_df.time.astype(str), format='%Y')
    del mean_df['time']
    mean_df = mean_df.rename(columns={'time_n': 'time'})

    if swap == 0:
      mean_df = mean_df[['time', coor[1], coor[0], 'scpdsi']]
      df_data = get_time_space(mean_df, time_dim = "time", lumped_space_dims = [coor[1],coor[0]])
    else:
      mean_df = mean_df[['time', coor[0], coor[1], 'scpdsi']]
      df_data = get_time_space(mean_df, time_dim = "time", lumped_space_dims = [coor[0],coor[1]])

    return df_data

netcdf_df = r_netCDF('scPDSI_scpdsi_q.nc')
print(netcdf_df)

