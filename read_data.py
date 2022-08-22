import os
import pandas as pd
import pickle
import xarray as xr
from sklearn import linear_model
import numpy as np
import nc_time_axis

def save_pickle(f_path, vari):
    with open(f_path, 'wb') as f:
        pickle.dump(vari, f)

def read_pickle(f_path):
    with open(f_path, 'rb') as f:
        df_test = pickle.load(f)
        return df_test
      
def r_execel(f_path, drop_val = 2):
    '''
    Формирование двумерного массива по годам из данных,
    полученных из ДКХ с заполнением пустых значений линейной регрессией
    f_path - путь до xlsx файла
    drop_val - число строчек, которые будут удалены с конца табилцы
    '''
    df = pd.read_excel(f_path, index_col=None)
    fn_list = df['file_name'].unique()

    trsgi_values = []
    for i in (range(1901, np.max(df['age'])+1)):
      one_year_arr = []
      print(i)
      df0 = df[df['age'] == i]
      for j in fn_list:
        re = df0[df0['file_name'] == j]['trsgi'].values
        if len(re)>0:
          one_year_arr.append(re[0])
        else:
          one_year_arr.append(None)

      trsgi_values.append(one_year_arr)

    df_trsgi_values = pd.DataFrame(data=trsgi_values)
    ind_list = []
    for index, val in df_trsgi_values.isna().sum().iteritems():
      if val < drop_val+1:
        ind_list.append(index)

    df_trsgi_values.drop(df_trsgi_values.tail(drop_val).index,inplace=True)
    arrX = df_trsgi_values[ind_list].to_numpy()

    m_list = []
    for i in range(len(df_trsgi_values.columns)):
      arrY = df_trsgi_values[i].to_numpy()
      ind_NONE = np.where(np.isnan(arrY))
      ind_not_NONE = np.where(~np.isnan(arrY))

      regr = linear_model.LinearRegression()
      regr.fit(arrX[ind_not_NONE], arrY[ind_not_NONE])
      if len(ind_NONE[0])>0:
        arrY[ind_NONE] = np.around(regr.predict(arrX[ind_NONE]),3)
      m_list.append(arrY)

    mat = np.array(m_list)
    res = np.transpose(mat)

    return res
