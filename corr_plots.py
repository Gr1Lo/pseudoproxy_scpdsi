import numpy as np
import scipy
import pandas as pd
from matplotlib import pyplot as plt
import netCDF4 as nc4
import colorednoise as cn



def match_point_grid(df, 
                             grid_df, 
                             var_name,
                             tr_ret_lon, 
                             tr_ret_lat,
                             coor0,
                             min_lon = -145, 
                             min_lat = 14, 
                             max_lon = -52, 
                             max_lat = 71):
    lat_lon_list = []
    tab_arr = []
    cou = 0
    for fn in (df['file_name'].unique()):
      df_t = df[df['file_name']==fn]
      df_t = df_t[df_t['age'].isin(grid_df['year'].unique())]

      if len(df_t)>0: 
        cou+=1
        if ((df_t['lat'].values[0] >= min_lat) & (df_t['lat'].values[0] <= max_lat) 
        & (df_t['lon'].values[0] >= min_lon) & (df_t['lon'].values[0] <= max_lon)):

          tr_lat_ind = (np.abs(df_t['lat'].values[0] - tr_ret_lat.values)).argmin()
          tr_lon_ind = (np.abs(df_t['lon'].values[0] - tr_ret_lon.values)).argmin()

          if 1:#[tr_lat_ind, tr_lon_ind] not in lat_lon_list:

              print('Расчет для точки ' +str(cou) + ' с индексом ' + str(tr_lat_ind) + ' ' + str(tr_lon_ind))

              grid_df0 = grid_df[(grid_df[coor0[0]]==tr_ret_lon[tr_lon_ind]) 
              & (grid_df[coor0[1]]==tr_ret_lat[tr_lat_ind]) 
              & (grid_df['year'].isin(df_t['age'].unique()))]

              merge_df = grid_df0.merge(df_t, left_on='year', right_on='age')
              corr_s = scipy.stats.spearmanr(merge_df[var_name].values,merge_df['trsgi'].values)[0]
              lat_lon_list.append([tr_lat_ind, tr_lon_ind])

              tab_arr.append([df_t['lat'].values[0],df_t['lon'].values[0], corr_s, df_t['age'], df_t['trsgi']])

    corr_tab = pd.DataFrame(columns = ['geo_meanLat',
                'geo_meanLon',
                'corr',
                'ages',
                'trsgi'], data = np.array(tab_arr))
    
    return corr_tab
  
  
  
  
def plot_corr_points(corr_tab, nc_summer, lons, lats, ttl,
                    latbounds = [ 14 , 71 ],
                    lonbounds = [ -145 , -52 ],
                    vmin =-7, vmax=30):
    
    

    fig, axs = plt.subplots(figsize=(15, 10), nrows=2,ncols=2,gridspec_kw={'height_ratios': [20,1.5],'width_ratios': [20,1]},constrained_layout=True)
    pcm=axs[0][0].pcolormesh(lons,lats,np.mean(nc_summer, axis=0),cmap='viridis', vmin = vmin, vmax=vmax)
    axs[0][0].set_xlim(left=lonbounds[0], right=lonbounds[1])
    axs[0][0].set_ylim(bottom=latbounds[0], top=latbounds[1])
    axs[1][1].remove()

    scatter = axs[0][0].scatter(corr_tab['geo_meanLon'],corr_tab['geo_meanLat'], c=corr_tab['corr'], s=80, edgecolors='black', cmap = "magma")

    cbar0=fig.colorbar(scatter,cax=axs[0][1], extend='both', orientation='vertical', ticks=[-1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1])
    cbar0.set_label('corr_spearman', fontsize=20)
    cbar=fig.colorbar(pcm,cax=axs[1][0], extend='both', orientation='horizontal', ticks=range(-7, 30))

    fig.suptitle(ttl, fontsize=25)
    
    
    
    
def plot_pseudo_corr_real(new_df, lons, lats, vsl_pseudo, lat_lon_list_vsl, l_w):
    a = np.empty((len(lats),len(lons),))
    a[:] = np.nan

    for ite in range(len(new_df)):
            real_trsgi = new_df.iloc[ite]['trsgi'].values
            real_ages = new_df.iloc[ite]['ages'].values

            #поиск ближайших координат в сетке
            lat_ind = (np.abs(new_df['geo_meanLat'].values[ite] - lats.values)).argmin()
            lon_ind = (np.abs(new_df['geo_meanLon'].values[ite] - lons.values)).argmin()

            t_lat = np.array(lat_lon_list_vsl)[:,0]
            t_lon = np.array(lat_lon_list_vsl)[:,1]
            ind_m = np.argwhere(( t_lat== lat_ind) & ((t_lon == lon_ind)))[0][0]

            proxy_trsgi = vsl_pseudo[:,0,ind_m][real_ages-l_w]

            corr_s = scipy.stats.spearmanr(proxy_trsgi,real_trsgi)[0]

            if np.isnan(a[lat_ind,lon_ind]) or a[lat_ind,lon_ind] < corr_s:
              a[lat_ind,lon_ind]=corr_s

    fig, axs = plt.subplots(figsize=(15, 10), nrows=2,gridspec_kw={'height_ratios': [20,1.5]},constrained_layout=True)
    pcm=axs[0].pcolormesh(lons,lats,a,cmap='tab20b', vmin =-0.25, vmax=0.8)
    cbar=fig.colorbar(pcm,cax=axs[1], extend='both', orientation='horizontal', ticks=[0,0.1,0.2,0.5,1])
    cbar.set_label('corr_spearman', fontsize=20)
    fig.suptitle('Значения корреляции реальных и псведо- прокси', fontsize=25)

    
    
def match_point_grid_1000(vsl_1000, lat_lon_list_1000, lons, lats, st_year=850):
  t_arr = []
  cou_year = 0
  for i in vsl_1000:
    c_year = st_year + cou_year
    lat_lon = lat_lon_list_1000[cou_year]
    ages = list(range(st_year,st_year+1000))
    ages = pd.Series(ages)
    i_s = pd.Series(i)
    t_arr.append([lats.values[lat_lon[0]],lons.values[lat_lon[1]], ages, i_s])
    cou_year += 1

  df_vsl_1000 = pd.DataFrame(columns = ['geo_meanLat',
                  'geo_meanLon',
                  'ages',
                  'trsgi'], data = np.array(t_arr))
  
  return df_vsl_1000


def plot_corr(new_df, lons, lats, grid_df, ttl, corr = None, var_name = 'scpdsi'):
    pseudo_p_arr = []
    a = np.empty((len(lats),len(lons),))
    a[:] = np.nan

    for ite in range(len(new_df)):
            real_trsgi = new_df.iloc[ite]['trsgi'].values
            real_ages = new_df.iloc[ite]['ages'].values
            df0 = pd.DataFrame(columns = ['trsgi',
                  'ages'], data = np.array([real_trsgi, real_ages]).T)
            

            #поиск ближайших координат в сетке
            lat_ind = (np.abs(new_df['geo_meanLat'].values[ite] - lats.values)).argmin()
            lon_ind = (np.abs(new_df['geo_meanLon'].values[ite] - lons.values)).argmin()

            df1 = grid_df[(grid_df.lat == lats[lat_ind]) & (grid_df.lon == lons[lon_ind])]

            merge_df = df1.merge(df0, left_on='year', right_on='ages')

            #рандомные значения шума разного цвета
            wn = cn.powerlaw_psd_gaussian(0, len(merge_df[var_name].values))
            va_std = np.std(merge_df[var_name].values)


            if corr is None:
              pseudo_p = merge_df['trsgi'].values
            else:
              #стандартизация по snr по аналогии с Feng Zhu
              snr = np.abs(corr)/((1-corr**2)**0.5) 
              noise_std = va_std/snr
              wn_snr = noise_std * wn
              pseudo_p = merge_df[var_name].values + wn_snr

            corr_s = scipy.stats.spearmanr(merge_df[var_name].values,pseudo_p)[0]

            if ~np.isnan(corr_s):
              pseudo_p_arr.append(pseudo_p)

            if ~np.isnan(corr_s) & (np.isnan(a[lat_ind,lon_ind]) or a[lat_ind,lon_ind] < corr_s):
              a[lat_ind,lon_ind]=corr_s

    if corr is not None:
          ttl = ttl + '\n(corr = ' + str(corr) + ')'
    else:
          ttl = ttl + ' (vsl)' 

    fig, axs = plt.subplots(figsize=(15, 10), nrows=2,gridspec_kw={'height_ratios': [20,1.5]},constrained_layout=True)
    pcm=axs[0].pcolormesh(lons,lats,a,cmap='tab20b', vmin =-0.25, vmax=0.8)
    cbar=fig.colorbar(pcm,cax=axs[1], extend='both', orientation='horizontal', ticks=[-0.2,0,0.1,0.2,0.5, 0.75, 1])
    cbar.set_label('corr_spearman', fontsize=20)
    fig.suptitle(ttl, fontsize=25)

    return np.array(pseudo_p_arr)
  
  
