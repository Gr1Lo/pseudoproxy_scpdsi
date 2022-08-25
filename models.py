import keras
from keras import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
import tensorflow as tf
from sklearn.utils import shuffle
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from matplotlib import colors
import numpy as np
from matplotlib import pyplot as plt
import scipy
from sklearn.model_selection import train_test_split

np.random.seed(123)
tf.random.set_seed(1234)




def train_and_test(trsgi, labels, p_v, keep_order = False, m_mask=None):
  np.random.seed(123)

  '''
  Разделение выборки на тренировочную и тестовую
  trsgi - значения предикторов,
  labels - предсказываемые значения,
  p_v - доля тестовой выборки,
  keep_order - использовать для теста первые записи (сохранять порядок) 
  '''

  if m_mask is None:
    nums = np.ones(len(trsgi))
    nums[:int(len(trsgi)*p_v)] = 0
    if ~keep_order:
      np.random.shuffle(nums)

    mask = 1 == nums
    mask = np.array(mask)
  else:
    mask = m_mask

  train_trsgi = trsgi[mask]
  train_labels = labels[mask]
  test_trsgi = trsgi[~mask]
  test_labels = labels[~mask]

  return train_trsgi, train_labels, test_trsgi, test_labels

#функция формирования тренировочного набора для RNN
def make_x_y(ts, data):
    """
    Parameters
    ts : число шагов для RNN
    data : numpy array с предикторами
    x - наборы предикторов, в том числе и с предыдущих шагов
    y - предикторы только с действующего шага
    """
    x, y = [], []
    offset = 0
    for i in data:
        if offset < len(data)-ts:
            x.append(data[offset:ts+offset])
            y.append(data[ts+offset])
            offset += 1

    return np.array(x), np.array(y)
  
  
def m_loss_func_weight(l_evfs, use_w):
    '''
    Кастомная функция потерь
    '''
    def loss(y_true, y_pred):

        evfs = l_evfs
        if use_w==False:
          evfs = np.ones(len(evfs))

        Yhat = y_pred * evfs
        Yhat0 = y_true * evfs

        loss0 = tf.math.abs(tf.math.subtract(Yhat,Yhat0))
        loss0 = tf.math.reduce_mean(loss0,axis=1) 

        return loss0

    return loss






def m_loss_func_weight_f(l_evfs, l_eofs, l_eigvals, l_pca, scale_type=2):
    '''
    Кастомная функция потерь
    '''
    def loss(y_true, y_pred):

        evfs = l_evfs
        eofs = l_eofs
        eigvals = l_eigvals
        pca = l_pca

        pcs = y_pred
        pcs0 = y_true

        if scale_type == 2:
          eofs = eofs[0:len(eofs)] / np.sqrt(eigvals[0:len(eofs)])[:, np.newaxis]
          pcs = y_pred[:, 0:len(eofs)] / np.sqrt(eigvals[0:len(eofs)])
          pcs0= y_true[:, 0:len(eofs)] / np.sqrt(eigvals[0:len(eofs)])

        if scale_type == 1:
          eofs = eofs[0:len(eofs)] * np.sqrt(eigvals[0:len(eofs)])[:, np.newaxis]
          pcs = y_pred[:, 0:len(eofs)] * np.sqrt(eigvals[0:len(eofs)])
          pcs0 = y_true[:, 0:len(eofs)] * np.sqrt(eigvals[0:len(eofs)])

        '''Yhat = np.dot(pcs, eofs.to_numpy())
        Yhat0 = np.dot(pcs0, eofs.to_numpy())'''

        '''Yhat = tf.reshape(tf.matmul(pcs, eofs.to_numpy()),-1)
        Yhat0 = tf.reshape(tf.matmul(pcs0, eofs.to_numpy()),-1)'''

        Yhat = tf.matmul(pcs, eofs.to_numpy())
        Yhat0 = tf.matmul(pcs0, eofs.to_numpy())
        

        '''Yhat = pca._scaler.inverse_transform(Yhat)
        Yhat0 = pca._scaler.inverse_transform(Yhat0)'''

        loss0 = tf.abs(tf.subtract(Yhat,Yhat0))

        #print(tf.boolean_mask(loss0, tf.math.is_finite(loss0[0]), axis=1))
        
        #print(tf.boolean_mask(loss0, mask, axis=1),axis=1)
        #masked = tf.boolean_mask(loss0, tf.math.is_finite(loss0[0]), axis=1)
        loss0 = tf.reduce_sum(loss0,axis=1)
        #print(loss0)
        
        
        

        return loss0

    return loss




def corr_loss(x, y):    
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return 1 - (r_num / r_den)


'''def corr_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = tf.reduce_mean(x, axis=1, keepdims=True)
    my = tf.reduce_mean(y, axis=1, keepdims=True)
    xm, ym = x - mx, y - my
    t1_norm = tf.nn.l2_normalize(xm, axis = 1)
    t2_norm = tf.nn.l2_normalize(ym, axis = 1)
    cosine = tf.compat.v1.losses.cosine_distance(t1_norm, t2_norm, axis = )
    return cosine'''

def simp_net_regression_1(trsgi_values, resp, ttl, model, shuffle_b, evfs = None, 
                          eofs = None, eigvals = None, pca = None, scale_type=2,
                          use_w=False, min_delta = 0.0001):

    '''
    Запуск обучения модели регрессии
    trsgi_values - набор значений, полученных по ДКХ
    resp - набор предсказываемых значений
    ttl - название графика, описывающего ход обучения
    model - модель, сформированная через get_model_*
    shuffle_b - перемешивание очередности датасета
    evfs - доля объясненной диспесии каждой EOF
    use_w - использовать evfs в качестве весов перед расчетом функции потерь
    min_delta - пороговое значение изменение функции потерь при обучении
    '''

    np.random.seed(123)
    tf.random.set_seed(1234)

    trsgi = trsgi_values[:]
    all_arr = resp[:]

    trsgi_values = np.asarray(trsgi_values)
    all_arr = np.asarray(all_arr)
    
    if use_w=='full':
      model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  run_eagerly=True,
                  loss = m_loss_func_weight_f(l_evfs=evfs,
                                              l_eofs = eofs, 
                                              l_eigvals = eigvals, 
                                              l_pca = pca, 
                                              scale_type=scale_type))
    elif use_w=='corr':
      model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
                  run_eagerly=True,
                  loss = corr_loss)
    else:
      model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
                  run_eagerly=True,
                  loss = m_loss_func_weight(l_evfs=evfs,use_w=use_w))
    

    if shuffle_b:
      trsgi_values, all_arr = shuffle(trsgi_values, all_arr)
      callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=min_delta)
      v_s = 0.2
    
    else:
      trsgi_values, y = make_x_y(1, trsgi_values)
      callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001)
      v_s = 0.2
      
    #x_train, x_val, y_train, y_val = train_test_split(trsgi_values, all_arr, test_size=0.2, shuffle=shuffle_b)
    history = model.fit(trsgi_values,
                            all_arr,
                            verbose=0,
                            epochs=200, 
                            batch_size = 10,
                            shuffle=shuffle_b,
                            validation_split = v_s,
                            callbacks=[callback]
                            )

    return model, history


def get_model_regression_1(n_inputs, n_outputs, use_drop = False, use_batch_norm = False):

  '''
  Описание сети для задачи регрессии
  n_inputs - число предикторов,
  n_outputs - количество предсказываемых значений, по умоляанию 1
  use_drop - параметр, отвечающий за рандомное отключение доли нейронов (30%)
  use_batch_norm - параметр, отвечающий за использование batch-нормализации
  '''

  model = Sequential()
  dr_v = 0.5
  use_batch_norm = True

  model.add(Dense(50, input_dim=n_inputs, kernel_initializer='normal', activation='linear',
                  kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.L2(1e-4),
                  activity_regularizer=regularizers.L2(1e-5)))
  '''if use_batch_norm == True:
    model.add(BatchNormalization())'''
  if use_drop == True:
    model.add(Dropout(dr_v))


  model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
  return model

def get_model_regression_RNN(n_inputs, n_outputs = 1, use_drop = False, use_batch_norm = False,
                             inp_shp=None):

  '''
  Описание рекурсивной сети для задачи регрессии
  n_inputs - число предикторов,
  n_outputs - количество предсказываемых значений, по умоляанию 1
  use_drop - параметр, отвечающий за рандомное отключение доли нейронов (30%)
  use_batch_norm - параметр, отвечающий за использование batch-нормализации,
  inp_shp - размаер входных данных
  '''


  model = Sequential()
  rnn_l = tf.keras.layers.SimpleRNN(50, input_shape=(inp_shp[1], inp_shp[2]), recurrent_dropout=0.1, 
                           return_sequences=True)
  '''lstm = keras.layers.LSTM(50, input_shape=(inp_shp[1], inp_shp[2]), recurrent_dropout=0.1, 
                           return_sequences=True)'''

  model.add(rnn_l)
  '''if use_batch_norm == True:
    model.add(BatchNormalization())'''
  if use_drop == True:
    model.add(Dropout(0.20))

  #model.add(keras.layers.RepeatVector(inp_shp[1]))
  # decoder layer
  #model.add(keras.layers.LSTM(100, activation='relu', return_sequences=True))
  model.add(keras.layers.TimeDistributed(Dense(n_outputs)))

  return model


def nse(targets,predictions):
    return 1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(predictions))**2))

def d_index(targets,predictions):
    return 1-(np.sum((targets-predictions)**2)/np.sum((np.abs(predictions-np.mean(targets))+np.abs(targets-np.mean(targets)))**2))
    




def run_model(pcs_CMIP6_1, vsl_1000_pc, 
              evfs_CMIP6_1, eofs_CMIP6_1, eigvals_CMIP6_1, pca_CMIP6_1, scale_type=2,
              type_m='NN', use_w=False, m_mask = None):
      '''
      Запуск обучения моделей
      pcs_CMIP6_1 - значения главных компонент scpdsi
      vsl_1000_pc - значения главных компонент trsgi
      evfs_CMIP6_1 - массив с долей объясненных дисперсий каждой компонентой,
      type_m - вариант модели:
          'NN' - стнадартная нейронная сеть
          'RNN' - реккурентная нейронная сеть
          'lm' - линейниая модель
      use_w - расчитывать функцию потерь с учетом evfs_CMIP6_1
      Возвращает:
      inverse_te_l - тестовые значения главных компонент scpdsi
      inverse_est - модельная оценка главных компонент scpdsi
      '''

      # create scaler
      scaler = StandardScaler()
      scaler_t = StandardScaler()
      scaler_vsl_1000 = StandardScaler()
      scaler_t_vsl_1000 = StandardScaler()

      # fit and transform in one step

      normalized_vsl_1000 = scaler_vsl_1000.fit_transform(pcs_CMIP6_1) #pcs_CMIP6_1
      vsl_1000_pc_normalized = scaler_t_vsl_1000.fit_transform(vsl_1000_pc) #vsl_1000_pc

      target = normalized_vsl_1000
      n_inputs = vsl_1000_pc_normalized.shape[1]

      if type_m=='NN':
        model = get_model_regression_1(n_inputs,
                                      n_outputs = pcs_CMIP6_1.shape[1],
                                      use_drop = True,
                                      use_batch_norm = True)

        tr_t, tr_l, te_t, te_l = train_and_test(vsl_1000_pc_normalized, target, 0.2, keep_order=False, m_mask=m_mask
                                                )

        '''model, history = simp_net_regression_1(tr_t, tr_l, 
                                              'test', model, 
                                              shuffle_b=True,
                                              evfs = evfs_CMIP6_1, 
                                              use_w=use_w)'''

        model, history = simp_net_regression_1(tr_t, tr_l, 
                                              'test', model, 
                                              shuffle_b=True,
                                              evfs = evfs_CMIP6_1, 
                                              eofs = eofs_CMIP6_1, 
                                              eigvals = eigvals_CMIP6_1, 
                                              pca = pca_CMIP6_1,
                                              use_w=use_w)
        
        plt.figure(figsize = (19,10))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['loss', 'val_loss'], loc='upper left')
        plt.show()
        

        score = model.evaluate(te_t, te_l, verbose=0)
        print(score)
        inverse_te_l = scaler_vsl_1000.inverse_transform(te_l)
        est = model.predict(te_t)
        inverse_est = scaler_vsl_1000.inverse_transform(est)

      
      elif type_m=='lm':
        tr_t, tr_l, te_t, te_l = train_and_test(vsl_1000_pc_normalized, target, 0.2, keep_order=False, m_mask=m_mask)
        model_0 = sm.OLS(tr_l,tr_t)

        model = model_0.fit()

        inverse_te_l = scaler_vsl_1000.inverse_transform(te_l)
        est = model.predict(te_t)
        inverse_est = scaler_vsl_1000.inverse_transform(est)

      elif type_m=='arlm':
        #target1000, target999 = make_x_y(1, target)

        tr_t, tr_l, te_t, te_l = train_and_test(vsl_1000_pc_normalized, target, 0.2, keep_order=False, m_mask=m_mask)
        target1000, target999 = make_x_y(1, tr_l)
        print(target999.shape,target1000[:,0,:].shape)
        model_0 = sm.OLS(target999,target1000[:-1][:,0,:])
        model = model_0.fit()

        target1000, target999 = make_x_y(1, te_l)
        inverse_te_l = scaler_vsl_1000.inverse_transform(target999)
        est = model.predict(target1000[:-1][:,0,:])
        inverse_est = scaler_vsl_1000.inverse_transform(est)

      elif type_m=='RNN':
        trsgi_values, y = make_x_y(1, vsl_1000_pc_normalized)

        inp_shp = trsgi_values.shape
        model = get_model_regression_RNN(n_inputs,
                                        n_outputs = target.shape[1],
                                        use_drop = True,
                                        use_batch_norm = True,
                                        inp_shp=inp_shp)
                    
        tr_t, tr_l, te_t, te_l = train_and_test(vsl_1000_pc_normalized, normalized_vsl_1000, 0.2, keep_order=True, m_mask=m_mask)

        model, history = simp_net_regression_1(tr_t, tr_l, 
                                              'test', model, shuffle_b = False,
                                              use_w=use_w,
                                              evfs = evfs_CMIP6_1, min_delta = 0.01)
        
        plt.figure(figsize = (19,10))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['loss', 'val_loss'], loc='upper left')
        plt.show()
        
        te_t, y = make_x_y(1, te_t)
        inverse_te_l = scaler_vsl_1000.inverse_transform(te_l)
        est = model.predict(te_t)
        inverse_est = scaler_vsl_1000.inverse_transform(est[:,0,:])

      return inverse_te_l, inverse_est



def rev_diff(y_pred, y_true, eofs, eigvals, pca, ds_n, ttl, p_type='diff', scale_type = 2, orig_pcs = False):
       
        '''
        Visual assessment of prediction results:
        y_pred - predicted components values, 
        y_true - test scpdsi values scpdsi, 
        eofs - transformation matrix, 
        eigvals - coefficients applied to eigenvectors that give the vectors their length or magnitude, 
        pca - returned from eof_an() object, 
        ds_n - 3d array for retrieving shape, 
        ttl - name of model, 
        p_type - verification metrics
                  'mae' - mean absolute error
                  'corr' - Spearman's Rank correlation coefficient
                  'd' - index of agreement “d”
                  'nse' - Nash-Sutcliffe Efficiency
        scale_type - converting loadings to components for reverse operation
        '''

        eg = np.sqrt(eigvals)
        if orig_pcs:
          eg = np.sqrt(eigvals[0:len(eofs)])

        if scale_type == 2:
          eofs = eofs / eg[:, np.newaxis]
          pcs = y_pred[:, 0:y_pred.shape[1]] / eg
          pcs0= y_true[:, 0:y_pred.shape[1]] / eg

        if scale_type == 1:
          eofs = eofs * eg[:, np.newaxis]
          pcs = y_pred[:, 0:y_pred.shape[1]] * eg
          pcs0 = y_true[:, 0:y_pred.shape[1]] * eg

        if scale_type == 0:
          eofs = eofs
          pcs = y_pred[:, 0:y_pred.shape[1]]
          pcs0 = y_true[:, 0:y_pred.shape[1]]

        Yhat = np.dot(pcs, eofs)
        Yhat = pca._scaler.inverse_transform(Yhat)
        u = Yhat

        if orig_pcs:
          u0 = y_true
        else:
          Yhat0 = np.dot(pcs0, eofs)
          Yhat0 = pca._scaler.inverse_transform(Yhat0)
          u0 = Yhat0

        if p_type=='corr':
          coor_ar = []
          for i in range(u0.shape[1]):
            i1 = u[:,i]
            i0 = u0[:,i]
            if ~np.isnan(i0[0]):
              corr2 = scipy.stats.spearmanr(i0,i1)[0]
              coor_ar.append(corr2)
            else:
              coor_ar.append(np.nan)

          loss0 = np.array(coor_ar)
          ttl_str = " average Spearman's Rank correlation coefficient = "
          vmin = -1
          vmax = 1

        elif p_type == 'mae':
          loss0 = np.abs(u - u0)
          loss0 = np.where(loss0>50, np.nan, loss0)
          loss0 = np.mean(loss0,axis=0)
          ttl_str = '; mean absolute error = '
          vmin = 0
          vmax = 8

        elif p_type == 'nse':
          nse_ar = []
          for i in range(u0.shape[1]):
            i1 = u[:,i]
            i0 = u0[:,i]
            if ~np.isnan(i0[0]):
              nse2 = nse(i0,i1)
              nse_ar.append(nse2)
            else:
              nse_ar.append(np.nan)

          loss0 = np.array(nse_ar)
          ttl_str = '; Nash-Sutcliff-Efficiency = '
          vmin = 0
          vmax = 1

        elif p_type == 'd':
          nse_ar = []
          for i in range(u0.shape[1]):
            i1 = u[:,i]
            i0 = u0[:,i]
            if ~np.isnan(i0[0]):
              nse2 = d_index(i0,i1)
              nse_ar.append(nse2)
            else:
              nse_ar.append(np.nan)

          loss0 = np.array(nse_ar)
          ttl_str = '; d-index = '
          vmin = 0
          vmax = 1

        new = np.reshape(loss0, (-1, ds_n.shape[2]))
        plt.figure(figsize = (19,10))
        im = plt.imshow(new[::-1], interpolation='none',
                        vmin=vmin, vmax=vmax,cmap='jet')

        cbar = plt.colorbar(im,
                            orientation='vertical')
        plt.axis('off')
        plt.tight_layout()

        loss0 = np.nanmean(loss0)

        plt.title(ttl + ttl_str + str(round(loss0,3)),fontsize=20)
        plt.show()

        return loss0
