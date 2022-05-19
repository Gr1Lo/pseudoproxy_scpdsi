import netCDF4
import numpy as np
import os
import sys
import pandas as pd

in_dir = sys.argv[1]
res_file = sys.argv[2]

if os.path.isfile(res_file):
    os.remove(res_file)

directory = os.fsencode(in_dir)
    
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".txt"):
        c_val = 0
        with open(in_dir + '/' + filename) as txtfile:
            for line in txtfile:
                if 'trsgi' in line:
                    c_val = 1

                if 'Northernmost_Latitude' in line:
                    in_lat = float(line.split(': ')[1])

                if 'Easternmost_Longitude' in line:
                    in_lon = float(line.split(': ')[1])

                if 'Elevation' in line:
                    elev = line.split(': ')[1].replace('\n', '')

                    


        if c_val == 1:
            
          df = pd.read_table(in_dir + '/' + filename, comment='#', header = 0, delim_whitespace=True)

          m_l = [filename.replace(".txt", ""),
          filename.replace(".txt", ""),
          'tree',
          elev,
          in_lat,
          in_lon,
          df["year"].values,
          'AD',
          'trsgi',
          None,
          df["trsgi"].values,
          'TRW']

          year_trsgi = pd.DataFrame(columns = ['paleoData_pages2kID',
            'dataSetName',
            'archiveType',
            'geo_meanElev',
            'geo_meanLat',
            'geo_meanLon',
            'year',
            'yearUnits',
            'paleoData_variableName',
            'paleoData_units',
            'paleoData_values',
            'paleoData_proxy',
            'netCDF_meanLat',
            'netCDF_meanLon'], data = [m_l])

          print(filename)


          if os.path.isfile(res_file):
            year_trsgi.to_csv(res_file, mode='a', header = False)
          else:
            year_trsgi.to_csv(res_file)  


