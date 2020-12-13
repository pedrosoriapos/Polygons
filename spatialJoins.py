import numpy as np
import pandas as pd
import itertools
import time

df_dub = df[['latitude', 'longitude]]
df_dub = df_dub.drop_duplicates()

min_lat_dub = df_dub.latitude.min()
max_lat_dub = df_dub.latitude.max()
min_long_dub = df_dub.longitude.min()
max_long_dub = df_dub.longitude.max()

# Creating grid of 280 x 280 meters
lats_dub = np.arange(min_lat_dub, max_lat_dub, .28/111)
longs_dub = np.arange(min_long_dub, max_long_dub, .28/111)
lats_dub_prim = [[lats_dub[i], lats_dub[i + 1]] for i in np.arange(0, len(lats_dub) - 1, 1)]
longs_dub_prim = [[longs_dub[i], longs_dub[i + 1]] for i in np.arange(0, len(longs_dub) - 1, 1)]

# DataFrame containing min lat, max lat, min long and max long of all the polygons of the grid
df_dub_pol = pd.DataFrame([list(np.array(i).reshape((1,4))[0] for i in list(itertools.product(lats_dub_prim, longs_dub_prim))])

# Spatial Joins
dict_pip_dubai = {}
# creating an index for all the polygons
indexes = list(np.arange(0, df_dub_pol.shape[0], 1000)) + [df_dub_pol.shape[0]]

for i in range(len(indexes) - 1):
  # Operation point in polygon through apply. As many columns as points there are. Value in ith column will be true if that point falls within that polygon
  df_pip_dubai = df_dub_pol.iloc[indexes[i]:indexes[i+1]].apply(lambda x: (df_dub.latitude >= x[0]) &\
                                                                          (df_dub.latitude <= x[1]) &\
                                                                          (df_dub.longitude >= x[2]) &\
                                                                          (df_dub.longitude <= x[3]), axis = 1)
  # Dictionary: keys: (from what index to what index, polygon wise). values: (indexes of what points fall within the polygon)                                                                      
  dict_pip_dubai[(indexes[i], indexes[i+1])] = df_pip_dubai.apply(lambda x: np.where(x)[0], axis = 1)
#list of pandas dataframes. 2 columns: [index of polygon, indexes of points]
pip_dubai = [pd.DataFrame([np.arange(j[0], j[1], 1), [list(i) for i in list(dict_pip_dubai[j])]]).T for j in dict_pip_dubai.keys()]
# concatenating the list of pandas data frames
df_pip_dubai = pip_dubai[0]
for i in range(len(pip_dubai) - 1):
  df_pip_dubai = pd.concat([df_pip_dubai, pip_dubai[i+1]], axis = 0)
#Number of points within that polygon
df_pip_dubai['n_points'] = df_pip_dubai[1].apply(lambda x: len(x))
 
#repeating the index of the polygon as many times as points there are within this polygon
index_pip = list(np.concatenate([list(np.repeat(i, j)) for i, j in zip(df_pip_dubai[0], df_pip_dubai.n_points)]))
#concatenating the points within a polygon
points_pip = list(np.concatenate([i for i in df_pip_dubai[1]]))

#Consolidated dataframe mapping all the points to the polygons, index wise
df_pip_dubai_prim = pd.DataFrame((index_pip, points_pip)).T
df_pip_dubai_prim.columns = ['polygon', 'point']

#Creating an index in df_dub to map coordinates to index points
df_dub['index'] = np.arange(0, df_dub.shape[0], 1)

#Merging. Mapping index of points to coordinates
df_pip_dubai_prim = df_pip_dubai_prim.merge(df_dub, left_on = 'point', right_on = 'index', how = 'left')

#Creating an index for df_dub_pol to map index to coordinates
df_dub_pol.columns = ['min_lat', 'max_lat', 'min_long', 'max_long']
df_dub_pol['index'] = np.arange(0, df_dub_pol.shape[0], 1)

#Merging. Mapping index of polygons to coordinates
df_pip_dubai_prim = df_pip_dubai_prim.merge(df_dub_pol, left_on = 'polygon', right_on = 'index', how = 'left')

#Calculating centroid of polygons
df_pip_dubai_prim['polygon_latitude'] = df_pip_dubai_prim.apply(lambda x: (x['min_lat'] + x['max_lat']) / 2, axis = 1)
df_pip_dubai_prim['polygon_longitude'] = df_pip_dubai_prim.apply(lambda x: (x['min_long'] + x['max_long']) / 2, axis = 1)

#Final DataFrame
df_pip_dubai_prim = df_pip_dubai_prim[['polygon', 'latitude', 'longitude', 'polygon_latitude', 'polygon_longitude']]


