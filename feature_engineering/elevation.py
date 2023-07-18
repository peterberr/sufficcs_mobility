# load in required packages
import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import CRS
import sys
import pickle
from citymob import import_csv_w_wkt_to_gdf
import rasterio

crs0= 3035 # crs for boundary 
cities_all=['Berlin','Dresden','Düsseldorf','Frankfurt am Main','Kassel','Leipzig','Magdeburg','Potsdam','Clermont','Dijon','Lille','Lyon','Montpellier','Nantes','Nimes','Paris','Toulouse','Madrid','Wien']

def elev(city):
    print('Elevation for '  + city)

    raster = rasterio.open('../../MSCA_data/EUDEM/Clips/' + city + '_EUDEM.tif')
    bld_dens=import_csv_w_wkt_to_gdf('../outputs/CenterSubcenter/' + city + '_dist.csv',crs=3035,geometry_col='wgt_center')
    coord_list = [(x, y) for x, y in zip(bld_dens["geometry"].x, bld_dens["geometry"].y)]
    bld_dens['elev'] = [x[0] for x in raster.sample(coord_list)]

    # a=np.array(bld_dens['elev'])
    # m=np.zeros((len(a), len(a)))
    # for x in range(len(a)):
    #     m[:,x]=[j-i for i, j in zip(np.repeat(a[x],len(a)), a[:])]

    # np.savetxt('../outputs/elevation/' + city + '.csv', m, delimiter=",")
    elevs_df=pd.DataFrame({'orig_geocode':np.repeat(bld_dens['geocode'],len(bld_dens)),
                      'des_geocode':np.tile(bld_dens['geocode'],len(bld_dens))})
    elevs_df['diff']=np.nan

    elevs_df['diff']=np.nan
    for i in bld_dens['geocode']:
        for j in  bld_dens['geocode']:
            elevs_df.loc[(elevs_df['orig_geocode']==i) & (elevs_df['des_geocode']==j),'diff']=bld_dens.loc[bld_dens['geocode']==j,'elev'].values[0]-bld_dens.loc[bld_dens['geocode']==i,'elev'].values[0]
    elevs_df.reset_index(inplace=True,drop=True)

    elevs_df.to_csv('../outputs/elevation/' + city + '_diff.csv',index=False)

#cities=pd.Series(cities_all)
cities=pd.Series(['Berlin','Dresden','Düsseldorf','Kassel','Leipzig','Magdeburg','Potsdam'])
cities.apply(elev) 