# script to calculat population density by municipality in Wien
# last update Peter Berrill Nov 25 2022

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from pyproj import CRS
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import pickle
from citymob import import_csv_w_wkt_to_gdf

# inputs:
# Austrian municipality (gemeinden) shapefiles and population counts
# City center point estimate for Wien

# outputs:
# city boundary shapefile
# population density by municipality
# summary stats on population, area, and area distribution of gemeinden (or other spatial units)
# pickled dictionary of Wien municipalities

crs0=3035

# load in shapefile of Austrian municipalities
# https://www.data.gv.at/katalog/dataset/stat_gliederung-osterreichs-in-gemeinden14f53
fp='C:/Users/peter/Documents/projects/city_mobility/shapefiles/austria/STATISTIK_AUSTRIA_GEM_20130101.shp'
gdf=gpd.read_file(fp)
gdf.to_crs(crs0,inplace=True)

gdf['ID_int']=gdf['ID'].astype(int)
gdf['area']=gdf.area*1e-6

# isolate the Wien municipalities
wien=gdf.loc[gdf['ID_int']>=90000,:]
# isloate the municipalities in lower austria 
la=gdf.loc[(gdf['ID_int']<40000) & (gdf['ID_int']>29999) ,:]

# load in the city centers shapefiles
centers=import_csv_w_wkt_to_gdf('../source/citycenters/centers.csv',crs=4326)
centers.to_crs(crs0,inplace=True)
# create a buffer of arbitrary size
city_center_buffer=centers.copy()
city_center_buffer['geometry']=city_center_buffer['geometry'].buffer(12500)  
# then use the buffer around Wien to identify further municipalities which are within the given buffer range
join=gpd.sjoin(city_center_buffer.loc[city_center_buffer['City']=='Wien',],la,how='right',predicate='intersects')

wien_la=join.dropna() #nb, optionally we can manually alter this, e.g. to retain Maria Enzerdorf and exclude e.g. Aderklaa
me=join.loc[join['ID']=='31716',]
wien_la=pd.concat([wien_la,me],ignore_index=True) # add Maria Enzerdorf
wien_la=wien_la.loc[wien_la['ID']!='30801',] # drop Aderklaa
wien['state']='Wien'
wien_la['state']='LowerAustria'

# still debatable whether we should include Achau, Groeser Enzersdorf, or Himberg.

# create a new Wien + extents gdf which includes those nearby municipalities in lower Austria
wien_ext=pd.concat([wien,wien_la.loc[:,('ID', 'NAME', 'geometry', 'ID_int', 'area', 'state')]],ignore_index=True)

print('Area: ', wien_ext['area'].sum())

# read in Austria population by municipality, source: https://www.data.gv.at/katalog/dataset/1dd64998-6836-3871-ac89-443f742bdc68
pop=pd.read_csv('../../MSCA_data/Austria_Population/OGD_f0743_VZ_HIS_GEM_3.csv',sep=';')
# rename columns
pop.columns=['ID','Year','Population']
# extract data only from the census year 2011
pop=pop.loc[pop['Year']=='H88-15',]
# make a municipality column based on the last 5 digits of the ID
pop['gem']=pop['ID'].map(lambda x: x[-5:])

# create a dictionary to map the old municipality ids to the new ones which took effect after the dissolution of the Bezirk Wien-Umgebung https://de.wikipedia.org/wiki/Bezirk_Wien-Umgebung
# this is not a comprehensive list, only those caught by the buffer around Wien are included
dict={'32403':'31949', # Gablits
'32404':'31235', # Gerasdorf bei Wien
'32406':'30732', # Himberg
'32408':'32144', # Klosterneuburg
'32409':'30734', # lanzendorf
'32410':'30735', # Leopoldsdorf
'32411':'30736',  # Maria-Lanzendorf	
'32412':'31950', # Mauerbach
'32416':'31952', # purkersdorf
'32419':'30740', # Schwechat
'32424':'30741' # Zwölfaxing
}
# create a new ID column containing the updated IDs
wien_ext.loc[:,'ID2']=wien_ext.loc[:,'ID'].replace(dict)

# merge the shapefile with the population data and calculate densities
wien_ext=wien_ext.merge(pop.drop(columns='ID'),left_on='ID2',right_on='gem')
wien_ext['Density']=wien_ext['Population']/wien_ext['area']

city_center_buffer2=centers.loc[city_center_buffer['City']=='Wien',].copy()
city_center_buffer2['geometry']=city_center_buffer2['geometry'].buffer(8000) 

# print some summary results
print('Population: ',wien_ext['Population'].sum())
print('Area: ',wien_ext['area'].sum())
print('Density: ',wien_ext['Population'].sum()/wien_ext['area'].sum())
# plot each gdf and save the plot
city='Wien'
fig, ax = plt.subplots(figsize=(10,10))
plt.title("Geo-units by federal state:  Wien") 
ax.add_artist(ScaleBar(1))
wien_ext.plot(ax=ax,column='state',legend=True)
plt.savefig('../outputs/density_geounits/Wien.png',facecolor='w')

# create and save the city boundary
boundary=gpd.GeoDataFrame(geometry=[wien_ext['geometry'].unary_union], crs=crs0)
boundary['crs']=crs0
boundary.to_csv('../outputs/city_boundaries/' + city + '.csv',index=False)

# save the gdfs and dict
wien_ext=wien_ext.loc[:,('ID2','geometry','NAME','state','Population','area','Density')]
wien_ext.rename(columns={'ID2':'geo_unit'},inplace=True )
wien_ext.sort_values(by='geo_unit',inplace=True)
wien_ext.to_csv('../outputs/density_geounits/' + city + '_pop_density.csv',index=False)

# create and save some summary stats
area_lores=pd.DataFrame(wien_ext['area'].describe()).reset_index()
sums=pd.DataFrame(wien_ext[['area','Population']].sum()).reset_index()
writer = pd.ExcelWriter('../outputs/density_geounits/summary_stats_' + city + '.xlsx', engine='openpyxl')

# include all the dfs/sheets here, and then save
area_lores.to_excel(writer, sheet_name='area_lores',index=False)
sums.to_excel(writer, sheet_name='area_pop_sum',index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()
writer.close()

# save dictionary of Wien municipalities
city_plz={}
city_plz.update( {'Wien' : wien_ext['geo_unit'].values})
with open('../dictionaries/city_postcode_AT.pkl', 'wb') as f:
    pickle.dump(city_plz, f)

print('Finished extracting density and shapefiles for ' + city)