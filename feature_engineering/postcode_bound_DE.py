# script to create boundary shapefiles and dictionaries of postcodes for German cities
# last update Peter Berrill Nov 25 2022 

import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import CRS
import pickle

# inputs:
# German official city boundary shapefiles, from the 2011 census. source:  https://www.zensus2011.de/EN/Media/Background_material/Background_material_node.html
# German city postcode shapefiles. source: https://www.suche-postleitzahl.org/downloads

# outputs:
# German city-postcode dictionaries, saved as pickle files
# custom German city boundary shapefiles, saved as csv files

crs0=3035
cities=['Berlin','Dresden','Düsseldorf','Frankfurt am Main','Kassel','Leipzig','Magdeburg','Potsdam']

# get the city boundaries from shapefile of German districts
dis=gpd.read_file('../shapefiles/VG250_1Jan2011_WGS84/VG250_Kreise.shp') # source https://www.zensus2011.de/EN/Media/Background_material/Background_material_node.html
dis.to_crs(3035,inplace=True)
dis.rename(columns={'GEN':'city_name'},inplace=True)

# get the official city boundaries according to the 2011 census
for city in cities:
    print(city)
    print('gdf_boundary' in locals())
    gdf_boundary0=dis.loc[(dis['city_name']==city) & (dis['DES']=='Kreisfreie Stadt'),('city_name','geometry')]

    if 'gdf_boundary' in locals():
        gdf_boundary=pd.concat([gdf_boundary,gdf_boundary0])
    else:
        gdf_boundary=gdf_boundary0

# create a 2km buffer around the cities, and set that as the geometry, in order to capture all postcodes in the spatial join below
gdf_boundary.rename(columns={'geometry':'boundary'},inplace=True)
gdf_boundary['boundary_buffer2km']=gdf_boundary['boundary'].buffer(2000)
gdf_boundary.set_geometry('boundary_buffer2km',inplace=True)

# read in postcodes gdf, these are the 5-digit postcode shapes downloaded from https://www.suche-postleitzahl.org/downloads on 8 June 2022. Original source is OSM
fp = "../shapefiles/plz-5stellig.shp/plz-5stellig.shp"
de_plz = gpd.read_file(fp)
de_plz.to_crs(crs0,inplace=True)
de_plz['area']=de_plz.area*1e-6

# extract only the city postcodes which are within the city boundaries (plus some buffer)
join = gpd.sjoin(de_plz,gdf_boundary, how="inner", predicate="within")
# the city of Madgeburg has an unusual geometry for postcode 39114, around the area of Pechau, the disconnected polygons are surrounded by parts of postcode 39217

# make a dictionary to store the postcodes for each city
city_plz=dict()
for city in cities:
    city_plz.update( {city : join.loc[join['city_name'] == city,'plz'].values})

# save post code list by city as pickel file, this excludes the added postcodes below.
city_plz_fp='../dictionaries/city_postcode_DE_basic.pkl'
a_file = open(city_plz_fp, "wb")
pickle.dump(city_plz, a_file)
a_file.close()

#create and save a city boundary csv shapefile for each city
for c in cities:
    city_boundary=gpd.GeoDataFrame(geometry=[de_plz.loc[de_plz['plz'].isin(city_plz[c]),'geometry'].unary_union],crs=crs0)
    city_boundary['crs']=crs0
    city_boundary.to_csv('../outputs/city_boundaries/' + c + '_basic.csv',index=False)

# Subjective addition of peripheral postcodes considered to be part of the city area ##

# postcodes are added if they are not too large (below 50km2 preferably), are not part of another city which is large enough in its own right (e.g. Potsdam from Berlin) to be considered a separate urban region, 
# and if it is sufficiently close to the city center (usually <20km), and do not have too low density (e.g. <500/km2), and preferably with mostly urbanised space. Finally, the set of postcodes should be contiguous.
# none of the summary metrics (e.g. mode share, daily distance) will be available based on residence for these postcodes, but they can be calculated by trip origin location

city_plz.update({
'Berlin':np.append(city_plz['Berlin'],['15366','14513','14612']),
'Dresden':np.append(city_plz['Dresden'],['01445','01705']),
'Düsseldorf':np.append(city_plz['Düsseldorf'],['40699','41460','40878','40721','40296','40789','40764','40880','40667']),
'Frankfurt am Main':np.append(city_plz['Frankfurt am Main'],['61118','65760','63263','61348','61352','65843']), 
'Kassel':np.append(city_plz['Kassel'],['34246','34225']),
'Leipzig':np.append(city_plz['Leipzig'],['04416']),
'Magdeburg':np.append(city_plz['Magdeburg'],['39179']),
#'Potsdam':np.append(city_plz['Potsdam'],['14513'])
})

# save post code list by city as pickel file
city_plz_fp='../dictionaries/city_postcode_DE.pkl'
a_file = open(city_plz_fp, "wb")
pickle.dump(city_plz, a_file)
a_file.close()
#create and save a city boundary csv shapefile for each city
for c in cities:
    city_boundary=gpd.GeoDataFrame(geometry=[de_plz.loc[de_plz['plz'].isin(city_plz[c]),'geometry'].unary_union],crs=crs0)
    city_boundary['crs']=crs0
    city_boundary.to_csv('../outputs/city_boundaries/' + c + '.csv',index=False)