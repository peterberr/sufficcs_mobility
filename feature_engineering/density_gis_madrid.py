# script to calculate population density by spatial unit in Madrid
# last update Peter Berrill Oct 21 2023

## load in required packages ####
import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import CRS
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar
from itertools import chain
import pickle
from shapely.geometry import Polygon, MultiPolygon
import sys

# inputs:
# Spatial unit shapefiles, data directly from survey, low-res (ZT208) and high res (ZT1259)
# census section shapefiles for 2018, source: (https://www.ine.es/ss/Satellite?c=Page&p=1259952026632&pagename=ProductosYServicios%2FPYSLayout&cid=1259952026632&L=1)
# census section in Madrid population (for 2018), source: https://www.ine.es/jaxi/Datos.htm?path=/t20/e245/p07/a2018/&file=2801.px

# outputs:
# city boundary shapefile
# population density by spatial units, at different levels of aggregation (low-res, mix-res, high-res)
# summary stats on population, area, and area distribution of gemeinden (or other spatial units)
# figures showing spatial units for each resolution
# pickled dictionaries to translate the geocodes from the survey to the mixed-res ids needed to merge with the geospatial data

crs0=3035
city='Madrid'

fp='../../MSCA_data/madrid/EDM2018/ZonificacionZT1259-shp/ZonificacionZT1259.shp'
gdf=gpd.read_file(fp)
gdf.to_crs(crs0,inplace=True)

fp2='../../MSCA_data/madrid/EDM2018/ZonificacionZT208-shp/ZonificacionZT208.shp'
gdf2=gpd.read_file(fp2)
gdf2.to_crs(crs0,inplace=True)
gdf2['Municipality']= [elem.split('-')[0] for elem in gdf2.ZT208]
gdf2['Municipality']=gdf2['Municipality'].map(lambda x: x.replace('+d','ó').replace('+s','é').replace('+r','ó'))

gdf['gu_hires_area']=gdf.area*1e-6 # in km2
gdf2['geo_unit_area']=gdf2.area*1e-6

gdf.rename(columns={'ZT1259':'geo_unit_highres'},inplace=True)
gdf2.rename(columns={'CD_ZT208':'geo_unit'},inplace=True)

# df=pd.DataFrame(gdf.drop(columns='geometry'))
# df2=pd.DataFrame(gdf2.drop(columns='geometry'))

# load in census section shapefiles,
# from here, for 2018 https://www.ine.es/ss/Satellite?c=Page&p=1259952026632&pagename=ProductosYServicios%2FPYSLayout&cid=1259952026632&L=1
# this covers all of the autonomous community of Madrid
fp='../../MSCA_data/madrid/seccionado_2018/SECC_CE_20180101.shp'
sec_gdf=gpd.read_file(fp)
sec_gdf.to_crs(crs0,inplace=True)

# load in population by census sections
# from https://www.ine.es/jaxi/Datos.htm?path=/t20/e245/p07/a2018/&file=2801.px
pop_cs_2018=pd.read_csv('../../MSCA_data/madrid/2801.csv',sep=';',dtype={'Sexo':'str','Sección':'str','Edad (grupos quinquenales)':'str','Total':'float'},thousands='.')

pop_cs_2018=pop_cs_2018.loc[(pop_cs_2018['Edad (grupos quinquenales)']=='Total') & (pop_cs_2018['Sexo']=='Ambos Sexos'),:]
pop_cs_2018.drop(columns=['Sexo','Edad (grupos quinquenales)'],inplace=True)
pop_cs_2018.rename(columns={'Sección':'CUSEC','Total':'Population'},inplace=True)

print('total population census sections is ', + pop_cs_2018.loc[pop_cs_2018['CUSEC']!='TOTAL','Population'].sum())

gdf4=gdf.overlay(gdf2,how='intersection')
# in all cases i've seen in gdf4, only one of the gu makes up effectively all of the gu_hires area, i.e. they are identical. 
# gu_hires 018-001 is a special case bc it is a multipolygon which doesn't contain its own centroid. 
# therefore doing a spatial join based on centroids will incorrectly allocate 018-001 to gu 164, rather than its correct match gu 163.
gdf4['geo_unit']=gdf4['geo_unit'].astype(int).astype(str)
gdf4['area_over']=gdf4.area

# this is a valid way to get the direct 1-1 matching of high-res to low-res spatial units
gdf5=gdf4.sort_values(by='area_over',ascending=False)
gdf5.drop_duplicates(subset='geo_unit_highres',keep='first',inplace=True)
# the close area match shows that the dropped rows are insignificant negligible matches prob due to inadvertent geom overlaps
print('area included in gdf used for concordances of lo:hi res geounits',gdf5['area_over'].sum()/gdf4['area_over'].sum())

# use this gdf to create a concordance df bw low and highres 
conc=gdf5.loc[:,['geo_unit','geo_unit_highres','CD_ZT1259']]
conc['geo_unit']=conc['geo_unit'].astype(int)
conc.sort_values(by=['geo_unit','geo_unit_highres'],inplace=True,ascending=True)
# merge conc into the highres gdf
gdf=gdf.merge(conc)

# merge the pop data into the census section df
sec_gdf_pop=sec_gdf.loc[:,['CUSEC','NMUN','geometry']].merge(pop_cs_2018)
sec_gdf_pop['sec_area']=sec_gdf_pop.area*1e-6

# create overlay bw high-res gdf and census sections and population
over_hi = gdf.overlay(sec_gdf_pop, how='intersection')
over_hi['overhi_area']=over_hi.area*1e-6
over_hi['sec_density']=over_hi['Population']/over_hi['sec_area']
over_hi['area_share']=over_hi['overhi_area']/over_hi['sec_area']
over_hi['Pop_calc']=over_hi['area_share']*over_hi['Population']

over_hi_sums=over_hi.groupby('geo_unit_highres')['Pop_calc'].sum().to_frame().reset_index()
over_hi_sums=gpd.GeoDataFrame(over_hi_sums.merge(gdf.loc[:,['CD_ZT1259','geo_unit_highres','geo_unit','gu_hires_area','geometry']]))
over_hi_sums['Density']=over_hi_sums['Pop_calc']/over_hi_sums['gu_hires_area']
over_hi_sums.sort_values(by='CD_ZT1259',inplace=True)

# calculate population per low-res unit by summing the high-res
over_lo_sums=over_hi_sums.groupby('geo_unit')['Pop_calc'].sum().reset_index()
over_lo_sums.rename(columns={'Pop_calc':'Pop_calc_lo_sum'},inplace=True)
over_lo_sums=gpd.GeoDataFrame(over_lo_sums.merge(gdf2.loc[:,['geo_unit','geo_unit_area','Municipality','geometry']]))
over_lo_sums['Density']=over_lo_sums['Pop_calc_lo_sum']/over_lo_sums['geo_unit_area']

gdf_lo=over_lo_sums.copy()
gdf_hi=over_hi_sums.copy()

gdf_lo.rename(columns={'Pop_calc_lo_sum':'Population','geo_unit_area':'area'},inplace=True)
gdf_hi.rename(columns={'Pop_calc':'Population','gu_hires_area':'area'},inplace=True)

ring=['Madrid','Alcorcón', 'Leganés', 'Getafe', 'Móstoles', 'Fuenlabrada', 'Coslada', 'Alcobendas', 'Pozuelo de Alarcon', 'San Fernando de Henares', # inner ring
'Torrejon de Ardoz','Parla','Alcala de Henares','Las Rozas de Madrid','San Sebastian de los Reyes','Rivas', 'Majadahonda'] # additional municips

gdf_lo=gdf_lo.loc[gdf_lo['Municipality'].isin(ring),:]
gdf_hi=gdf_hi.loc[gdf_hi['geo_unit'].isin(gdf_lo['geo_unit']),:]

# now make the mixed res gdf
size_thresh=10
large=gdf_lo.loc[gdf_lo['area']>size_thresh,'geo_unit']

sub=gdf_hi.loc[gdf_hi['geo_unit'].isin(large),:]
sub=sub.loc[:,('geo_unit_highres','geometry','area','Population','Density')]
sub.rename(columns={'geo_unit_highres':'geocode'},inplace=True)
sub['source']='Small units'

gdf_mix=gdf_lo.loc[~gdf_lo['geo_unit'].isin(large),('geo_unit','geometry','area','Population','Density')]
gdf_mix.rename(columns={'geo_unit':'geocode'},inplace=True)
gdf_mix['geocode']=gdf_mix['geocode'].astype(int).astype(str)
gdf_mix['source']='Large units'
gdf_mix=gpd.GeoDataFrame(pd.concat([gdf_mix,sub], ignore_index=True))
print('Populations represented in low-res, high-res, and mixres density files, and original data:')
print(gdf_lo['Population'].sum())
print(gdf_hi['Population'].sum())
print(gdf_mix['Population'].sum())
print(sec_gdf_pop['Population'].sum())
# plot each gdf and save the plot

cmap = mpl.colors.ListedColormap(['#1f77b4', 'red'])
fig, ax = plt.subplots(figsize=(10, 10))
#plt.title("Aggregated (blue) and higher resolution (red) geo-units:  Madrid") 
# scale=ScaleBar(1,location='lower right')
# ax.add_artist(scale)
ax.set_title("Aggregated and higher resolution geo-units: " + city,size=16)
gdf_mix.plot(ax=ax,column='source',edgecolor='black',cmap=cmap,legend=True)
#plt.legend(loc='center right')
#sub.plot(ax=ax, color='red',alpha=0.8,edgecolor='black')
plt.axis('off');
plt.savefig('../outputs/density_geounits/Madrid_mixed.png',facecolor='w')

fig, ax = plt.subplots(figsize=(10, 10))
#plt.title("Aggregated geo-units:  Madrid") 
ax.add_artist(ScaleBar(1))
ax.set_title("Aggregated geo-units: Madrid",size=16)
gdf_lo.plot(ax=ax,edgecolor='black')
plt.axis('off');
plt.savefig('../outputs/density_geounits/Madrid_low.png',facecolor='w')

fig, ax = plt.subplots(figsize=(10, 10))
#plt.title("High resolution geo-units:  Madrid") 
ax.add_artist(ScaleBar(1))
ax.set_title("High resolution geo-units: Madrid",size=16)
gdf_hi.plot(ax=ax,edgecolor='black',color='red')
plt.axis('off');
plt.savefig('../outputs/density_geounits/Madrid_high.png',facecolor='w')

# create and save the city boundary
uuall=gdf_lo.unary_union
g1=Polygon(uuall.geoms[0].exterior)
g2=Polygon(uuall.geoms[1].exterior)
mp=MultiPolygon([g1,g2])

boundary=gpd.GeoDataFrame(geometry=[mp], crs=crs0)
boundary['crs']=crs0
boundary.to_csv('../outputs/city_boundaries/' + city + '.csv',index=False)

# save the gdfs and dict
gdf_lo['geo_unit']=gdf_lo['geo_unit'].astype(int).astype(str)
gdf_lo.sort_values(by='geo_unit',inplace=True)
gdf_lo.to_csv('../outputs/density_geounits/' + city + '_pop_density_lowres.csv',index=False)

#gdf_mix.sort_values(by='geocode',inplace=True)
gdf_mix.to_csv('../outputs/density_geounits/' + city + '_pop_density_mixres.csv',index=False)

gdf_hi.sort_values(by='geo_unit_highres',inplace=True)
gdf_hi['geo_unit']=gdf_hi['geo_unit'].astype(int).astype(str)
gdf_hi.to_csv('../outputs/density_geounits/' + city + '_pop_density_highres.csv',index=False)

# create and save dictionary to map high res to mixed res geocodes
long=pd.DataFrame(gdf_hi.loc[:,('geo_unit_highres','geo_unit')])
long['geo_unit']=long['geo_unit'].astype(int).astype(str)

zone_sec=long.set_index('geo_unit_highres').T.to_dict('records')[0]
with open('../dictionaries/' + city + '_zone_sec.pkl', 'wb') as f:
    pickle.dump(zone_sec, f)

long.loc[long['geo_unit_highres'].isin(sub['geocode']),'geo_unit']=long.loc[long['geo_unit_highres'].isin(sub['geocode']),'geo_unit_highres']

geo_dict=long.set_index('geo_unit_highres').T.to_dict('records')[0]
with open('../dictionaries/' + city + '_mixed_geocode.pkl', 'wb') as f:
    pickle.dump(geo_dict, f)

# create and save some summary stats

area_mixed=pd.DataFrame(gdf_mix['area'].describe()).reset_index()
area_hires=pd.DataFrame(gdf_hi['area'].describe()).reset_index()
area_lores=pd.DataFrame(gdf_lo['area'].describe()).reset_index()
sums=pd.DataFrame(gdf_lo[['area','Population']].sum()).reset_index()
sums['index'].replace({'Population':'population'},inplace=True)
sums=pd.concat([sums,pd.DataFrame([{'index':'density',0:sums.iloc[1,1]/sums.iloc[0,1]}])])
writer = pd.ExcelWriter('../outputs/density_geounits/summary_stats_' + city + '.xlsx', engine='openpyxl')

# include all the dfs/sheets here, and then save
area_mixed.to_excel(writer, sheet_name='area_mixres',index=False)
area_hires.to_excel(writer, sheet_name='area_hires',index=False)
area_lores.to_excel(writer, sheet_name='area_lores',index=False)
sums.to_excel(writer, sheet_name='area_pop_sum',index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()
writer.close()

print('Finished extracting density and shapefiles for ' + city)