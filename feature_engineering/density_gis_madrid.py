## load in required packages ####
import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import CRS
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from itertools import chain
import pickle
from shapely.geometry import Polygon, MultiPolygon


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

fp='C:/Users/peter/Documents/projects/MSCA_data/madrid/EDM2018/ZonificacionZT1259-shp/ZonificacionZT1259.shp'
gdf=gpd.read_file(fp)
gdf.to_crs(crs0,inplace=True)

fp2='C:/Users/peter/Documents/projects/MSCA_data/madrid/EDM2018/ZonificacionZT208-shp/ZonificacionZT208.shp'
gdf2=gpd.read_file(fp2)
gdf2.to_crs(crs0,inplace=True)
gdf2['Municipality']= [elem.split('-')[0] for elem in gdf2.ZT208]
gdf2['Municipality']=gdf2['Municipality'].map(lambda x: x.replace('+d','ó').replace('+s','é').replace('+r','ó'))

gdf['area']=gdf.area*1e-6 # in km2
gdf2['area']=gdf2.area*1e-6

gdf.rename(columns={'ZT1259':'geo_unit_highres'},inplace=True)
gdf2.rename(columns={'CD_ZT208':'geo_unit'},inplace=True)

df=pd.DataFrame(gdf.drop(columns='geometry'))
df2=pd.DataFrame(gdf2.drop(columns='geometry'))

# load in census section shapefiles, 
# # from https://www.madrid.es/portales/munimadrid/es/Inicio/Ayuntamiento/Estadistica/Areas-de-informacion-estadistica/Territorio--climatologia-y-medio-ambiente/Territorio/Cartografia?vgnextfmt=detNavegacion&vgnextoid=aa9309789246c210VgnVCM2000000c205a0aRCRD&vgnextchannel=e59b40ebd232a210VgnVCM1000000b205a0aRCRD
# fp='C:/Users/peter/Documents/projects/MSCA_data/madrid/Secciones2017/Secciones.shp'

# from here, for 2018 https://www.ine.es/ss/Satellite?c=Page&p=1259952026632&pagename=ProductosYServicios%2FPYSLayout&cid=1259952026632&L=1
# this covers all of the autonomous community of Madrid
fp='../../MSCA_data/madrid/seccionado_2018/SECC_CE_20180101.shp'
sec_gdf=gpd.read_file(fp)
sec_gdf.to_crs(crs0,inplace=True)

# load in population by census sections
# from http://www-2.munimadrid.es/TSE6/control/seleccionDatosBarrio
# for file in os.listdir('../../MSCA_data/madrid/Population2018/'):
#     #print(file)
#     p=pd.read_excel('../../MSCA_data/madrid/Population2018/' + file,skiprows=4)
#     p.dropna(inplace=True)
#     if file=='Arganzuela.xls':
#         pop=p.copy()
#     else:
#         pop=pd.concat([pop,p])
# pop.drop(columns='Edad',inplace=True)
# pop.rename(columns={'Sección':'CODSECCION','Total':'Population'},inplace=True)

pop_cs_2018=pd.read_csv('../../MSCA_data/madrid/2801.csv',sep=';',dtype={'Sexo':'str','Sección':'str','Edad (grupos quinquenales)':'str','Total':'float'},thousands='.')

pop_cs_2018=pop_cs_2018.loc[(pop_cs_2018['Edad (grupos quinquenales)']=='Total') & (pop_cs_2018['Sexo']=='Ambos Sexos'),:]
pop_cs_2018.drop(columns=['Sexo','Edad (grupos quinquenales)'],inplace=True)
pop_cs_2018.rename(columns={'Sección':'CUSEC','Total':'Population'},inplace=True)

print('total population census sections is ', + pop_cs_2018.loc[pop_cs_2018['CUSEC']!='TOTAL','Population'].sum())

# set the center of the census sections as the shapefile geometry 
sec_gdf['center']=sec_gdf.centroid
sec_gdf.rename(columns={'geometry':'polygon_sec'},inplace=True)
sec_gdf.set_geometry('center',inplace=True)
sec_df=pd.DataFrame(sec_gdf.drop(columns=['polygon_sec','center']))

# spatial join the highres survey shapefile with census sections population data 

# join the section dataframe and the low-res sector (ZT208) shapefiles
join3=gpd.sjoin(sec_gdf,gdf2,how='left',predicate='within')

sec_sect=join3[['geo_unit','CUSEC']].drop_duplicates()

if sec_sect['CUSEC'].value_counts().max()>1:
    print('census section mapped to more than one ZT208 zone')
    sys.exit()

# add in population by census section

#join3=join3.merge(pop_cs_2018)

# merge population with the low-res sectors and census section mapping
sec_sect_pop=sec_sect.merge(pop_cs_2018)
# calculate total population by low-res survey sector
#sec_pop=join3.groupby('CD_ZT208')['Population'].sum().to_frame().reset_index()
sec_pop=sec_sect_pop.groupby('geo_unit')['Population'].sum().to_frame().reset_index()

# mergre the sum populations into the low-res gdf2
gdf2=gdf2.merge(sec_pop)

# # then calculate density
gdf2['Density']=gdf2['Population']/gdf2['area']


# make (geo)dataframe of the population by section (zt208)
# pop_208=gdf2.merge(sec_pop)
# pop_208['Density']=pop_208['Population']/pop_208['area']
# pop_208_df=pd.DataFrame(pop_208.drop(columns='geometry'))

# fig, ax = plt.subplots(figsize=(9, 12))
# ax.add_artist(ScaleBar(1))
#pop_208.plot(ax=ax,column='Density',cmap='Blues')
#gdf2.plot(ax=ax,column='Density',cmap='Blues')

print('total population in the 208 sectors ', + gdf2['Population'].sum())

###

# join the section dataframe and the -res (ZT1259) shapefiles
join4=gpd.sjoin(sec_gdf,gdf,how='left',predicate='within')

zone_sect=join4[['geo_unit_highres','CUSEC']].drop_duplicates()

if zone_sect['CUSEC'].value_counts().max()>1:
    print('census section mapped to more than one ZT1259 zone')
    sys.exit()
# add in population by census section
# join4=join4.merge(pop)
#join4=join4.merge(pop_cs_2018)

# merge population with the high-res zones and census section mapping
zone_sect_pop=zone_sect.merge(pop_cs_2018)
# calculate total population by low-res survey sector
zone_pop=zone_sect_pop.groupby('geo_unit_highres')['Population'].sum().to_frame().reset_index()

# geo concordances
sec_sect=join3[['geo_unit','CUSEC']].drop_duplicates().dropna()
zone_sect=join4[['CD_ZT1259','geo_unit_highres','CUSEC']].drop_duplicates().dropna()
zone_sec_sect=zone_sect.merge(sec_sect,on='CUSEC')
# better to use a sjoin between gdf and gdf2
#zone_sec=zone_sec_sect.drop(columns=['CD_ZT1259', 'CUSEC']).drop_duplicates()

# calculate population density in the high res survey zones
sec_gdf['area']=sec_gdf['polygon_sec'].area*1e-6
sec_gdf_pop=sec_gdf.merge(pop_cs_2018)
sec_gdf_pop['Density']=round(sec_gdf_pop['Population']/sec_gdf_pop['area'])
sec_gdf_pop.set_geometry('polygon_sec',inplace=True)

gdf0=gdf.copy()
gdf0.rename(columns={'geometry':'polygon_1259'},inplace=True)
gdf0['center']=gdf0['polygon_1259'].centroid
gdf0.set_geometry('center',inplace=True)

sec_gdf_pop0=sec_gdf_pop.loc[:,('CUSEC','NMUN','polygon_sec','area','Population','Density')]
sec_gdf_pop0['center']=sec_gdf_pop0.centroid
sec_gdf_pop0.set_geometry('center',inplace=True)

gdfj=gpd.sjoin(gdf0.loc[:,('CD_ZT1259', 'geo_unit_highres','polygon_1259', 'area', 'center')],sec_gdf_pop.loc[:,('CUSEC','NMUN','polygon_sec','area','Population','Density')],how='left',predicate='within')
gdfj.set_geometry('polygon_1259',inplace=True)
dfj=pd.DataFrame(gdfj.drop(columns=['center','polygon_1259']))
gdfj2=gpd.sjoin(sec_gdf_pop0,gdf.loc[:,('CD_ZT1259', 'geo_unit_highres','geometry', 'area')],how='left',predicate='within')
gdfj2.set_geometry('polygon_sec',inplace=True)
dfj2=pd.DataFrame(gdfj2.drop(columns=['center','polygon_sec']))

# see which high res zones were missed out due to not having a census section centroid within their geometry
miss=gdfj.loc[~gdfj['CD_ZT1259'].isin(gdfj2['CD_ZT1259']),'CD_ZT1259'].sort_values().reset_index(drop=True)
gdf_miss=gdf.loc[:,('CD_ZT1259','geo_unit_highres','geometry','area')].merge(gdfj.loc[gdfj['CD_ZT1259'].isin(miss),('CD_ZT1259','geo_unit_highres','Density')].sort_values(by='CD_ZT1259')).sort_values(by='CD_ZT1259')

zone_pop=gdfj2.groupby('CD_ZT1259')['Population'].sum().to_frame().reset_index()
gdf_hi=gdf.loc[:,('CD_ZT1259','geo_unit_highres','geometry','area')].merge(zone_pop).sort_values(by='CD_ZT1259').reset_index(drop=True)
gdf_hi['Density']=round(gdf_hi['Population']/gdf_hi['area'])
gdf_hi.drop(columns='Population',inplace=True)
gdf_hi=pd.concat([gdf_hi,gdf_miss]).sort_values(by='CD_ZT1259').reset_index(drop=True)

# get the concordance bw high and low res geounits
conc=gpd.sjoin(gdf0.loc[:,('CD_ZT1259', 'geo_unit_highres', 'center')],gdf2.loc[:,('geo_unit','geometry')],how='left',predicate='within')
conc=conc.loc[:,('geo_unit','geo_unit_highres','CD_ZT1259')].drop_duplicates().sort_values(by=['geo_unit','geo_unit_highres']).reset_index(drop=True)
# add in the multipolygon high-res unti '810' which is incorrectly allocated to geo_unit 73 (which is not included in our city definition) instead of geo_unit 106
conc.loc[conc['CD_ZT1259']==810,'geo_unit']=106.0

# merge in the low res geo_unit to allow downselecting later.
gdf_hi=gdf_hi.merge(conc)
gdf_hi=gdf_hi.loc[:,('CD_ZT1259', 'geo_unit_highres','geo_unit', 'geometry', 'area', 'Density',)]

gdf2['Density']=gdf2['Population']/gdf2['area']
gdf_lo=gdf2.loc[:,('geo_unit','Municipality', 'geometry', 'area', 'Density','Population')]

# downselect to desired municipalities
ring=['Madrid','Alcorcón', 'Leganés', 'Getafe', 'Móstoles', 'Fuenlabrada', 'Coslada', 'Alcobendas', 'Pozuelo de Alarcon', 'San Fernando de Henares', # inner ring
'Torrejon de Ardoz','Parla','Alcala de Henares','Las Rozas de Madrid','San Sebastian de los Reyes','Rivas', 'Majadahonda'] # additional municips

gdf_lo=gdf_lo.loc[gdf_lo['Municipality'].isin(ring),:]
gdf_hi=gdf_hi.loc[gdf_hi['geo_unit'].isin(gdf_lo['geo_unit']),:]


# now make the mixed res gdf
size_thresh=10
large=gdf2.loc[gdf2['area']>size_thresh,'geo_unit']

sub=gdf_hi.loc[gdf_hi['geo_unit'].isin(large),:]
sub=sub.loc[:,('geo_unit_highres','geometry','area','Density')]
sub.rename(columns={'geo_unit_highres':'geocode'},inplace=True)

gdf_mix=gdf_lo.loc[~gdf_lo['geo_unit'].isin(large),('geo_unit','geometry','area','Density')]
gdf_mix.rename(columns={'geo_unit':'geocode'},inplace=True)
gdf_mix['geocode']=gdf_mix['geocode'].astype(int).astype(str)

print('area distribution of retained large geo-units, ')
print(gdf_mix['area'].describe())
print('area distribution of smaller geo-units, ')
print(sub['area'].describe())
# make the concatenated gdf
gdf_mix=gpd.GeoDataFrame(pd.concat([gdf_mix,sub], ignore_index=True))

# plot each gdf and save the plot

fig, ax = plt.subplots(figsize=(10, 10))
plt.title("Aggregated (blue) and higher resolution (red) geo-units:  Madrid") 
ax.add_artist(ScaleBar(1))
gdf_mix.plot(ax=ax,edgecolor='black')
sub.plot(ax=ax, color='red',alpha=0.8,edgecolor='black')
plt.savefig('../outputs/density_geounits/Madrid_mixed.png',facecolor='w')

fig, ax = plt.subplots(figsize=(10, 10))
plt.title("Aggregated geo-units:  Madrid") 
ax.add_artist(ScaleBar(1))
gdf_lo.plot(ax=ax,edgecolor='black')
plt.savefig('../outputs/density_geounits/Madrid_low.png',facecolor='w')

fig, ax = plt.subplots(figsize=(10, 10))
plt.title("High resolutin geo-units:  Madrid") 
ax.add_artist(ScaleBar(1))
gdf_hi.plot(ax=ax,edgecolor='black',color='red')
plt.savefig('../outputs/density_geounits/Madrid_high.png',facecolor='w')

# create and save the city boundary
uuall=gdf_lo.unary_union
g1=Polygon(uuall.geoms[0].exterior)
g2=Polygon(uuall.geoms[1].exterior)
mp=MultiPolygon([g1,g2])
#boundary=gpd.GeoDataFrame(geometry=[gdf_lo['geometry'].unary_union], crs=crs0)
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
# insert
#gdf_hi['geo_unit_highres']=gdf_hi['geo_unit_highres'].astype('str').map(lambda x: x.replace('.','').replace(' ',''))
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