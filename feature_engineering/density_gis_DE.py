# script to calculat population density by postcode in German cities
# last update Peter Berrill Nov 25 2022

import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import CRS
import pickle

# inputs:
# German city postcode shapefiles and population counts
# German city-postcode dictionaries, loaded as pickle files

# outputs:
# population density by postcode
# summary stats on population, area, and area distribution of postcodes (or other spatial units)

cities=['Berlin','Dresden','Düsseldorf','Frankfurt am Main','Kassel','Leipzig','Magdeburg','Potsdam','Clermont','Dijon','Lille','Lyon','Montpellier','Nantes','Nimes','Paris','Toulouse','Madrid','Wien']
countries=['Germany','Germany','Germany','Germany','Germany','Germany','Germany','Germany','France','France','France','France','France','France','France','France','France','Spain','Austria']
crs0=3035

# german population and shapefile of postcodes is from here https://www.suche-postleitzahl.org/plz-karte-erstellen
# issue is that it is not dated, it is probably recent, but not matched with the survey data which is 2018. maybe i can get official data for 2018 from https://www.statistikportal.de/de or https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/_inhalt.html
# more info available here too https://blog.suche-postleitzahl.org/post/132153774751/einwohnerzahl-auf-plz-gebiete-abbilden

# german population by postcode
pa=pd.read_csv('C:/Users/peter/Documents/projects/MSCA_data/GermanPopulationPostcode/plz-5stellig-daten.csv')
pa['postcode']=pa['plz'].map(lambda x: str(x).zfill(5))

# germann postcodes gdf
fp="C:/Users/peter/Documents/projects/city_mobility/shapefiles/plz-5stellig.shp/plz-5stellig.shp"
de_plz = gpd.read_file(fp)
de_plz.to_crs(crs0,inplace=True)
# keep only the desired info from de_plz and pa
de_plz=de_plz.loc[:,('plz','geometry')].merge(pa.drop(columns='plz'),left_on='plz',right_on='postcode')
de_plz['area']=de_plz.area*1e-6
de_plz['Density']=de_plz['einwohner']/de_plz['area']
de_plz.rename(columns={'plz':'geo_unit','einwohner':'population'},inplace=True)

de_plz.sort_values('geo_unit',inplace=True)
de_plz.reset_index(drop=True,inplace=True)
de_plz.drop(columns=(['qkm','postcode']),inplace=True)

with open('../dictionaries/city_postcode_DE.pkl','rb') as f:
    city_plz = pickle.load(f)

def dens(city):
    city_dens=de_plz.loc[de_plz['geo_unit'].isin(city_plz[city]),:].copy()
    city_dens.to_csv('../outputs/density_geounits/' + city + '_pop_density.csv',index=False)

    area_lores=pd.DataFrame(city_dens['area'].describe()).reset_index()
    sums=pd.DataFrame(city_dens[['area','population']].sum()).reset_index()
    sums=pd.concat([sums,pd.DataFrame([{'index':'density',0:sums.iloc[1,1]/sums.iloc[0,1]}])])

    writer = pd.ExcelWriter('../outputs/density_geounits/summary_stats_' + city + '.xlsx', engine='openpyxl')

    # include all the dfs/sheets here, and then save
    area_lores.to_excel(writer, sheet_name='area_lores',index=False)
    sums.to_excel(writer, sheet_name='area_pop_sum',index=False)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()

cities_DE=pd.Series(['Berlin','Dresden','Düsseldorf','Frankfurt am Main','Kassel','Leipzig','Magdeburg','Potsdam'])
cities_DE.apply(dens)