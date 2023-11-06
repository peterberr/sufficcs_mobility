import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import rasterio
from rasterstats import zonal_stats
from rasterio.merge import merge
from citymob import import_csv_w_wkt_to_gdf
from shapely import wkt

cities=['Berlin','Dresden','Düsseldorf','Frankfurt am Main','Kassel','Leipzig','Magdeburg','Potsdam']
countries=['Germany','Germany','Germany','Germany','Germany','Germany','Germany','Germany']
crs0=3035
# load in gridded German population from 2011 census
fp='../../MSCA_data/Germany_population_new/csv_Bevoelkerung_100m_Gitter/Zensus_Bevoelkerung_100m-Gitter.csv'
pop_grid=pd.read_csv(fp,sep=';')
pop_grid.loc[pop_grid['Einwohner']==-1,'Einwohner']=np.nan

# load in geodataframe of postcode polygons
fp='../../sufficcs_mobility/dictionaries/city_postcode_DE_basic.pkl'
with open(fp,'rb') as f:
    DE_plz = pickle.load(f)

def dens_DE(city):
    print(city)
    if city=='Dresden':
        fp='../../MSCA_data/Germany_population_new/Dresden/Stadtteile.csv'
        df=pd.read_csv(fp,sep=';')
        df['geometry']=df['shape'].str.replace('SRID=4326;','')
        df.head()

        gdf = gpd.GeoDataFrame(df, geometry=df['geometry'].apply(wkt.loads),crs=4326)
        gdf.drop(columns=['sst','sst_klar','historie','shape','aend'],inplace=True)
        gdf.to_crs(3035,inplace=True)

        # population data from https://www.dresden.de/de/leben/stadtportrait/statistik/demografiemonitor/Demografiemedien/atlas.html
        fp='../../MSCA_data/Germany_population_new/Dresden/Dresden_pop_stadtteile.csv'
        pop_st=pd.read_csv(fp,encoding='latin-1')
        pop_st['2018_2011']=pop_st['2018']/pop_st['2011']


        gdf['blocknr']=gdf['blocknr'].astype(str)
        gdf=gdf.merge(pop_st.loc[:,['Nummer','2011','2018','2018_2011']],left_on='blocknr',right_on='Nummer',how='left')
        gdf.rename(columns={'blocknr':'unit_id'},inplace=True)

    if city == 'Leipzig':
        fp='../../MSCA_data/Germany_population_new/Leipzig/Ortsteile_Leipzig_UTM33N_shp/ot.shp'
        gdf=gpd.read_file(fp)
        gdf.to_crs(3035,inplace=True)

        # pop data source https://opendata.leipzig.de/dataset/einwohner-jahreszahlen-kleinraumig/resource/624a34ca-df67-4375-85ef-cefcd4cc0168
        fp='../../MSCA_data/Germany_population_new/Leipzig/Bevölkerungsbestand_Einwohner.csv'
        pop_st=pd.read_csv(fp,encoding='utf-8')
        pop_st=pop_st.loc[pop_st['Sachmerkmal']=='Einwohner insgesamt',:]
        pop_st['2018_2011']=pop_st['2018']/pop_st['2011']

        gdf=gdf.merge(pop_st.loc[:,['Gebiet','2011','2018','2018_2011']],left_on='Name',right_on='Gebiet',how='left')
        gdf.rename(columns={'OT':'unit_id'},inplace=True)
        
    if city == 'Frankfurt am Main':
            fp='../../MSCA_data/Germany_population_new/Frankfurt am Main/Stadtteile/Stadtteile_Frankfurt.shp'
            gdf=gpd.read_file(fp)
            gdf.to_crs(crs0,inplace=True)

            # load population data for stadteile (or other spatial unit) 
            # source https://statistikportal.frankfurt.de/strukturdatenatlas/stadtteile/html/atlas.html
            fp='../../MSCA_data/Germany_population_new/Frankfurt am Main/Stadtteile_pop.csv'
            pop_st=pd.read_csv(fp,encoding='latin-1')
            pop_st['2018_2011']=pop_st['2018']/pop_st['2011']

            gdf['STTLNR']=gdf['STTLNR'].astype(int).astype(str)
            gdf=gdf.merge(pop_st.loc[:,['Codes','2011','2018','2018_2011']],left_on='STTLNR',right_on='Codes',how='left')
            gdf.rename(columns={'STTLNR':'unit_id'},inplace=True)
            gdf.drop(columns=['OBJECTID','Codes'],inplace=True)

    if city == 'Düsseldorf':
        fp='../../MSCA_data/Germany_population_new/Düsseldorf/Stadtteile-shp/Düsseldorf_Stadtteile.shp'
        gdf=gpd.read_file(fp)
        gdf.to_crs(crs0,inplace=True)

        # load population data for stadteile (or other spatial unit) 
        # source https://www.duesseldorf.de/fileadmin/Amt12/statistik/stadtforschung/download/05_bevoelkerung/SD_2019_Kap_5.pdf
        fp='../../MSCA_data/Germany_population_new/Düsseldorf/pop_2011_2018_st.csv'
        pop_st=pd.read_csv(fp,encoding='latin-1')
        pop_st['2018_2011']=pop_st['2018']/pop_st['2011']
        pop_st['Stadtteil']=pop_st['Stadtteil'].astype(str)

        gdf['Stadtteil']=gdf['Stadtteil'].astype(int).astype(str)
        gdf=gdf.merge(pop_st.loc[:,['Stadtteil','2011','2018','2018_2011']],how='left')
        gdf.rename(columns={'Stadtteil':'unit_id'},inplace=True)
        gdf.drop(columns=['Quelle','Stand','Stadtbezir','SHAPE_Leng','SHAPE_Area'],inplace=True)

    if city == 'Magdeburg':
        fp='../../MSCA_data/Germany_population_new/Magdeburg/Stadtteile/2019-11-12_STB_ETRS89.shp'
        gdf=gpd.read_file(fp)
        gdf.to_crs(crs0,inplace=True)
        # dissolve to convert from statistiche bezirke to stadtteile.
        # the stadtteil is identified by the first 2 digits of the 3-digit SB, see here e.g. on page 3 https://www.magdeburg-tourist.de/media/custom/37_27260_1.PDF?1506336132
        gdf['unit_id']=gdf['SBZ_Text'].str[:2]
        gdf=gdf.dissolve(by='unit_id', aggfunc='sum').reset_index()
        gdf.drop(columns=['DGN_LEVEL','SBZ_Nummer'],inplace=True)

        # load population data for stadteile (or other spatial unit) 
        # source https://statistik.magdeburg.de/KISS-MD/, https://www.magdeburg.de/PDF/Bev%C3%B6lkerung_Demografie_2019_Heft_104.PDF?ObjSvrID=37&ObjID=38422&ObjLa=1&Ext=PDF&WTR=1&_ts=1602221783
        fp='../../MSCA_data/Germany_population_new/Magdeburg/Pop_2018_2011.csv'
        pop_st=pd.read_csv(fp,encoding='latin-1')
        pop_st['Gebiet_NR']=pop_st['Gebiet_NR'].astype(str).str.zfill(2)
        pop_st['2018_2011']=pop_st['2018']/pop_st['2011']

        gdf=gdf.merge(pop_st.loc[:,['Gebiet_NR','2011','2018','2018_2011']],left_on='unit_id',right_on='Gebiet_NR',how='left')
        gdf.drop(columns=['Gebiet_NR'],inplace=True)

    if city=='Kassel':
        fp='../../MSCA_data/Germany_population_new/Kassel/Ortsbezirke_Stadt_Kassel/Ortsbezirke_20231017_GK.shp'
        gdf=gpd.read_file(fp)
        gdf.to_crs(crs0,inplace=True)
        gdf.drop(columns=['OBJECTID','pitID','pitID','pitClass','pitValue'],inplace=True)

        # load population data for stadteile (or other spatial unit) 
        # sources https://www.kassel.de/statistik/berichte/archiv/Jahresbericht_2014.pdf, https://www.kassel.de/statistik/berichte/archiv/Jahresbericht_2018.pdf
        fp='../../MSCA_data/Germany_population_new/Kassel/pop_st.csv'
        pop_st=pd.read_csv(fp,encoding='latin-1')
        pop_st['2018_2011']=pop_st['2018']/pop_st['2011']
        pop_st['unit_id']=pop_st['Stadtteile'].str[:2]
        pop_st['st_name']=pop_st['Stadtteile'].str[3:]
        pop_st.drop(columns='Stadtteile',inplace=True)

        gdf.rename(columns={'Ortsbezirk':'unit_id'},inplace=True)
        gdf=gdf.merge(pop_st.loc[:,['unit_id','2011','2018','2018_2011']],on='unit_id',how='left')
        gdf.drop(columns=['Shape_STAr','Shape_STLe'],inplace=True)
        gdf.loc[gdf['unit_id']=='25','2018_2011']=1 # assume no change in pop in Dönchelandschaft (ortsbezirksfrei). according to these stats noone lives there, but maybe census is different

    if city == 'Potsdam':
        fp='../../MSCA_data/Germany_population_new/Potsdam/stadtteile/stadtteile.shp'
        gdf=gpd.read_file(fp)
        gdf.to_crs(crs0,inplace=True)

        # load population data for stadteile (or other spatial unit) 
        fp='../../MSCA_data/Germany_population_new/Potsdam/Potsdam_pop_st_2017_2018.csv'
        pop_st=pd.read_csv(fp,encoding='latin-1')
        pop_st['Stadtteil']=pop_st['Stadtteil'].astype(str)

        gdf.rename(columns={'id_st':'unit_id'},inplace=True)

    # now for all cities, finish the calculations and save the densities by plz

    # load city 100m grid
    fp='../../MSCA_data/Germany_population_new/DE_Grid100m/' + city + '.shp'
    grid_city=gpd.read_file(fp)
    grid_city['Center']=grid_city.centroid
    grid_city.rename(columns={'geometry':'grid_polygon'},inplace=True)
    grid_city.set_geometry('Center',inplace=True)
    grid_city.drop(columns=['index_righ','crs'],inplace=True,errors='ignore')

    # join the gdf population data with the city grid,
    grid_unit_city=gpd.sjoin(gdf,grid_city)
    grid_unit_city.drop(columns='index_right',inplace=True)
    # these should be same when merging on grid centroid
    print(len(grid_unit_city))
    print(len(grid_unit_city['id'].unique()))
    print('check we have unit grid ids')
    print(len(grid_unit_city)==len(grid_unit_city['id'].unique()))
    # merge the city grid data with german grid population from 2011 census. this takes around 30sec
    grid_pop_city=grid_unit_city.merge(pop_grid.loc[:,['Gitter_ID_100m','Einwohner']],left_on='id',right_on='Gitter_ID_100m')

    # workaround for potsdam for which we don't have stadtteil population for 2011
    if city=='Potsdam':
        sum_st_2011=grid_pop_city.groupby('unit_id')['Einwohner'].sum().to_frame().reset_index().rename(columns={'Einwohner':'2011','unit_id':'Stadtteil'})
        pop_st=pop_st.merge(sum_st_2011,how='outer')
        pop_st['2018_2011']=pop_st['2018']/pop_st['2011']
        pop_st.rename(columns={'Stadtteil':'unit_id'},inplace=True)
        grid_pop_city=grid_pop_city.merge(pop_st.loc[:,['unit_id','2018_2011']])

    if city=='Potsdam':
        grid_pop_city['census2local']=1
    else:
        grid_pop_city['census2local']=gdf['2011'].sum()/grid_pop_city['Einwohner'].sum()
    grid_pop_city['Pop_2018']=grid_pop_city['Einwohner']*grid_pop_city['2018_2011']*grid_pop_city['census2local']

    grid_pop_city['grid_center']=grid_pop_city['grid_polygon'].centroid
    grid_pop_city.set_geometry('grid_center',inplace=True)

    # load in geodataframe of city postcode geometries
    city_plz=DE_plz[city]
    fp='../../sufficcs_mobility/source/GTFS/postcodes_gpkg_all/'+city+'_postcodes'+'.gpkg'
    gdf_plz=gpd.read_file(fp)
    gdf_plz.to_crs(3035,inplace=True)
    gdf_plz['geocode']=gdf_plz['geocode'].astype(str).str.zfill(5)
    gdf_plz=gdf_plz.loc[gdf_plz['geocode'].isin(city_plz.tolist())]
    gdf_plz['Area']=1e-6*gdf_plz.area

    # add in postcode labels with sjon
    grid_pop_city_plz=gpd.sjoin(grid_pop_city,gdf_plz)
    # sum population by plz
    pop_plz=grid_pop_city_plz.groupby('geocode')['Pop_2018'].sum().to_frame().reset_index()

    # add plz population to gdf
    gdf_plz=gdf_plz.merge(pop_plz)
    gdf_plz['Density']=round(gdf_plz['Pop_2018']/gdf_plz['Area'])
    gdf_plz['Pop_2018']=round(gdf_plz['Pop_2018'])

    gdf_plz.drop(columns='geometry').to_csv('../outputs/density_geounits/'+city+'_pop_density.csv',index=False)

    area_lores=pd.DataFrame(gdf_plz['Area'].describe()).reset_index()

    summ=gdf_plz.loc[:,['Area','Pop_2018']].sum().reset_index()
    summ=pd.concat([summ,pd.DataFrame([{'index':'density',0:summ.iloc[1,1]/summ.iloc[0,1]}])])
    summ.rename(columns={'index':'variable',0:'value'},inplace=True)
    #summ.to_csv('../outputs/density_geounits/summary_stats_'+city+'.csv',index=False)

    area_lores=pd.DataFrame(gdf_plz['Area'].describe()).reset_index()
    writer = pd.ExcelWriter('../outputs/density_geounits/summary_stats_' + city + '.xlsx', engine='openpyxl')
    area_lores=pd.DataFrame(gdf_plz['Area'].describe()).reset_index()
    area_lores.to_excel(writer, sheet_name='area_lores',index=False)
    summ.to_excel(writer, sheet_name='area_pop_sum',index=False)
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()

cities_DE=pd.Series(['Dresden','Düsseldorf','Frankfurt am Main','Kassel','Leipzig','Magdeburg','Potsdam'])
cities_DE.apply(dens_DE)

# calculate Berlin densities separately
city='Berlin'
print(city)
# load in geodataframe of city postcode geometries
city_plz=DE_plz[city]
fp='../../sufficcs_mobility/source/GTFS/postcodes_gpkg_all/'+city+'_postcodes'+'.gpkg'
gdf_plz=gpd.read_file(fp)
gdf_plz.to_crs(3035,inplace=True)
gdf_plz['geocode']=gdf_plz['geocode'].astype(str).str.zfill(5)
gdf_plz=gdf_plz.loc[gdf_plz['geocode'].isin(city_plz.tolist())]
gdf_plz['Area']=1e-6*gdf_plz.area
gdf_plz=gdf_plz.loc[gdf_plz['geocode']!='16548',]

# load population data
pop=pd.read_csv('../../MSCA_data/berlin/Berlin housing stock/pop_plz_BBStatistk.csv',encoding='latin')
pop['geocode']=pop['geocode'].astype(str)
pop=pop.groupby('geocode')['Population'].sum().to_frame().reset_index()
pop=pop.loc[~pop['geocode'].isin(['15566','15569']),:]
pop.rename(columns={'Population':'Pop_2018'},inplace=True)

gdf_plz=gdf_plz.merge(pop)
gdf_plz['Density']=round(gdf_plz['Pop_2018']/gdf_plz['Area'])
gdf_plz.drop(columns='geometry').to_csv('../outputs/density_geounits/'+city+'_pop_density.csv',index=False)

summ=gdf_plz.loc[:,['Area','Pop_2018']].sum().reset_index()
summ=pd.concat([summ,pd.DataFrame([{'index':'density',0:summ.iloc[1,1]/summ.iloc[0,1]}])])
summ.rename(columns={'index':'variable',0:'value'},inplace=True)
# summ.to_csv('../outputs/density_geounits/summary_stats_'+city+'.csv',index=False)
area_lores=pd.DataFrame(gdf_plz['Area'].describe()).reset_index()
writer = pd.ExcelWriter('../outputs/density_geounits/summary_stats_' + city + '.xlsx', engine='openpyxl')
area_lores=pd.DataFrame(gdf_plz['Area'].describe()).reset_index()
area_lores.to_excel(writer, sheet_name='area_lores',index=False)
summ.to_excel(writer, sheet_name='area_pop_sum',index=False)
# Close the Pandas Excel writer and output the Excel file.
writer.save()
writer.close()

area_lores=pd.DataFrame(gdf_plz['Area'].describe()).reset_index()
area_lores.to_csv('../outputs/density_geounits/areadist_'+city+'.csv',index=False)