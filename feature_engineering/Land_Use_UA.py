# script to calculate land use shares in selected cities
# last update Peter Berrill June 20 2023

# load in required packages
import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import CRS
import sys
import pickle
from citymob import import_csv_w_wkt_to_gdf

# inputs:
# geopackages of urban land use classes for each city from Urban Atlas 2018, or Urban Atlas 2012, whichever is closer to the survey year https://land.copernicus.eu/local/urban-atlas
# city boundary shapefiles
# city postcode or geocode shapefiles

# outputs:
# csv file of urban land use shares for each geocode

crs0= 3035 # crs for boundary 
cities_all=['Berlin','Dresden','Düsseldorf','Frankfurt am Main','Kassel','Leipzig','Magdeburg','Potsdam','Clermont','Dijon','Lille','Lyon','Montpellier','Nantes','Nimes','Paris','Toulouse','Madrid','Wien']
countries=['Germany','Germany','Germany','Germany','Germany','Germany','Germany','Germany','France','France','France','France','France','France','France','France','France','Spain','Austria']
ua_year=['2018','2018','2018','2018','2018','2018','2018','2018','2012','2018','2018','2018','2012','2018','2018','2012','2012','2018','2012']
ua_ver=['v013','v013','v013','v013','v013','v013','v013','v013','revised_v021','v013','v013','v013','revised_v021','v013','v013','revised_v021','revised_v021','v013','revised_v021']

def land_use_ua(city):
    print('Starting ' + city)
    country=countries[cities_all.index(city)]
    year=ua_year[cities_all.index(city)]
    ver=ua_ver[cities_all.index(city)]

    if city=='Potsdam':
        fp='../../MSCA_data/UrbanAtlas/Berlin/Data/Berlin_UA2018_v013.gpkg'

    else: 
        fp='../../MSCA_data/UrbanAtlas/' + city + '/Data/' + city + '_UA' + year + '_' + ver +'.gpkg'
    gdf=gpd.read_file(fp)

    if year =='2012':
        gdf.rename(columns={'class_2012':'class_2018'},inplace=True)

    # create urban fabric (residential) and commercial and road land-uses
    gdf['Class']='Other'
    gdf.loc[gdf['class_2018'].isin(['Discontinuous dense urban fabric (S.L. : 50% -  80%)','Discontinuous medium density urban fabric (S.L. : 30% - 50%)','Discontinuous low density urban fabric (S.L. : 10% - 30%)','Discontinuous very low density urban fabric (S.L. : < 10%)','Continuous urban fabric (S.L. : > 80%)']),'Class']='Urban_Fabric'
    gdf.loc[gdf['class_2018'].isin(['Industrial, commercial, public, military and private units','Construction sites','Airports','Port areas']),'Class']='Industrial_Commercial'
    gdf.loc[gdf['class_2018'].isin(['Fast transit roads and associated land','Other roads and associated land']),'Class']='Road'

    # make a non-road classification, to calculate by reverse the road area
    gdf['RoadStatus']='NonRoad'
    gdf.loc[gdf['class_2018'].isin(['Fast transit roads and associated land','Other roads and associated land']),'RoadStatus']='Road'


    # make an aggregated LU classification, this time distinguishing 'urban' and 'non-urban' areas
    gdf['UrbanStatus']='Non-Urban'
    gdf.loc[gdf['class_2018'].isin(['Discontinuous dense urban fabric (S.L. : 50% -  80%)',
    'Discontinuous medium density urban fabric (S.L. : 30% - 50%)',
    'Discontinuous low density urban fabric (S.L. : 10% - 30%)',
    'Discontinuous very low density urban fabric (S.L. : < 10%)',
    'Continuous urban fabric (S.L. : > 80%)',
    'Industrial, commercial, public, military and private units',
    'Sports and leisure facilities',
    'Green urban areas',
    'Isolated structures',
    'Mineral extraction and dump sites', # should this be included?
    'Fast transit roads and associated land',
    'Railways and associated land ',
    'Airports',
    'Other roads and associated land',
    'Port areas',
    'Construction sites',
    ]),'UrbanStatus']='Urban'
    gdf['UrbanStatus'].value_counts()

    # Read in boundaries
    fp='../outputs/city_boundaries/' + city + '.csv'
    gdf_boundary = import_csv_w_wkt_to_gdf(fp,crs=crs0,geometry_col='geometry')
    # convert the land-uses to base crs, if different
    if gdf.crs!=crs0:
        gdf=gdf.to_crs(crs0)

    # read in postcodes gdf and isolate to specific city
    if country=='Germany':
        fp = "../shapefiles/plz-5stellig.shp/plz-5stellig.shp"
        de_plz = gpd.read_file(fp)
        de_plz=de_plz.to_crs(crs0)
        plz_code=de_plz.plz

        city_poly_fp='../dictionaries/city_postcode_DE.pkl'
        a_file = open(city_poly_fp, "rb")
        city_poly_dict = pickle.load(a_file)

        # if the de_plz codes are string and the city_poly_dict codes are int, then can use de_plz['geocode'].astype('int')
        city_poly=de_plz.loc[(de_plz['plz'].isin(city_poly_dict[city]))]

        # change geocode label 
        if 'plz' in city_poly.columns:
            city_poly.rename(columns={'plz':'geocode'},inplace=True)
    else:
        if city == 'Paris':
            fp = '../outputs/density_geounits/'+city+'_pop_density_lowres.csv'
            city_poly=import_csv_w_wkt_to_gdf(fp,crs0,gc='geo_unit')
        elif city =='Wien':
            fp = '../outputs/density_geounits/'+city+'_pop_density.csv'
            city_poly=import_csv_w_wkt_to_gdf(fp,crs0,gc='geocode')
        else:
            fp = '../outputs/density_geounits/'+city+'_pop_density_mixres.csv'
            city_poly=import_csv_w_wkt_to_gdf(fp,crs0,gc='geocode')
    
        city_poly.drop(columns='Density',inplace=True)
        # change geocode label 
        if 'geo_unit' in city_poly.columns: # for Paris at least
            city_poly.rename(columns={'geo_unit':'geocode'},inplace=True)
        city_poly['geocode']=city_poly['geocode'].astype('str')

    # calculate postcode area
    city_poly['area']=city_poly.area

    # restrict land-use data to those within our boundary 
    intersect=gpd.overlay(gdf_boundary,gdf,how='intersection')
    intersect.drop(columns=['crs','country'],inplace=True)

    # make list of postcodes
    polylist=city_poly['geocode'].tolist()
    # predefine stats
    urbfabarea_poly=[]
    comarea_poly=[]
    roadarea_poly=[]
    urbarea_poly=[]

    # calculate stats for each postcode in a for loop
    for p in polylist:
        poly_gdf1=gpd.overlay(city_poly.loc[city_poly['geocode']==p,], intersect, how='intersection')
        poly_gdf1['area2']=poly_gdf1.area

        urbfabarea_poly1=poly_gdf1.loc[poly_gdf1['Class']=='Urban_Fabric',].groupby('geocode')['area2'].agg('sum')
        if len(urbfabarea_poly1)>0:
            urbfabarea_poly.append(urbfabarea_poly1[0])
        else:
            urbfabarea_poly.append(0)

        comarea_poly1=poly_gdf1.loc[poly_gdf1['Class']=='Industrial_Commercial',].groupby('geocode')['area2'].agg('sum')
        if len(comarea_poly1)>0:
            comarea_poly.append(comarea_poly1[0])
        else:
            comarea_poly.append(0)

        roadarea_poly1=poly_gdf1.loc[poly_gdf1['Class']=='Road',].groupby('geocode')['area2'].agg('sum')
        if len(roadarea_poly1)>0:
            roadarea_poly.append(roadarea_poly1[0])
        else:
            roadarea_poly.append(0)

        urbarea_poly1=poly_gdf1.loc[poly_gdf1['UrbanStatus']=='Urban',].groupby('geocode')['area2'].agg('sum')
        if len(urbarea_poly1)>0:
            urbarea_poly.append(urbarea_poly1[0])  
        else:
            urbarea_poly.append(0)

    data={'geocode':polylist,'tot_area':city_poly['area'].tolist(),'urb_fabric_area':urbfabarea_poly,'commercial_area':comarea_poly,'road_area':roadarea_poly,'urban_area':urbarea_poly}

    lu_poly=pd.DataFrame(data)
    lu_poly['pc_urb_fabric']=lu_poly['urb_fabric_area']/lu_poly['tot_area']
    lu_poly['pc_comm']=lu_poly['commercial_area']/lu_poly['tot_area']
    lu_poly['pc_road']=lu_poly['road_area']/lu_poly['tot_area']
    lu_poly['pc_urban']=lu_poly['urban_area']/lu_poly['tot_area']

    lu_poly['pc_urb_fabric_urban']=lu_poly['urb_fabric_area']/lu_poly['urban_area']
    lu_poly['pc_comm_urban']=lu_poly['commercial_area']/lu_poly['urban_area']
    lu_poly['pc_road_urban']=lu_poly['road_area']/lu_poly['urban_area']

    lu_poly.drop(columns=['urb_fabric_area','commercial_area','road_area','urban_area'],inplace=True)
    lu_poly['geocode']=lu_poly['geocode'].astype('str')

    lu_poly.to_csv('../outputs/LU/UA_'+city+'.csv',index=False)
    print('Finished with ' + city)

cities=pd.Series(cities_all)
cities.apply(land_use_ua) 