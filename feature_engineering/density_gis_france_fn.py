# script to calculat population density by spatial units in French cities
# last update Peter Berrill Nov 25 2022

import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import CRS
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import pickle
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

# inputs:
# French spatial unit (differs by city) shapefiles, data directly from surveys
# French IRIS shapefiles (2014 definition), by Department, definition of IRIS here (https://www.insee.fr/en/metadonnees/definition/c1523). source: https://wxs.ign.fr/1yhlj2ehpqf3q6dt6a2y7b64/telechargement/inspire/CONTOURS-IRIS-2014-01-01%24CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/file/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01.7z
# French population in 2012 and 2017 by IRIS. sources (https://www.insee.fr/fr/statistiques/2028582) for 2012 and (https://www.insee.fr/fr/statistiques/4799309) for 2017

# outputs:
# city boundary shapefile
# population density by spatial units, at different levels of aggregation (low-res, mix-res, high-res)
# summary stats on population, area, and area distribution of gemeinden (or other spatial units)
# figures showing spatial units for each resolution
# pickled dictionaries to translate the geocodes from the survey to the mixed-res ids needed to merge with the geospatial data

def remove_invalid_geoms(gdf,crs0,gdf_name):
    '''
    Identify invalid geometries, and replace them with valid geometries using the Shapely `make_valid` method.
    '''
    gdf['valid']=gdf.is_valid
    if any(gdf.is_valid==False):
        pc=round(100*sum(gdf.is_valid==False)/len(gdf))
        print(str(pc) + 'percent of geometries are invalid in ' + gdf_name)
        ivix=gdf.loc[gdf['valid']==False,].index
        geos=[]
        for i in ivix:
            gi=make_valid(gdf.loc[i,'geometry'])
            geos.append(gi)
        d=gdf.loc[gdf['valid']==False,].drop(columns=['geometry','valid'])
        d['geometry']=geos
        gdfv=gpd.GeoDataFrame(d,crs=crs0)
        gdfv['valid']=gdfv.is_valid
        gdf_new=gpd.GeoDataFrame(pd.concat([gdf.loc[gdf['valid']==True,:],gdfv]),crs=crs0)
        gdf_new.sort_index(inplace=True)
        gdf=gdf_new.copy()

        if any(gdf_new.is_valid==False):
            print('Unable to make valid all geometries')
    
    return(gdf)

# size_thresh=10  # km2 if this threshold is smaller than 12.91 km2, then need to insert an exception for Lyon geometry 247551 within faulty geometry 247003
# in pg 10 of the 'ATLAS' pdf file for Clermont, we see that there are 32 zones where face to face interviews were done in the region of Greater Clermont. 24 further sectors in the broader had telephone interviews.
# Originaly we restricted analysis to the 32 areas of greater Clermont, but this made quite a large total area (1,350 km2, 1.5x larger than the area we use for Paris or Berlin), 
# so we instead restrict to the first 19 sectors which includes Clermont vill and Hors-Clermont, see maps on pg 12-13 of ATLAS.

def french_density_shapefiles(city,size_thresh):
    crs0=3035
    #size_thresh=10
    if city=='Clermont':
        dep='63'
        survey_yr=2012
        fp='C:/Users/peter/Documents/projects/MSCA_data/FranceRQ/lil-0924_Clermont.csv/Doc/SIG/EDGT Clermont2012_DTIR.mid'
        gdf=gpd.read_file(fp)
        gdf.to_crs(crs0,inplace=True)
        geo_unit=gdf['NUM_DTIR'].sort_values().unique()
        geo_unit=geo_unit[0:19]

        # make a shapefile consisting only of the larger sector geo units
        gdf=gdf.loc[gdf['NUM_DTIR'].isin(geo_unit),] 
        gdf2=gdf[['DTIR','geometry']].dissolve(by='DTIR').reset_index()
        # calculate area of geo units for each gdf
        gdf=remove_invalid_geoms(gdf,crs0, 'gdf')
        gdf['area']=gdf.area*1e-6 # in km2
        gdf2=remove_invalid_geoms(gdf2,crs0, 'gdf2')
        gdf2['area']=gdf2.area*1e-6
        # rename geounits for constitency with all French cities
        gdf.rename(columns={'DFIN':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'DTIR':'geo_unit'},inplace=True)
        gdf2.rename(columns={'DTIR':'geo_unit'},inplace=True)
        # load in IRIS shapefiles for Clermont Department
        fp='../../../projects/city_mobility/shapefiles/France other/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        iris_gdf.set_geometry('center',inplace=True)

    if city=='Montpellier':
        dep='34'
        survey_yr=2014
        fp='C:/Users/peter/Documents/projects/MSCA_data/FranceRQ/lil-0937_Montpellier.csv/Doc/SIG/EDGT Montpellier_EDVM Beziers_Zones fines.mid'
        gdf=gpd.read_file(fp)
        #gdf=gdf.loc[(gdf['ID_ENQ']==1) & (gdf['NUM_SECTEUR'].astype('int')<58),]
        gdf=gdf.to_crs(crs0)
        gdf=remove_invalid_geoms(gdf,crs0, 'gdf')
        gdf['area']=gdf.area*1e-6

        # load a shapefile consisting only of the larger sector geo units
        fp2='C:/Users/peter/Documents/projects/MSCA_data/FranceRQ/lil-0937_Montpellier.csv/Doc/SIG/EDGT Montpellier_EDVM Beziers_DTIR.mid'
        gdf2=gpd.read_file(fp2)
        geo_unit=gdf2.loc[gdf2['D5_D10']=='01', 'DTIR'].sort_values().unique()
        # restrict surveys to the 'Montpellier Agglomération', with face to face interviews
        gdf2=gdf2.loc[(gdf2['ID_ENQ']==1) & (gdf2['DTIR'].isin(geo_unit)),]
        gdf=gdf.loc[(gdf['ID_ENQ']==1) & (gdf['NUM_SECTEUR'].isin(geo_unit)),]
        gdf2=gdf2.to_crs(crs0)
        gdf2=remove_invalid_geoms(gdf2,crs0, 'gdf2')
        gdf2['area']=gdf2.area*1e-6
        # rename geounits for constitency with all French cities
        gdf.rename(columns={'NUM_ZF_2013':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'NUM_SECTEUR':'geo_unit'},inplace=True)
        gdf2.rename(columns={'DTIR':'geo_unit'},inplace=True)
        # load in IRIS shapefiles for Department
        fp='../../../projects/city_mobility/shapefiles/France other/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        iris_gdf.set_geometry('center',inplace=True)

    if city=='Lyon': # think i use the .tab file https://stackoverflow.com/questions/22218069/how-to-load-mapinfo-file-into-geopandas for mapinfo gis file
        dep='69'
        survey_yr=2015
        fp='C:/Users/peter/Documents/projects/MSCA_data/FranceRQ/lil-1023_Lyon.csv/Doc/SIG/EDGT_AML2015_ZF_GT.TAB'
        gdf=gpd.read_file(fp)
        # restrict to D12 zones 01 to 04 (DTIR<258), these are sufficiently close to the center of Lyon, and combined make up an area of 841km2
        geo_unit=gdf.loc[gdf['DTIR'].astype('int')<258,'DTIR'].sort_values().unique()
        gdf=gdf.loc[gdf['DTIR'].isin(geo_unit),] 
        gdf=gdf.to_crs(crs0)
        gdf=remove_invalid_geoms(gdf,crs0, 'gdf')
        gdf['area']=gdf.area*1e-6

        fp2='C:/Users/peter/Documents/projects/MSCA_data/FranceRQ/lil-1023_Lyon.csv/Doc/SIG/EDGT_AML2015_DTIR.TAB'
        gdf2=gpd.read_file(fp2)
        # restrict to selected zones
        gdf2=gdf2.loc[gdf2['DTIR'].isin(geo_unit),] 
        gdf2=gdf2.to_crs(crs0)
        gdf2=remove_invalid_geoms(gdf2,crs0, 'gdf2')
        gdf2['area']=gdf2.area*1e-6
        # rename geounits for constitency with all French cities
        gdf.rename(columns={'ZF2015_Nouveau_codage':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'DTIR':'geo_unit'},inplace=True)
        gdf2.rename(columns={'DTIR':'geo_unit'},inplace=True)
        # load in IRIS shapefiles for Lyon Department
        fp='../../../projects/city_mobility/shapefiles/France other/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        iris_gdf.set_geometry('center',inplace=True)

    if city=='Toulouse':
        dep='31'
        survey_yr=2013
        fp='../../MSCA_data/FranceRQ/lil-0933_Toulouse.csv/Doc/SIG/ZONE_FINE_EMD2013_FINAL4.mid' # hires gdf
        gdf=gpd.read_file(fp)
        gdf.to_crs(crs0,inplace=True)
        geo_unit=gdf['SECTEUR_EMD2013'].sort_values().unique()
        geo_unit=geo_unit[0:56]

        # make a shapefile consisting only of the larger sector geo units
        gdf=gdf.loc[gdf['SECTEUR_EMD2013'].isin(geo_unit),] 
        gdf2=gdf[['SECTEUR_EMD2013','geometry']].dissolve(by='SECTEUR_EMD2013').reset_index()
        # # calculate area of geo units for each gdf
        gdf=remove_invalid_geoms(gdf,crs0, 'gdf')
        gdf['area']=gdf.area*1e-6 # in km2
        gdf2=remove_invalid_geoms(gdf2,crs0, 'gdf2')
        gdf2['area']=gdf2.area*1e-6
        # # rename geounits for constitency with all French cities
        gdf.rename(columns={'ZF_SEC_EMD2013':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'SECTEUR_EMD2013':'geo_unit'},inplace=True)
        gdf2.rename(columns={'SECTEUR_EMD2013':'geo_unit'},inplace=True)
        # # load in IRIS shapefiles for Department
        fp='../../../projects/city_mobility/shapefiles/France other/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        iris_gdf.set_geometry('center',inplace=True)

    if city=='Nantes': 
        dep='44'
        survey_yr=2015
        fp='C:/Users/peter/Documents/projects/MSCA_data/FranceRQ/lil-1024_Nantes.csv/Doc/SIG/EDGT44_2015_ZF.TAB'
        gdf=gpd.read_file(fp)
        geo_unit=gdf.loc[gdf['NOM_D10']=='Nantes Métropole','NUM_DTIR'].sort_values().unique()
        gdf=gdf.loc[gdf['NUM_DTIR'].isin(geo_unit),] 
        gdf=gdf.to_crs(crs0)
        gdf=remove_invalid_geoms(gdf,crs0, 'gdf')
        gdf['area']=gdf.area*1e-6

        fp2='C:/Users/peter/Documents/projects/MSCA_data/FranceRQ/lil-1024_Nantes.csv/Doc/SIG/EDGT44_2015_DTIR.TAB'
        gdf2=gpd.read_file(fp2)
        # restrict to selected zones
        gdf2=gdf2.loc[gdf2['NUM_DTIR'].isin(geo_unit),] 
        gdf2=gdf2.to_crs(crs0)
        gdf2=remove_invalid_geoms(gdf2,crs0, 'gdf2')
        gdf2['area']=gdf2.area*1e-6
        # # rename geounits for constitency with all French cities
        gdf.rename(columns={'Id_zf_cerema':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'NUM_DTIR':'geo_unit'},inplace=True)
        gdf2.rename(columns={'NUM_DTIR':'geo_unit'},inplace=True)
        # load in IRIS shapefiles for Lyon Department
        fp='../../../projects/city_mobility/shapefiles/France other/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        iris_gdf.set_geometry('center',inplace=True)

    if city=='Nimes':
        dep='30'
        survey_yr=2015
        fp='C:/Users/peter/Documents/projects/MSCA_data/FranceRQ/lil-1135_Nimes.csv/Doc/SIG/EMD_Nimes_2014_2015_ZF.TAB'
        gdf=gpd.read_file(fp)
        # restrict to D12 zones 01 to 04, these are sufficiently close to the center of Lyon, and combined make up an area of 841km2
        geo_unit=gdf.loc[gdf['NOM_DTIR']=='NIMES','NUM_DTIR'].sort_values().unique() # restricting to the Nimes hors hypercentre, which covers dtir 1-20.
        gdf=gdf.loc[gdf['NUM_DTIR'].isin(geo_unit),] 
        gdf=gdf.to_crs(crs0)
        gdf=remove_invalid_geoms(gdf,crs0, 'gdf')
        gdf['area']=gdf.area*1e-6

        fp2='C:/Users/peter/Documents/projects/MSCA_data/FranceRQ/lil-1135_Nimes.csv/Doc/SIG/EMD_Nimes_2014_2015_DTIR.TAB' 
        gdf2=gpd.read_file(fp2)
        # restrict to selected zones
        gdf2=gdf2.loc[gdf2['NUM_DTIR'].isin(geo_unit),] 
        gdf2=gdf2.to_crs(crs0)
        gdf2=remove_invalid_geoms(gdf2,crs0, 'gdf2')
        gdf2['area']=gdf2.area*1e-6
        # # rename geounits for constitency with all French cities
        gdf.rename(columns={'NUM_ZF_2013':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'NUM_DTIR':'geo_unit'},inplace=True)
        gdf2.rename(columns={'NUM_DTIR':'geo_unit'},inplace=True)
        # load in IRIS shapefiles for Lyon Department
        fp='../../../projects/city_mobility/shapefiles/France other/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        iris_gdf.set_geometry('center',inplace=True)
        
    if city=='Dijon':
        dep='21'
        survey_yr=2016 
        fp='C:/Users/peter/Documents/projects/MSCA_data/FranceRQ/lil-1214_Dijon.csv/Doc/SIG/EDGT_DIJON_2016_ZF.TAB'
        gdf=gpd.read_file(fp)
        # restrict to D12 zones 01 to 04, these are sufficiently close to the center of Lyon, and combined make up an area of 841km2
        geo_unit=gdf.loc[gdf['NUM_D2']=='01','NUM_DTIR'].sort_values().unique() # restricting to the Ville de Dijon and Grand Dijonhypercentre, which covers dtir 1-20.
        gdf=gdf.loc[gdf['NUM_DTIR'].isin(geo_unit),] 
        gdf=gdf.to_crs(crs0)
        gdf=remove_invalid_geoms(gdf,crs0, 'gdf')
        gdf['area']=gdf.area*1e-6

        fp2='C:/Users/peter/Documents/projects/MSCA_data/FranceRQ/lil-1214_Dijon.csv/Doc/SIG/EDGT_DIJON_2016_DTIR.TAB' 
        gdf2=gpd.read_file(fp2)
        # restrict to selected zones
        gdf2=gdf2.loc[gdf2['NUM_DTIR'].isin(geo_unit),] 
        gdf2=gdf2.to_crs(crs0)
        gdf2=remove_invalid_geoms(gdf2,crs0, 'gdf2')
        gdf2['area']=gdf2.area*1e-6

        # # rename geounits for constitency with all French cities
        gdf.rename(columns={'NUM_ZF':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'NUM_DTIR':'geo_unit'},inplace=True)
        gdf2.rename(columns={'NUM_DTIR':'geo_unit'},inplace=True)
        # load in IRIS shapefiles 
        fp='../../../projects/city_mobility/shapefiles/France other/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        iris_gdf.set_geometry('center',inplace=True)

    if city=='Lille':
        dep='59'
        survey_yr=2015 
        fp='C:/Users/peter/Documents/projects/MSCA_data/FranceRQ/lil-1152_Lille.csv/Doc/SIG/EDGT_LILLE_2016_ZF.TAB'
        gdf=gpd.read_file(fp)
        # restrict to D12 zones 01 to 04, these are sufficiently close to the center of Lyon, and combined make up an area of 841km2
        geo_unit=gdf.loc[gdf['ST']<158,'ST'].sort_values().unique() # restricting to the Nimes hors hypercentre, which covers dtir 1-20.
        gdf=gdf.loc[gdf['ST'].isin(geo_unit),] 
        gdf=gdf.to_crs(crs0)
        gdf=remove_invalid_geoms(gdf,crs0, 'gdf')
        gdf['area']=gdf.area*1e-6

        fp2='C:/Users/peter/Documents/projects/MSCA_data/FranceRQ/lil-1152_Lille.csv/Doc/SIG/EDGT_LILLE_2016_DTIR.TAB'
        gdf2=gpd.read_file(fp2)
        # restrict to selected zones
        gdf2=gdf2.loc[gdf2['ST'].isin(geo_unit),] 
        gdf2=gdf2.to_crs(crs0)
        gdf2=remove_invalid_geoms(gdf2,crs0, 'gdf2')
        gdf2['area']=gdf2.area*1e-6
        # # rename geounits for constitency with all French cities
        gdf.rename(columns={'ZFIN2016F':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'ST':'geo_unit'},inplace=True)
        gdf2.rename(columns={'ST':'geo_unit'},inplace=True)
        # load in IRIS shapefiles for Lyon Department
        fp='../../../projects/city_mobility/shapefiles/France other/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        iris_gdf.set_geometry('center',inplace=True)

    if city=='Paris': 
        deps=['75','92','93','94']
        ext_com=['91027','91326','91432','91479','91589','91687','95018']
        survey_yr=2010
        fp='C:/Users/peter/Documents/projects/city_mobility/shapefiles/code-postal-code-insee-2015/code-postal-code-insee-2015.shp'
        gdf0=gpd.read_file(fp)
        gdf0=gdf0.loc[:,('insee_com','nom_com','superficie','population','code_dept','nom_dept','code_reg','nom_reg','nom_de_la_c','geometry')].drop_duplicates()
        ext_com=['91027','91326','91432','91479','91589','91687','95018']
        gdf2=gdf0.loc[(gdf0['insee_com'].isin(ext_com)) | (gdf0['code_dept'].isin(deps)), ]

        gdf2=gdf2.to_crs(crs0)
        gdf2['area']=gdf2['geometry'].area*1e-6
        df2=pd.DataFrame(gdf2.drop(columns='geometry'))

        gdf_idf=gdf0.loc[gdf0['code_dept'].isin(['75','77','78','91','92','93','94','95']), ]
        gdf_idf.to_crs(crs0,inplace=True)
        del gdf0

        fp='C:/Users/peter/Documents/projects/MSCA_data/FranceRQ/lil-0883_IleDeFrance.csv/Doc/Carreaux_shape_mifmid/carr100m.shp'
        gdf=gpd.read_file(fp)
        # assumed crs of the grid
        gdf.set_crs('epsg:27561',inplace=True)
        gdf.to_crs(crs0,inplace=True)
        gdf['area']=gdf.area
        # boundary_idf_grid=gpd.GeoDataFrame(geometry=[gdf['geometry'].unary_union],crs=crs0)
        # boundary=gpd.GeoDataFrame(geometry=[gdf2['geometry'].unary_union],crs=crs0)

        # restrict higher res grid gdf to selected communes, i.e. those in Greater Paris
        gdf.rename(columns={'area':'area_cell'},inplace=True)
        gdf=gpd.sjoin(gdf.loc[:,('IDENT','EGT','geometry','area_cell')],gdf2.loc[:,('insee_com','nom_com','code_dept','geometry')],how='left',predicate='within').dropna(subset='insee_com')
        gdf.drop(columns='index_right',inplace=True)

        # rename geounits for constitency with all French cities
        gdf.rename(columns={'IDENT':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'insee_com':'geo_unit'},inplace=True)
        gdf2.rename(columns={'insee_com':'geo_unit'},inplace=True)

        # load in IRIS shapefiles for each Department, then combine
        fp0='../../../projects/city_mobility/shapefiles/France other/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + deps[0] + '-2014/CONTOURS-IRIS_D0' + deps[0] + '.shp'
        iris_gdf0=gpd.read_file(fp0)
        fp1='../../../projects/city_mobility/shapefiles/France other/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + deps[1] + '-2014/CONTOURS-IRIS_D0' + deps[1] + '.shp'
        iris_gdf1=gpd.read_file(fp1)
        fp2='../../../projects/city_mobility/shapefiles/France other/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + deps[2] + '-2014/CONTOURS-IRIS_D0' + deps[2] + '.shp'
        iris_gdf2=gpd.read_file(fp2)
        fp3='../../../projects/city_mobility/shapefiles/France other/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + deps[3] + '-2014/CONTOURS-IRIS_D0' + deps[3] + '.shp'
        iris_gdf3=gpd.read_file(fp3)
        fp4='../../../projects/city_mobility/shapefiles/France other/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + '91' + '-2014/CONTOURS-IRIS_D0' + '91' + '.shp'
        iris_gdf4=gpd.read_file(fp4)
        fp5='../../../projects/city_mobility/shapefiles/France other/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + '95' + '-2014/CONTOURS-IRIS_D0' + '95' + '.shp'
        iris_gdf5=gpd.read_file(fp5)

        iris_gdf=gpd.GeoDataFrame(pd.concat([iris_gdf0,iris_gdf1,iris_gdf2,iris_gdf3,iris_gdf4,iris_gdf5]),crs=iris_gdf0.crs)
        iris_gdf.to_crs(crs0,inplace=True)
        
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        iris_gdf.set_geometry('center',inplace=True)

        # make a spatial join of the iris gdf and the more aggregated gdf - gdf2
        # there should not be replication of the iris codes, each iris code must be mapped to only one large sector, and each large sector will typically have >1 iris
        join3=gpd.sjoin(iris_gdf,gdf2,how='left',predicate='within').dropna()
        sec_iris=join3[['geo_unit','DCOMIRIS']].drop_duplicates()
        if sec_iris['DCOMIRIS'].value_counts().max()>1:
            print('iris code mapped to more than one sector in ' + city)
            sys.exit()

        # load in the population data by IRIS for 2012 and 2017
        fp = '../../MSCA_data/France_Population/base-ic-evol-struct-pop-2017_csv/base-ic-evol-struct-pop-2017.csv'
        pop17=pd.read_csv(fp,sep=';',dtype={'IRIS':str,'COM':str,'LAB_IRIS':str})
        pop17=pop17[['IRIS','P17_POP']]

        fp = '../../MSCA_data/France_Population/infra-population-2012/base-ic-evol-struct-pop-2012.csv'
        pop12=pd.read_csv(fp,sep=',')

        # extract the population only for the relevant dept
        pop12['Dep']=pop12['IRIS'].str[:2]
        pop17['Dep']=pop17['IRIS'].str[:2]
        pop12_dep=pop12.loc[pop12['IRIS'].isin(iris_gdf['DCOMIRIS']),:]
        pop17_dep=pop17.loc[pop17['IRIS'].isin(iris_gdf['DCOMIRIS']),:]

        # use outer merges here to keep rows with iris codes which may not exist in either 2012 or 2017 
        sec_iris_pop=sec_iris.merge(pop12_dep,left_on='DCOMIRIS',right_on='IRIS',how='outer')
        sec_iris_pop=sec_iris_pop.drop(columns='DCOMIRIS').merge(pop17_dep[['IRIS','P17_POP']],left_on='IRIS',right_on='IRIS',how='outer')
        # calculate total population by  survey sector
        sec_pop=sec_iris_pop.groupby('geo_unit')['P12_POP'].sum().to_frame().reset_index()
        sec_pop['P12_POP']=round(sec_pop['P12_POP'])
        sec_pop17=sec_iris_pop.groupby('geo_unit')['P17_POP'].sum().to_frame().reset_index()
        sec_pop17['P17_POP']=round(sec_pop17['P17_POP'])
        sec_pop=sec_pop.merge(sec_pop17)

        # merge the sum populations into the low-res gdf2
        gdf2=gdf2.merge(sec_pop)

        if survey_yr in [2010,2011,2012,2013,2014]:
            gdf2['Population']=gdf2['P12_POP']

        if survey_yr in [2015,2016,2017,2018]:
            gdf2['Population']=gdf2['P17_POP']

        gdf2['Density']=gdf2['Population']/gdf2['area']

        # merge in the population counts to the iris gdfs
        iris_gdf['area']=iris_gdf['polygon_iris'].area*1e-6
        iris_gdf=iris_gdf.drop(columns='IRIS').merge(pop17_dep.drop(columns='Dep'),left_on='DCOMIRIS',right_on='IRIS',how='outer')
        iris_gdf=iris_gdf.drop(columns='IRIS').merge(pop12_dep.drop(columns='Dep'),left_on='DCOMIRIS',right_on='IRIS',how='outer')
        iris_gdf['Density_2012']=iris_gdf['P12_POP']/iris_gdf['area']
        iris_gdf['Density_2017']=iris_gdf['P17_POP']/iris_gdf['area']

        # spatial join with iris polygons to combine the highres shapefile with the iris population density data
        gdf0=gdf.copy()
        gdf0['polygon_ZF']=gdf0.loc[:,'geometry']
        gdf0['center']=gdf0.centroid
        gdf0.set_geometry('center',inplace=True)
        iris_gdf_copy=iris_gdf.copy()
        iris_gdf_copy.set_geometry('polygon_iris',inplace=True)

        gdfj=gpd.sjoin(gdf0,iris_gdf_copy[['DCOMIRIS','polygon_iris','Density_2012','Density_2017']],how='left',predicate='within')
        gdfj.set_geometry('polygon_ZF',inplace=True)
        # check for nas
        if any(gdfj['Density_2017'].isna()):
            print(city + ' merged with loss of 2017 IRIS population data')

        if any(gdfj['Density_2012'].isna()):
            print(city + ' merged with loss of 2012 IRIS population data')

        if (any(gdfj['Density_2017'].isna()) == False) & (any(gdfj['Density_2012'].isna()) == False):
            print(city + ' merged without loss of any IRIS population data')
        # sub in 2017 data where 2012 data is na
        gdfj.loc[gdfj['Density_2012'].isna(),'Density_2012']=gdfj.loc[gdfj['Density_2012'].isna(),'Density_2017']
        gdfj['area']=gdfj['area_cell']*1e-6
        gdfj.drop(columns='area_cell',inplace=True)

        large=gdf2.loc[gdf2['area']>size_thresh,'geo_unit']
        
        sub=gdfj.loc[gdfj['geo_unit'].isin(large),]

        # reformat sub to be concatenated with gdf2
        sub=sub.loc[:,['geo_unit_highres','geometry','area','Density_2012']]
        sub.rename(columns={'geo_unit_highres':'geocode','Density_2012':'Density'},inplace=True)
        # sub['area']=sub['area_cell']*1e-6
        sub=sub.loc[:,('geocode','geometry','area','Density')]

        # reformate gdf2 to be concatenated with sub
        gdf2_concat=gdf2.loc[~gdf2['geo_unit'].isin(large),('geo_unit','geometry','area','Density')]
        gdf2_concat.rename(columns={'geo_unit':'geocode'},inplace=True)
        # make the concatenated gdf
        gdf2_concat=gpd.GeoDataFrame(pd.concat([gdf2_concat,sub], ignore_index=True))
        gdf2_concat['geocode']=gdf2_concat.loc[:,'geocode'].astype('str')
        

        # make the dictionary to translate the geocodes from the survey to the ones needed to merge with the geospatial data
        long=pd.DataFrame(gdf.loc[:,('geo_unit_highres','geo_unit')])
        # # insert to remove . or spaces from the 'geo_unit_highres'
        # long['geo_unit_highres']=long['geo_unit_highres'].astype('str').map(lambda x: x.replace('.','').replace(' ',''))
        long.loc[long['geo_unit_highres'].isin(sub['geocode']),'geo_unit']=long.loc[long['geo_unit_highres'].isin(sub['geocode']),'geo_unit_highres']
        geo_dict=long.set_index('geo_unit_highres').T.to_dict('records')[0]

        # save a figure of the mixed resolution geounits
        fig, ax = plt.subplots(figsize=(10,10))
        gdf2.plot(ax=ax,edgecolor='black')
        plt.title("Aggregated (blue) and higher resolution (red) geo-units: " + city) 
        ax.add_artist(ScaleBar(1))
        sub.plot(ax=ax, color='red',alpha=0.8,edgecolor='black')   
        plt.savefig('../outputs/density_geounits/'+ city + '_mixed.png',facecolor='w')
        # save a figure of the high resolution geounits
        fig, ax = plt.subplots(figsize=(10,10))
        gdf.plot(ax=ax,edgecolor='black',color='red')
        plt.title("High resolution (red) resolution geo-units: " + city) 
        ax.add_artist(ScaleBar(1))
        plt.savefig('../outputs/density_geounits/'+ city + '_high.png',facecolor='w')
        # save a figure of the low resolution geounit
        fig, ax = plt.subplots(figsize=(10,10))
        gdf2.plot(ax=ax,edgecolor='black')
        plt.title("Aggregated (blue) resolution geo-units: " + city) 
        ax.add_artist(ScaleBar(1))                   
        plt.savefig('../outputs/density_geounits/'+ city + '_low.png',facecolor='w')

        # save geodataframes and dictionary
        # save the shapefiles of population by aggregated sector
        gdf_low=gdf2.loc[:,('geo_unit','geometry','area','Population','Density')]
        gdf_low[['Population','Density']]=round(gdf_low.loc[:,('Population','Density')])
        gdf_low.sort_values(by='geo_unit',inplace=True)
        gdf_low.to_csv('../outputs/density_geounits/' + city + '_pop_density_lowres.csv',index=False)
        #  save the shapefiles of population by mix high-res and aggregated sector
        gdf2_concat.sort_values(by='geocode',inplace=True)
        gdf2_concat.to_csv('../outputs/density_geounits/' + city + '_pop_density_mixres.csv',index=False)
        # # save the shapefiles of the highres sector, not for Paris
        gdf_hi=gdfj.loc[:,('geo_unit_highres','geometry','area','Density_2012','Density_2017')]

        # save gridded data for Paris here if we want
        # gdf_hi.sort_values(by='geo_unit_highres',inplace=True)
        # gdf_hi.to_csv('../outputs/density_geounits/' + city + '_pop_density_highres.csv',index=False)

        # create and save the city boundary
        boundary=gpd.GeoDataFrame(geometry=[gdf_low['geometry'].unary_union], crs=crs0)
        boundary['crs']=crs0
        boundary.to_csv('../outputs/city_boundaries/' + city + '.csv',index=False)

        # save dictionary
        with open('../dictionaries/' + city + '_mixed_geocode.pkl', 'wb') as f:
            pickle.dump(geo_dict, f)

        # create and save some summary stats

        area_mixed=pd.DataFrame(gdf2_concat['area'].describe()).reset_index()
        area_hires=pd.DataFrame(gdf_hi['area'].describe()).reset_index()
        area_lores=pd.DataFrame(gdf_low['area'].describe()).reset_index()
        sums=pd.DataFrame(gdf_low[['area','Population']].sum()).reset_index()
        writer = pd.ExcelWriter('../outputs/density_geounits/summary_stats_' + city + '.xlsx', engine='openpyxl')

        # include all the dfs/sheets here, and then save
        area_mixed.to_excel(writer, sheet_name='area_mixres',index=False)
        area_hires.to_excel(writer, sheet_name='area_hires',index=False)
        area_lores.to_excel(writer, sheet_name='area_lores',index=False)
        sums.to_excel(writer, sheet_name='area_pop_sum',index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        writer.close()
        print('Finished extracting density and shapefiles in Paris and exiting script.')
        
    if city!='Paris':
        gdf['geo_unit_highres']=gdf['geo_unit_highres'].astype('str').map(lambda x: x.replace('.','').replace(' ',''))
        # there should not be replication of the iris codes, each iris code must be mapped to only one large sector, and each large sector will typically have >1 iris
        join3=gpd.sjoin(iris_gdf,gdf2,how='left',predicate='within').dropna()
        sec_iris=join3[['geo_unit','DCOMIRIS']].drop_duplicates()
        if sec_iris['DCOMIRIS'].value_counts().max()>1:
            print('iris code mapped to more than one sector in ' + city)
            sys.exit()

        # load in the population data by IRIS for 2012 and 2017
        fp = '../../MSCA_data/France_Population/base-ic-evol-struct-pop-2017_csv/base-ic-evol-struct-pop-2017.csv'
        pop17=pd.read_csv(fp,sep=';',dtype={'IRIS':str,'COM':str,'LAB_IRIS':str})
        pop17=pop17[['IRIS','P17_POP']]

        fp = '../../MSCA_data/France_Population/infra-population-2012/base-ic-evol-struct-pop-2012.csv'
        pop12=pd.read_csv(fp,sep=',')

        # extract the population only for the relevant dept
        pop12['Dep']=pop12['IRIS'].str[:2]
        pop17['Dep']=pop17['IRIS'].str[:2]
        pop12_dep=pop12.loc[pop12['Dep']==dep,:]
        pop17_dep=pop17.loc[pop17['Dep']==dep,:]

        # use outer merges here to keep rows with iris codes which may not exist in either 2012 or 2017 
        sec_iris_pop=sec_iris.merge(pop12_dep,left_on='DCOMIRIS',right_on='IRIS',how='outer')
        sec_iris_pop=sec_iris_pop.drop(columns='DCOMIRIS').merge(pop17_dep[['IRIS','P17_POP']],left_on='IRIS',right_on='IRIS',how='outer')
        # calculate total population by  survey sector
        sec_pop=sec_iris_pop.groupby('geo_unit')['P12_POP'].sum().to_frame().reset_index()
        sec_pop['P12_POP']=round(sec_pop['P12_POP'])
        sec_pop17=sec_iris_pop.groupby('geo_unit')['P17_POP'].sum().to_frame().reset_index()
        sec_pop17['P17_POP']=round(sec_pop17['P17_POP'])
        sec_pop=sec_pop.merge(sec_pop17)

        # mergre the sum populations into the low-res gdf2
        gdf2=gdf2.merge(sec_pop)

        if survey_yr in [2010,2011,2012,2013,2014]:
            gdf2['Population']=gdf2['P12_POP']

        if survey_yr in [2015,2016,2017,2018]:
            gdf2['Population']=gdf2['P17_POP']

        gdf2['Density']=gdf2['Population']/gdf2['area']

        # merge in the population counts to the iris gdfs
        iris_gdf['area']=iris_gdf['polygon_iris'].area*1e-6
        iris_gdf=iris_gdf.drop(columns='IRIS').merge(pop17_dep.drop(columns='Dep'),left_on='DCOMIRIS',right_on='IRIS',how='outer')
        iris_gdf=iris_gdf.drop(columns='IRIS').merge(pop12_dep.drop(columns='Dep'),left_on='DCOMIRIS',right_on='IRIS',how='outer')
        iris_gdf['Density_2012']=iris_gdf['P12_POP']/iris_gdf['area']
        iris_gdf['Density_2017']=iris_gdf['P17_POP']/iris_gdf['area']

        # spatial join with iris polygons to combine the highres shapefile with the iris population density data
        gdf0=gdf.copy()
        gdf0['polygon_ZF']=gdf0.loc[:,'geometry']
        gdf0['center']=gdf0.centroid
        gdf0.set_geometry('center',inplace=True)
        iris_gdf_copy=iris_gdf.copy()
        iris_gdf_copy.set_geometry('polygon_iris',inplace=True)

        large=gdf2.loc[gdf2['area']>size_thresh,'geo_unit']
        # here is the join
        gdfj=gpd.sjoin(gdf0,iris_gdf_copy[['DCOMIRIS','polygon_iris','Density_2012','Density_2017']],how='left',predicate='within')
        gdfj.set_geometry('polygon_ZF',inplace=True)
        # check for nas

        if any(gdfj['Density_2017'].isna()):
            print(city + ' merged with loss of 2017 IRIS population data')

        if any(gdfj['Density_2012'].isna()):
            print(city + ' merged with loss of 2012 IRIS population data')

        if (any(gdfj['Density_2017'].isna()) == False) & (any(gdfj['Density_2012'].isna()) == False):
            print(city + ' merged without loss of any IRIS population data')

        # check for 0 areas in gdfj, and if they exist, 'fold' them into the containing polygons
        if any(gdfj['area']==0) | any(gdfj['geometry'].geom_type.values == 'Point'):
            point_code2=pd.DataFrame(gdfj.loc['Point' == gdfj['geometry'].geom_type.values,'geo_unit_highres'])
            point_code2['containing']='0'
            if (city=='Lyon'):
                # remove point geom coded '247551' which is contained in the faulty geometry '247003', add in at the end
                point_code2=point_code2.loc[point_code2['geo_unit_highres']!='247551',:]

            for p in point_code2['geo_unit_highres']:
                indexer =point_code2[point_code2.geo_unit_highres==p].index.values[0]
                m=gdfj['geometry'].contains(gdfj.loc[indexer, 'geometry'])
                m=m[m.index!=indexer]
                mix=m.index[np.where(m)][0]
                point_code2.loc[indexer,'containing']=gdfj.loc[mix,'geo_unit_highres']

            if (city=='Lyon'):
                point_code2=pd.concat([point_code2, pd.DataFrame({'geo_unit_highres':['247551'],'containing':['247003']})])
            # remove the points from the gdfj gdf
            gdfj_nopoints=gdfj.loc[~gdfj['geo_unit_highres'].isin(point_code2['geo_unit_highres']),]

        sub=gdfj.loc[gdfj['geo_unit'].isin(large),]

        # reformat sub to be concatenated with gdf2
        sub=sub.loc[:,['geo_unit_highres','geometry','area','Density_2017']]
        sub.rename(columns={'geo_unit_highres':'geocode','Density_2017':'Density'},inplace=True)
        #sub['geocode']=sub['geocode'].astype('str').map(lambda x: x.replace('.','').replace(' ',''))

        # reformate gdf2 to be concatenated with sub
        gdf2_concat=gdf2.loc[~gdf2['geo_unit'].isin(large),('geo_unit','geometry','area','Density')]
        gdf2_concat.rename(columns={'geo_unit':'geocode'},inplace=True)
        gdf2_concat['geocode']=gdf2_concat.loc[:,'geocode'].astype('str')

        print('area distribution of retained large geo-units, ' + city)
        print(gdf2_concat['area'].describe())
        print('area distribution of smaller geo-units, ' + city)
        print(sub['area'].describe())
        # make the concatenated gdf
        gdf2_concat=gpd.GeoDataFrame(pd.concat([gdf2_concat,sub], ignore_index=True))

        # make the dictionary to translate the geocodes from the survey to the ones needed to merge with the geospatial data
        long=pd.DataFrame(gdf.loc[:,('geo_unit_highres','geo_unit')])
        # # insert to remove . or spaces from the 'geo_unit_highres'
        #long['geo_unit_highres']=long['geo_unit_highres'].astype('str').map(lambda x: x.replace('.','').replace(' ',''))
        long.loc[long['geo_unit_highres'].isin(sub['geocode']),'geo_unit']=long.loc[long['geo_unit_highres'].isin(sub['geocode']),'geo_unit_highres']

        geo_dict=long.set_index('geo_unit_highres').T.to_dict('records')[0]

        if 'point_code2' in locals(): # only if we had to replace points with their containing geometries
            # create the point_code dictionary
            #point_code2['geo_unit_highres']=point_code2['geo_unit_highres'].astype('str').map(lambda x: x.replace('.','').replace(' ',''))
            pc_dict2=point_code2.set_index('geo_unit_highres').T.to_dict('records')[0]
            
            long2=long.copy()
            long2.loc[long2['geo_unit_highres'].isin(point_code2['geo_unit_highres']),'geo_unit']=long2.loc[long2['geo_unit_highres'].isin(point_code2['geo_unit_highres']),'geo_unit_highres'].map(pc_dict2).values
            geo_dict_all=long2.set_index('geo_unit_highres').T.to_dict('records')[0]
            # save dictionary
            with open('../dictionaries/' + city + '_allpoly_geocode.pkl', 'wb') as f:
                pickle.dump(geo_dict_all, f)

        # save a figure of the mixed resolution geounits
        fig, ax = plt.subplots(figsize=(10,10))
        gdf2.plot(ax=ax,edgecolor='black')
        plt.title("Aggregated (blue) and higher resolution (red) geo-units: " + city) 
        ax.add_artist(ScaleBar(1))
        sub.plot(ax=ax, color='red',alpha=0.8,edgecolor='black')   
        plt.savefig('../outputs/density_geounits/'+ city + '_mixed.png',facecolor='w')
        # save a figure of the high resolution geounits
        fig, ax = plt.subplots(figsize=(10,10))
        gdf.plot(ax=ax,edgecolor='black',color='red')
        plt.title("High resolution (blue) resolution geo-units: " + city) 
        ax.add_artist(ScaleBar(1))
        plt.savefig('../outputs/density_geounits/'+ city + '_high.png',facecolor='w')
        # save a figure of the low resolution geounit
        fig, ax = plt.subplots(figsize=(10,10))
        gdf2.plot(ax=ax,edgecolor='black')
        plt.title("Aggregated (blue) resolution geo-units: " + city) 
        ax.add_artist(ScaleBar(1))                   
        plt.savefig('../outputs/density_geounits/'+ city + '_low.png',facecolor='w')

        # save geodataframes and dictionary
        # save the shapefiles of population by aggregated sector
        gdf_low=gdf2.loc[:,('geo_unit','geometry','area','Population','Density')]
        gdf_low[['Population','Density']]=round(gdf_low.loc[:,('Population','Density')])
        gdf_low.sort_values(by='geo_unit',inplace=True)
        gdf_low.to_csv('../outputs/density_geounits/' + city + '_pop_density_lowres.csv',index=False)
        #  save the shapefiles of population by mix high-res and aggregated sector
        gdf2_concat.sort_values(by='geocode',inplace=True)
        # insert
        #gdf2_concat['geocode']=gdf2_concat['geocode'].astype('str').map(lambda x: x.replace('.','').replace(' ',''))
        gdf2_concat.to_csv('../outputs/density_geounits/' + city + '_pop_density_mixres.csv',index=False)
        # save the shapefiles of the highres sector
        if 'gdfj_nopoints' in locals():
            gdf_hi=gdfj_nopoints.loc[:,('geo_unit_highres','geometry','area','Density_2012','Density_2017')]
        else: gdf_hi=gdfj.loc[:,('geo_unit_highres','geometry','area','Density_2012','Density_2017')]

        gdf_hi.sort_values(by='geo_unit_highres',inplace=True)
        # insert
        #gdf_hi['geo_unit_highres']=gdf_hi['geo_unit_highres'].astype('str').map(lambda x: x.replace('.','').replace(' ',''))
        gdf_hi.to_csv('../outputs/density_geounits/' + city + '_pop_density_highres.csv',index=False)

        # create and save the city boundary
        boundary=gpd.GeoDataFrame(geometry=[gdf_low['geometry'].unary_union], crs=crs0)
        if city in ['Dijon','Lille']:
            uuall=gdf_low.unary_union
            polyb=Polygon(uuall.geoms[0].exterior)
            boundary=gpd.GeoDataFrame(geometry=[polyb], crs=crs0)
        boundary['crs']=crs0
        boundary.to_csv('../outputs/city_boundaries/' + city + '.csv',index=False)

        # save dictionary
        with open('../dictionaries/' + city + '_mixed_geocode.pkl', 'wb') as f:
            pickle.dump(geo_dict, f)

        # create and save some summary stats

        area_mixed=pd.DataFrame(gdf2_concat['area'].describe()).reset_index()
        area_hires=pd.DataFrame(gdf_hi['area'].describe()).reset_index()
        area_lores=pd.DataFrame(gdf_low['area'].describe()).reset_index()
        sums=pd.DataFrame(gdf_low[['area','Population']].sum()).reset_index()
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

cities=pd.Series(['Clermont','Toulouse','Montpellier','Lyon','Nantes','Nimes','Lille','Dijon','Paris'])
cities.apply(french_density_shapefiles,args=(10,)) # args refers to the size threshold above which to divide large units into their smaller sub-components, e.g. 10km2