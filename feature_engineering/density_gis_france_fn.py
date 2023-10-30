# script to calculat population density by spatial units in French cities
# this is the new version which uses overlay functions to estimate local populations
# last update Peter Berrill Oct 25 2023

import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import CRS
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar
import pickle
from citymob import remove_holes, remove_invalid_geoms, remove_slivers
import sys

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

def french_density_shapefiles(city,size_thresh):
    crs0=3035
    #size_thresh=10
    if city=='Clermont':
        dep='63'
        survey_yr=2012
        fp='../../MSCA_data/FranceRQ/lil-0924_Clermont.csv/Doc/SIG/EDGT Clermont2012_DTIR.mid'
        gdf=gpd.read_file(fp)
        gdf.to_crs(crs0,inplace=True)
        geo_unit=gdf['NUM_DTIR'].sort_values().unique()
        # limit to DTIRs 101:119, this corresponds to the PTU SMTC (Périmètre des Transports Urbains, Syndicat Mixte des Transports en Commun), including Clermont ville and Hors-Clermont, see maps on pg 8 of http://www.authezat.fr/wp-content/uploads/2013/02/2013_01_28_dossier_presse_edgt_resultats.pdf
        # equivalent to 21 communes of  Clermont Communauté (replaced by Clermont Auvergne Métropole in 2018, but survey is from 2012)
        geo_unit=geo_unit[0:19]
        gdf=gdf.loc[gdf['NUM_DTIR'].isin(geo_unit),] 
        # clean up (remove holes and make invalid geoms valid)
        gdf=remove_invalid_geoms(gdf,crs0,'gdf',city)
        gdf=remove_holes(gdf,100,city)

        # make a shapefile consisting only of the larger sector geo units
        gdf2=gdf[['DTIR','geometry']].dissolve(by='DTIR').reset_index()
        # clean up
        gdf2=remove_invalid_geoms(gdf2,crs0,'gdf2',city)
        gdf2=remove_holes(gdf2,100,city)

        # calculate area of geo units for each gdf
        gdf['area_hires']=gdf.area*1e-6 # in km2
        gdf2['area_lores']=gdf2.area*1e-6

        # rename geounits for constitency with all French cities
        gdf.rename(columns={'DFIN':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'DTIR':'geo_unit'},inplace=True)
        gdf2.rename(columns={'DTIR':'geo_unit'},inplace=True)
        # load in IRIS shapefiles for Clermont Department
        fp='../../MSCA_data/France_Shapefiles/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf['iris_area']=iris_gdf.area*1e-6
        # iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        # iris_gdf.set_geometry('center',inplace=True)

    if city=='Montpellier':
        dep='34'
        survey_yr=2014
        fp='../../MSCA_data/FranceRQ/lil-0937_Montpellier.csv/Doc/SIG/EDGT Montpellier_EDVM Beziers_Zones fines.mid'
        gdf=gpd.read_file(fp)
        # boundary of gdf is set below for Montpellier
        gdf=gdf.to_crs(crs0)
        gdf=remove_invalid_geoms(gdf,crs0,'gdf',city)
        gdf=remove_holes(gdf,100,city)
        gdf['area_hires']=gdf.area*1e-6

        # load a shapefile consisting only of the larger sector geo units
        fp2='../../MSCA_data/FranceRQ/lil-0937_Montpellier.csv/Doc/SIG/EDGT Montpellier_EDVM Beziers_DTIR.mid'
        gdf2=gpd.read_file(fp2)
        geo_unit=gdf2.loc[gdf2['D5_D10']=='01', 'DTIR'].sort_values().unique()
        # restrict surveys to the 'Montpellier Agglomération', with face to face interviews
        gdf2=gdf2.loc[(gdf2['ID_ENQ']==1) & (gdf2['DTIR'].isin(geo_unit)),]
        gdf=gdf.loc[(gdf['ID_ENQ']==1) & (gdf['NUM_SECTEUR'].isin(geo_unit)),]
        gdf2=gdf2.to_crs(crs0)
        gdf2=remove_invalid_geoms(gdf2,crs0,'gdf2',city)
        gdf2=remove_holes(gdf2,100,city)
        gdf2['area_lores']=gdf2.area*1e-6
        # rename geounits for constitency with all French cities
        gdf.rename(columns={'NUM_ZF_2013':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'NUM_SECTEUR':'geo_unit'},inplace=True)
        gdf2.rename(columns={'DTIR':'geo_unit'},inplace=True)
        # load in IRIS shapefiles for Department
        fp='../../MSCA_data/France_Shapefiles/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf['iris_area']=iris_gdf.area*1e-6
        # iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        # iris_gdf.set_geometry('center',inplace=True)

    if city=='Lyon': # think i use the .tab file https://stackoverflow.com/questions/22218069/how-to-load-mapinfo-file-into-geopandas for mapinfo gis file
        dep='69'
        survey_yr=2015
        fp='../../MSCA_data/FranceRQ/lil-1023_Lyon.csv/Doc/SIG/EDGT_AML2015_ZF_GT.TAB'
        gdf=gpd.read_file(fp)
        # restrict to D12 zones 01 to 04 (DTIR<258), these are sufficiently close to the center of Lyon, and combined make up an area of 841km2 (quite largr).
        # It corresponds to all of Métropole de Lyon plus some additional nearby regions: Sepal, + a little bit of Ouest Rhône
        #geo_unit=gdf.loc[gdf['DTIR'].astype('int')<258,'DTIR'].sort_values().unique()
        geo_unit=gdf.loc[gdf['D10'].isin(['D12-01','D12-02']),'DTIR'].sort_values().unique()
        gdf=gdf.loc[gdf['DTIR'].isin(geo_unit),] 
        gdf=gdf.to_crs(crs0)
        gdf=remove_invalid_geoms(gdf,crs0,'gdf',city)
        gdf=remove_holes(gdf,100,city)
        gdf['area_hires']=gdf.area*1e-6

        # load a shapefile consisting only of the larger sector geo units
        fp2='../../MSCA_data/FranceRQ/lil-1023_Lyon.csv/Doc/SIG/EDGT_AML2015_DTIR.TAB'
        gdf2=gpd.read_file(fp2)
        # restrict to selected zones
        gdf2=gdf2.loc[gdf2['DTIR'].isin(geo_unit),] 
        gdf2=gdf2.to_crs(crs0)
        gdf2=remove_invalid_geoms(gdf2,crs0, 'gdf2', city)
        gdf2=remove_holes(gdf2,100,city)
        gdf2['area_lores']=gdf2.area*1e-6
        # rename geounits for constitency with all French cities
        gdf.rename(columns={'ZF2015_Nouveau_codage':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'DTIR':'geo_unit'},inplace=True)
        gdf2.rename(columns={'DTIR':'geo_unit'},inplace=True)
        # load in IRIS shapefiles for Lyon Department
        fp='../../MSCA_data/France_Shapefiles/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf['iris_area']=iris_gdf.area*1e-6
        # iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        # iris_gdf.set_geometry('center',inplace=True)

    if city=='Toulouse':
        dep='31'
        survey_yr=2013
        fp='../../MSCA_data/FranceRQ/lil-0933_Toulouse.csv/Doc/SIG/ZONE_FINE_EMD2013_FINAL4.mid' # hires gdf
        gdf=gpd.read_file(fp)
        gdf.to_crs(crs0,inplace=True)
        geo_unit=gdf['SECTEUR_EMD2013'].sort_values().unique()
        # restrict to Toulouse and it's near periphery, as shown on on pg2 of this document https://www.tisseo.fr/sites/default/files/Enquete_menage_deplacement.pdf, which makes up 781km2
        geo_unit=geo_unit[0:56]
        gdf=gdf.loc[gdf['SECTEUR_EMD2013'].isin(geo_unit),] 
        # clean up
        gdf=remove_invalid_geoms(gdf,crs0,'gdf',city)
        gdf=remove_holes(gdf,100,city)

        # make a shapefile consisting only of the larger sector geo units
        gdf2=gdf[['SECTEUR_EMD2013','geometry']].dissolve(by='SECTEUR_EMD2013').reset_index()
        gdf2=remove_invalid_geoms(gdf2,crs0, 'gdf2', city)
        gdf2=remove_holes(gdf2,100,city)
        # # calculate area of geo units for each gdf
        gdf['area_hires']=gdf.area*1e-6 # in km2
        gdf2['area_lores']=gdf2.area*1e-6
        # # rename geounits for constitency with all French cities
        gdf.rename(columns={'ZF_SEC_EMD2013':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'SECTEUR_EMD2013':'geo_unit'},inplace=True)
        gdf2.rename(columns={'SECTEUR_EMD2013':'geo_unit'},inplace=True)
        # # load in IRIS shapefiles for Department
        fp='../../MSCA_data/France_Shapefiles/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf['iris_area']=iris_gdf.area*1e-6
        # iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        # iris_gdf.set_geometry('center',inplace=True)

    if city=='Nantes': 
        dep='44'
        survey_yr=2015
        fp='../../MSCA_data/FranceRQ/lil-1024_Nantes.csv/Doc/SIG/EDGT44_2015_ZF.TAB'
        gdf=gpd.read_file(fp)

        # restrict to Nantes Metropole
        geo_unit=gdf.loc[gdf['NOM_D10']=='Nantes Métropole','NUM_DTIR'].sort_values().unique()
        gdf=gdf.loc[gdf['NUM_DTIR'].isin(geo_unit),] 
        gdf=gdf.to_crs(crs0)
        gdf=remove_invalid_geoms(gdf,crs0,'gdf',city)
        gdf=remove_holes(gdf,100,city)
        gdf['area_hires']=gdf.area*1e-6

        # load a shapefile consisting only of the larger sector geo units
        fp2='../../MSCA_data/FranceRQ/lil-1024_Nantes.csv/Doc/SIG/EDGT44_2015_DTIR.TAB'
        gdf2=gpd.read_file(fp2)
        # restrict to selected zones
        gdf2=gdf2.loc[gdf2['NUM_DTIR'].isin(geo_unit),] 
        gdf2=gdf2.to_crs(crs0)
        gdf2=remove_invalid_geoms(gdf2,crs0, 'gdf2', city)
        gdf2=remove_holes(gdf2,100,city)
        gdf2['area_lores']=gdf2.area*1e-6

        # rename geounits for constitency with all French cities
        gdf.rename(columns={'Id_zf_cerema':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'NUM_DTIR':'geo_unit'},inplace=True)
        gdf2.rename(columns={'NUM_DTIR':'geo_unit'},inplace=True)
        # load in IRIS shapefiles for Lyon Department
        fp='../../MSCA_data/France_Shapefiles/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf['iris_area']=iris_gdf.area*1e-6
        # iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        # iris_gdf.set_geometry('center',inplace=True)

    if city=='Nimes':
        dep='30'
        survey_yr=2015
        fp='../../MSCA_data/FranceRQ/lil-1135_Nimes.csv/Doc/SIG/EMD_Nimes_2014_2015_ZF.TAB'
        gdf=gpd.read_file(fp)
        # restrict to Nimes city, which covers dtir 1-20. This is small (161km2). Could optionally extend to Communauté d'agglomération Nîmes Métropole, but that would leave us with a very low density (326/km2) and it is questionable whether that area is 'city'
        geo_unit=gdf.loc[gdf['NOM_DTIR']=='NIMES','NUM_DTIR'].sort_values().unique()  
        gdf=gdf.loc[gdf['NUM_DTIR'].isin(geo_unit),] 
        gdf=gdf.to_crs(crs0)
        gdf=remove_invalid_geoms(gdf,crs0, 'gdf',city)
        gdf=remove_holes(gdf,100,city)
        gdf['area_hires']=gdf.area*1e-6

        # load a shapefile consisting only of the larger sector geo units
        fp2='../../MSCA_data/FranceRQ/lil-1135_Nimes.csv/Doc/SIG/EMD_Nimes_2014_2015_DTIR.TAB' 
        gdf2=gpd.read_file(fp2)
        gdf2=gdf2.to_crs(crs0)
        # restrict to selected zones
        gdf2=gdf2.loc[gdf2['NUM_DTIR'].isin(geo_unit),] 
        # clean up and add area
        gdf2=remove_invalid_geoms(gdf2,crs0, 'gdf2',city)
        gdf2=remove_holes(gdf2,100,city)
        gdf2['area_lores']=gdf2.area*1e-6

        # # rename geounits for constitency with all French cities
        gdf.rename(columns={'NUM_ZF_2013':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'NUM_DTIR':'geo_unit'},inplace=True)
        gdf2.rename(columns={'NUM_DTIR':'geo_unit'},inplace=True)
        # load in IRIS shapefiles for Lyon Department
        fp='../../MSCA_data/France_Shapefiles/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf['iris_area']=iris_gdf.area*1e-6
        # iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        # iris_gdf.set_geometry('center',inplace=True)
        
    if city=='Dijon':
        dep='21'
        survey_yr=2016 
        fp='../../MSCA_data/FranceRQ/lil-1214_Dijon.csv/Doc/SIG/EDGT_DIJON_2016_ZF.TAB'
        gdf=gpd.read_file(fp)
        # restrict  to the Ville de Dijon and Grand Dijon hypercentre, which covers dtir 1-20. These all had face to face interviews.
        geo_unit=gdf.loc[gdf['NUM_D2']=='01','NUM_DTIR'].sort_values().unique() 
        gdf=gdf.loc[gdf['NUM_DTIR'].isin(geo_unit),] 
        gdf=gdf.to_crs(crs0)
        gdf=remove_invalid_geoms(gdf,crs0, 'gdf', city)
        gdf=remove_holes(gdf,100,city)
        gdf['area_hires']=gdf.area*1e-6

        # load a shapefile consisting only of the larger sector geo units
        fp2='../../MSCA_data/FranceRQ/lil-1214_Dijon.csv/Doc/SIG/EDGT_DIJON_2016_DTIR.TAB' 
        gdf2=gpd.read_file(fp2)
        # restrict to selected zones
        gdf2=gdf2.loc[gdf2['NUM_DTIR'].isin(geo_unit),] 
        gdf2=gdf2.to_crs(crs0)
        gdf2=remove_invalid_geoms(gdf2,crs0, 'gdf2', city)
        gdf2=remove_holes(gdf2,100,city)
        gdf2['area_lores']=gdf2.area*1e-6

        # rename geounits for constitency with all French cities
        gdf.rename(columns={'NUM_ZF':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'NUM_DTIR':'geo_unit'},inplace=True)
        gdf2.rename(columns={'NUM_DTIR':'geo_unit'},inplace=True)
        # load in IRIS shapefiles 
        fp='../../MSCA_data/France_Shapefiles/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf['iris_area']=iris_gdf.area*1e-6
        # iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        # iris_gdf.set_geometry('center',inplace=True)

    if city=='Lille':
        dep='59'
        survey_yr=2015 
        fp='../../MSCA_data/FranceRQ/lil-1152_Lille.csv/Doc/SIG/EDGT_LILLE_2016_ZF.TAB'
        gdf=gpd.read_file(fp)
        # restrict to the métropole européenne de lille, which covers the French part of the eurumetripole, and includes also the cities of Tourcoing and Roubaix.
        geo_unit=gdf.loc[gdf['ST']<158,'ST'].sort_values().unique() 
        gdf=gdf.loc[gdf['ST'].isin(geo_unit),] 
        gdf=gdf.to_crs(crs0)
        # clean up and add area
        gdf=remove_invalid_geoms(gdf,crs0, 'gdf', city)
        gdf=remove_holes(gdf,100,city)
        gdf['area_hires']=gdf.area*1e-6

        # load a shapefile consisting only of the larger sector geo units
        fp2='../../MSCA_data/FranceRQ/lil-1152_Lille.csv/Doc/SIG/EDGT_LILLE_2016_DTIR.TAB'
        gdf2=gpd.read_file(fp2)
        # restrict to selected zones
        gdf2=gdf2.loc[gdf2['ST'].isin(geo_unit),] 
        gdf2=gdf2.to_crs(crs0)
        gdf2=remove_invalid_geoms(gdf2,crs0, 'gdf2', city)
        gdf2=remove_holes(gdf2,100,city)
        gdf2['area_lores']=gdf2.area*1e-6
        # # rename geounits for constitency with all French cities
        gdf.rename(columns={'ZFIN2016F':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'ST':'geo_unit'},inplace=True)
        gdf2.rename(columns={'ST':'geo_unit'},inplace=True)
        # load in IRIS shapefiles for Lyon Department
        fp='../../MSCA_data/France_Shapefiles/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + dep + '-2014/CONTOURS-IRIS_D0' + dep + '.shp'
        iris_gdf=gpd.read_file(fp)
        iris_gdf=iris_gdf.to_crs(crs0)
        # set the center of the IRIS polygons as the shapefile geometry 
        iris_gdf['center']=iris_gdf.centroid
        iris_gdf['iris_area']=iris_gdf.area*1e-6
        # iris_gdf.rename(columns={'geometry':'polygon_iris'},inplace=True)
        # iris_gdf.set_geometry('center',inplace=True)

    if city=='Paris': # first do for Paris 
        # define boundary as Paris, plus departments Hauts-de-Seine, Seine-St Deint, and Val-de-Marne, plus selected communes in the departments of Essonne and Val-d'Oise. 
        # Based on definition of Métropole du Grand Paris https://en.wikipedia.org/wiki/Paris
        deps=['75','92','93','94']
        ext_com=['91027','91326','91432','91479','91589','91687','95018']
        survey_yr=2010
        fp='../../MSCA_data/France_Shapefiles/code-postal-code-insee-2015/code-postal-code-insee-2015.shp'
        gdf0=gpd.read_file(fp)
        gdf0=gdf0.loc[:,('insee_com','nom_com','superficie','population','code_dept','nom_dept','code_reg','nom_reg','nom_de_la_c','geometry')].drop_duplicates()
        ext_com=['91027','91326','91432','91479','91589','91687','95018']
        gdf2=gdf0.loc[(gdf0['insee_com'].isin(ext_com)) | (gdf0['code_dept'].isin(deps)), ]

        gdf2=gdf2.to_crs(crs0)
        gdf2['area']=gdf2['geometry'].area*1e-6
        df2=pd.DataFrame(gdf2.drop(columns='geometry'))

        gdf_idf=gdf0.loc[gdf0['code_dept'].isin(['75','77','78','91','92','93','94','95']), ] # we never use this
        gdf_idf.to_crs(crs0,inplace=True)
        del gdf0

        fp='../../MSCA_data/FranceRQ/lil-0883_IleDeFrance.csv/Doc/Carreaux_shape_mifmid/carr100m.shp'
        gdf=gpd.read_file(fp)
        # assumed crs of the grid
        gdf.set_crs('epsg:27561',inplace=True)
        gdf.to_crs(crs0,inplace=True)
        gdf['area']=gdf.area

        # restrict higher res grid gdf to selected communes, i.e. those in Greater Paris
        gdf.rename(columns={'area':'area_cell'},inplace=True)
        gdf=gpd.sjoin(gdf.loc[:,('IDENT','EGT','geometry','area_cell')],gdf2.loc[:,('insee_com','nom_com','code_dept','geometry')],how='left',predicate='within').dropna(subset='insee_com')
        gdf.drop(columns='index_right',inplace=True)

        # rename geounits for constitency with all French cities
        gdf.rename(columns={'IDENT':'geo_unit_highres'},inplace=True)
        gdf.rename(columns={'insee_com':'geo_unit'},inplace=True)
        gdf2.rename(columns={'insee_com':'geo_unit'},inplace=True)

        # load in IRIS shapefiles for each Department, then combine
        fp0='../../MSCA_data/France_Shapefiles/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + deps[0] + '-2014/CONTOURS-IRIS_D0' + deps[0] + '.shp'
        iris_gdf0=gpd.read_file(fp0)
        fp1='../../MSCA_data/France_Shapefiles/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + deps[1] + '-2014/CONTOURS-IRIS_D0' + deps[1] + '.shp'
        iris_gdf1=gpd.read_file(fp1)
        fp2='../../MSCA_data/France_Shapefiles/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + deps[2] + '-2014/CONTOURS-IRIS_D0' + deps[2] + '.shp'
        iris_gdf2=gpd.read_file(fp2)
        fp3='../../MSCA_data/France_Shapefiles/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + deps[3] + '-2014/CONTOURS-IRIS_D0' + deps[3] + '.shp'
        iris_gdf3=gpd.read_file(fp3)
        fp4='../../MSCA_data/France_Shapefiles/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + '91' + '-2014/CONTOURS-IRIS_D0' + '91' + '.shp'
        iris_gdf4=gpd.read_file(fp4)
        fp5='../../MSCA_data/France_Shapefiles/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS_2-0__SHP_LAMB93_FXX_2014-01-01/CONTOURS-IRIS/1_DONNEES_LIVRAISON_2014/CONTOURS-IRIS_2-0_SHP_LAMB93_D0' + '95' + '-2014/CONTOURS-IRIS_D0' + '95' + '.shp'
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
        if sec_iris['DCOMIRIS'].value_counts().max()>1: # if this happens the code will be broken, this can happen in Lyon if including point geometries
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

        # get 2010 populations for Paris
        fp='../../MSCA_data/France_population_new/Paris_all_commune_2010.csv'
        pop10=pd.read_csv(fp,encoding='latin-1')
        pop10['geo_unit']=pop10['geo_unit'].astype(str)

        sec_pop=sec_pop.merge(pop10)

        # merge the sum populations into the low-res gdf2
        gdf2=gdf2.merge(sec_pop)
        # for Paris, we use 2010 population
        gdf2['Population']=gdf2['P10_POP']

        # if survey_yr in [2012,2013,2014]:
        #     gdf2['Population']=gdf2['P12_POP']

        # if survey_yr in [2015,2016,2017,2018]:
        #     gdf2['Population']=gdf2['P17_POP']

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
        # sub in 2017 data where 2012-2016 data is na
        # gdfj.loc[gdfj['Density_2012'].isna(),'Density_2012']=gdfj.loc[gdfj['Density_2012'].isna(),'Density_2017']
        # gdfj.loc[gdfj['Density_2013'].isna(),'Density_2013']=gdfj.loc[gdfj['Density_2013'].isna(),'Density_2017']
        # gdfj.loc[gdfj['Density_2014'].isna(),'Density_2014']=gdfj.loc[gdfj['Density_2014'].isna(),'Density_2017']
        # gdfj.loc[gdfj['Density_2015'].isna(),'Density_2015']=gdfj.loc[gdfj['Density_2015'].isna(),'Density_2017']
        # gdfj.loc[gdfj['Density_2016'].isna(),'Density_2013']=gdfj.loc[gdfj['Density_2016'].isna(),'Density_2017']
        gdfj['area']=gdfj['area_cell']*1e-6
        gdfj.drop(columns='area_cell',inplace=True)

        large=gdf2.loc[gdf2['area']>size_thresh,'geo_unit']
        
        sub=gdfj.loc[gdfj['geo_unit'].isin(large),]

        # reformat sub to be concatenated with gdf2
        if survey_yr in [2010,2011,2012,2013,2014]:
            sub=sub.loc[:,['geo_unit_highres','geometry','area','Density_2012','DCOMIRIS']]
            sub.rename(columns={'geo_unit_highres':'geocode','Density_2012':'Density'},inplace=True)

        if survey_yr in [2015,2016,2017,2018]:
            sub=sub.loc[:,['geo_unit_highres','geometry','area','Density_2017','DCOMIRIS']]
            sub.rename(columns={'geo_unit_highres':'geocode','Density_2017':'Density'},inplace=True)
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
        print('Finished extracting density and shapefiles in Paris and exiting script.')
        
    if city!='Paris':
        gdf['geo_unit_highres']=gdf['geo_unit_highres'].astype('str').map(lambda x: x.replace('.','').replace(' ',''))
        # there should not be replication of the iris codes, each iris code must be mapped to only one large sector, and each large sector will typically have >1 iris
        join3=gpd.sjoin(iris_gdf,gdf2,how='left',predicate='within').dropna()
        sec_iris=join3[['geo_unit','DCOMIRIS']].drop_duplicates()
        if sec_iris['DCOMIRIS'].value_counts().max()>1: # check why the code breaks here in Lyon
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

        iris_gdf=iris_gdf.drop(columns='IRIS').merge(pop17_dep,left_on='DCOMIRIS',right_on='IRIS',how='outer')
        iris_gdf=iris_gdf.drop(columns=['Dep','IRIS']).merge(pop12_dep,left_on='DCOMIRIS',right_on='IRIS',how='outer')
        print('N iris with pop data but no iris code: ', len(iris_gdf.loc[iris_gdf['DCOMIRIS'].isna()]))
        iris_gdf.dropna(subset='DCOMIRIS',inplace=True)
        iris_gdf['P12_POP'].fillna(0,inplace=True)
        iris_gdf['P17_POP'].fillna(0,inplace=True)

        # interpolate to get population in bw 2012 and 2017 if necessary.
        xp=[2012, 2017]
        iris_gdf['P13_POP']=iris_gdf['P14_POP']=iris_gdf['P15_POP']=iris_gdf['P16_POP']=np.nan
        for gu in iris_gdf['DCOMIRIS']:
            yp=iris_gdf.loc[iris_gdf['DCOMIRIS']==gu,['P12_POP','P17_POP']].values[0]
            iris_gdf.loc[iris_gdf['DCOMIRIS']==gu,['P13_POP','P14_POP','P15_POP','P16_POP']]=np.interp([2013,2014,2015,2016],xp,yp)

        # identify the correct population data to use
        if survey_yr==2012:
            iris_gdf['Population']=iris_gdf['P12_POP']

        if survey_yr==2013:
            iris_gdf['Population']=iris_gdf['P13_POP']

        if survey_yr==2014:
            iris_gdf['Population']=iris_gdf['P14_POP']

        if survey_yr==2015:
            iris_gdf['Population']=iris_gdf['P15_POP']

        if survey_yr==2016:
            iris_gdf['Population']=iris_gdf['P16_POP']

        if survey_yr==2017:
            iris_gdf['Population']=iris_gdf['P17_POP']

        # to deal with points, zero area geoms:
        if any(gdf['area_hires']==0) | any(gdf['geometry'].geom_type.values == 'Point'):
            gdf.loc[gdf['area_hires']==0,'geometry']=gdf.loc[gdf['area_hires']==0].buffer(5) # make it a 5m radius circle
            gdf.loc[gdf['area_hires']==0,'area_hires']=gdf.loc[gdf['area_hires']==0].area*1e-6

        # create overlay bw high-res gdf and census sections and population
        over_hi = gdf.overlay(iris_gdf, how='intersection')
        over_hi['area_over']=over_hi.area*1e-6
        # calculate population by overlay area
        over_hi['iris_density']=over_hi['Population']/over_hi['iris_area']
        over_hi['area_share']=over_hi['area_over']/over_hi['iris_area']
        over_hi['Pop_calc']=over_hi['area_share']*over_hi['Population']

        # sum population by highres gdf
        over_hi_sums=over_hi.groupby('geo_unit_highres')['Pop_calc'].sum().to_frame().reset_index()
        over_hi_sums=gpd.GeoDataFrame(over_hi_sums.merge(gdf.loc[:,['geo_unit_highres','geo_unit','area_hires','geometry']]))
        over_hi_sums['Density']=over_hi_sums['Pop_calc']/over_hi_sums['area_hires']
        over_hi_sums.sort_values(by='geo_unit_highres',inplace=True)

        # calculate population per low-res unit by summing the high-res
        over_lo_sums=over_hi_sums.groupby('geo_unit')['Pop_calc'].sum().reset_index()
        over_lo_sums.rename(columns={'Pop_calc':'Pop_calc_lo_sum'},inplace=True)
        over_lo_sums=gpd.GeoDataFrame(over_lo_sums.merge(gdf2.loc[:,['geo_unit','area_lores','geometry']]))
        over_lo_sums['Density']=over_lo_sums['Pop_calc_lo_sum']/over_lo_sums['area_lores']

        gdf_lo=over_lo_sums.copy()
        gdf_hi=over_hi_sums.copy()

        gdf_lo.rename(columns={'Pop_calc_lo_sum':'Population'},inplace=True) # ,'area_lores':'area'
        gdf_hi.rename(columns={'Pop_calc':'Population'},inplace=True) # ,'area_hires':'area'

        # now make the mixed res gdf
        size_thresh=10
        large=gdf_lo.loc[gdf_lo['area_lores']>size_thresh,'geo_unit']
        sub=gdf_hi.loc[gdf_hi['geo_unit'].isin(large),:]

        sub2=over_hi.loc[over_hi['geo_unit'].isin(large),:]
        sub22=sub2.copy()
        sub22.sort_values(by='area_over',ascending=False,inplace=True)
        sub22.drop_duplicates(subset='geo_unit_highres',keep='first',inplace=True)

        # make a list/dictionary of high-res geounits and corresponding IRIS codes, taking the IRIS which has largest area in the overlap
        hr_iris=sub22[['geo_unit_highres','DCOMIRIS']]
        hr_iris.sort_values(by='geo_unit_highres',inplace=True)
        hr_iris.reset_index(inplace=True,drop=True)

        # merge the IRIS codes into sub
        sub=sub.merge(hr_iris)
        # calculate sum pop by IRISH in sub
        sub_pop_sum=sub.groupby('DCOMIRIS')['Population'].sum().reset_index()

        # create dissolved sub, by IRIS
        sub_diss=sub.dissolve(by='DCOMIRIS').reset_index()
        sub_diss.sort_values(by='geo_unit_highres',inplace=True)
        sub_diss['diss_area']=sub_diss.area*1e-6

        # add in population to the dissolved df and calculate density
        sub_pop_sum.rename(columns={'Population':'Population_diss'},inplace=True)
        sub_diss=sub_diss.merge(sub_pop_sum)
        sub_diss['Density_diss']=sub_diss['Population_diss']/sub_diss['diss_area']

        # Make a dict of the sub codes so that the representative unit can be mapped to from each individual highres geounit
        sub_dict=sub_diss[['DCOMIRIS','geo_unit_highres']]
        sub_dict.rename(columns={'geo_unit_highres':'geo_unit_rep'},inplace=True)
        sub_dict=sub_dict.merge(sub[['DCOMIRIS','geo_unit_highres']])
        sub_dict=sub_dict.loc[:,['geo_unit_highres','geo_unit_rep']]
        sub_dict.sort_values(by='geo_unit_highres',inplace=True)

        sub_diss=sub_diss.loc[:,['geo_unit_highres','geometry','diss_area','Population_diss', 'Density_diss']]
        sub_diss.rename(columns={'geo_unit_highres':'geocode','diss_area':'area','Population_diss':'Population','Density_diss':'Density'},inplace=True)
        # finally create the mixed res gdf
        gdf_mix=gdf_lo.loc[~gdf_lo['geo_unit'].isin(large),('geo_unit','geometry','area_lores','Population','Density')]
        gdf_mix.rename(columns={'geo_unit':'geocode','area_lores':'area'},inplace=True)
        #gdf_mix['geocode']=gdf_mix['geocode'].astype(int).astype(str)
        gdf_mix['geocode']=gdf_mix['geocode'].astype(str)
        gdf_mix['source']='large_units'
        sub_diss['source']='small_units_agg'
        gdf_mix=gpd.GeoDataFrame(pd.concat([gdf_mix,sub_diss], ignore_index=True))
        gdf_mix=gdf_mix.sort_values(by='geocode')

        dict1=over_hi_sums.loc[~over_hi_sums['geo_unit_highres'].isin(hr_iris['geo_unit_highres']),['geo_unit_highres','geo_unit']]
        dict1.rename(columns={'geo_unit':'geo_unit_rep'},inplace=True)
        # make combined geodict
        geo_dict=pd.DataFrame(pd.concat([dict1,sub_dict]))
        geo_dict.sort_values(by='geo_unit_highres',inplace=True)
        geo_dict.reset_index(inplace=True,drop=True)

        if all(geo_dict['geo_unit_highres']==gdf_hi['geo_unit_highres']):
            print('dictionary contains all hires codes')
        geo_dict=geo_dict.set_index('geo_unit_highres').T.to_dict('records')[0]

        # # check for 0 areas in gdfj, and if they exist, 'fold' them into the containing polygons
        # if any(gdfj['area']==0) | any(gdfj['geometry'].geom_type.values == 'Point'):
        #     point_code2=pd.DataFrame(gdfj.loc['Point' == gdfj['geometry'].geom_type.values,'geo_unit_highres'])
        #     point_code2['containing']='0'
        #     if (city=='Lyon'):
        #         # remove point geom coded '247551' which is contained in the faulty geometry '247003', add in at the end
        #         point_code2=point_code2.loc[point_code2['geo_unit_highres']!='247551',:]

        #     for p in point_code2['geo_unit_highres']:
        #         indexer =point_code2[point_code2.geo_unit_highres==p].index.values[0]
        #         m=gdfj['geometry'].contains(gdfj.loc[indexer, 'geometry'])
        #         m=m[m.index!=indexer]
        #         mix=m.index[np.where(m)][0]
        #         point_code2.loc[indexer,'containing']=gdfj.loc[mix,'geo_unit_highres']

        #     if (city=='Lyon'):
        #         point_code2=pd.concat([point_code2, pd.DataFrame({'geo_unit_highres':['247551'],'containing':['247003']})])
        #     # remove the points from the gdfj gdf, not running this for now, as urban form metrics can be calculated later as a buffer around selected points, as necessary, and as is done in the lines ummediately above
        #     # gdfj_nopoints=gdfj.loc[~gdfj['geo_unit_highres'].isin(point_code2['geo_unit_highres']),]


        # save a figure of the mixed resolution geounits
        cmap = mpl.colors.ListedColormap(['#1f77b4', 'red'])
        fig, ax = plt.subplots(figsize=(10,10))
        gdf_mix.plot(ax=ax,column='source',edgecolor='black',cmap=cmap)
        plt.title("Aggregated (blue) and higher resolution (red) geo-units: " + city) 
        ax.add_artist(ScaleBar(1)) 
        plt.savefig('../outputs/density_geounits/'+ city + '_mixed.png',facecolor='w')
        plt.close()

        # save a figure of the high resolution geounits
        fig, ax = plt.subplots(figsize=(10,10))
        gdf.plot(ax=ax,edgecolor='black',color='red')
        plt.title("High resolution (blue) resolution geo-units: " + city) 
        ax.add_artist(ScaleBar(1))
        plt.savefig('../outputs/density_geounits/'+ city + '_high.png',facecolor='w')
        plt.close()
        # save a figure of the low resolution geounit
        fig, ax = plt.subplots(figsize=(10,10))
        gdf2.plot(ax=ax,edgecolor='black')
        plt.title("Aggregated (blue) resolution geo-units: " + city) 
        ax.add_artist(ScaleBar(1))                   
        plt.savefig('../outputs/density_geounits/'+ city + '_low.png',facecolor='w')
        plt.close()

        # save geodataframes and dictionary
        # save the shapefiles of population by aggregated sector
        gdf_low=gdf_lo.copy()
        gdf_low.rename(columns={'area_lores':'area'},inplace=True)
        gdf_low=gdf_low.loc[:,('geo_unit','geometry','area','Population','Density')]
        gdf_low[['Population','Density']]=round(gdf_low.loc[:,('Population','Density')])
        gdf_low.sort_values(by='geo_unit',inplace=True)
        gdf_low.to_csv('../outputs/density_geounits/' + city + '_pop_density_lowres.csv',index=False)

        #  save the shapefiles of population by mix high-res and aggregated sector
        gdf_hi.rename(columns={'area_hires':'area'},inplace=True)
        gdf_hi=gdf_hi.loc[:,('geo_unit_highres','geometry','area','Population','Density')]
        gdf_hi.sort_values(by='geo_unit_highres',inplace=True)
        gdf_hi.to_csv('../outputs/density_geounits/' + city + '_pop_density_highres.csv',index=False)
        
        # save mixres density file
        gdf_mix.to_csv('../outputs/density_geounits/' + city + '_pop_density_mixres.csv',index=False)

        # create and save the city boundary
        boundary=gpd.GeoDataFrame(geometry=[gdf_low['geometry'].unary_union], crs=crs0)
        if city != 'Lyon':
            boundary=remove_slivers(boundary)
        boundary['crs']=crs0
        print('saving boundary for ' + city)
        boundary.to_csv('../outputs/city_boundaries/' + city + '.csv',index=False)

        # save dictionary
        with open('../dictionaries/' + city + '_mixed_geocode.pkl', 'wb') as f:
            pickle.dump(geo_dict, f)

        # create and save some summary stats

        area_mixed=pd.DataFrame(gdf_mix['area'].describe()).reset_index()
        area_hires=pd.DataFrame(gdf_hi['area'].describe()).reset_index()
        area_lores=pd.DataFrame(gdf_low['area'].describe()).reset_index()
        sums=pd.DataFrame(gdf_low[['area','Population']].sum()).reset_index()
        sums['index'].replace({'Population':'population'},inplace=True)
        sums=pd.concat([sums,pd.DataFrame([{'index':'density',0:sums.iloc[1,1]/sums.iloc[0,1]}])])
        writer = pd.ExcelWriter('../outputs/density_geounits/summary_stats_' + city + '.xlsx', engine='openpyxl')

        # include all the dfs/sheets here, and then save
        area_mixed.to_excel(writer, sheet_name='area_mixre',index=False)
        area_hires.to_excel(writer, sheet_name='area_hires',index=False)
        area_lores.to_excel(writer, sheet_name='area_lores',index=False)
        
        sums.to_excel(writer, sheet_name='area_pop_sum',index=False)
        
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        writer.close()

        print('Finished extracting density and shapefiles for ' + city)
cities=pd.Series(['Nimes','Toulouse'])
cities.apply(french_density_shapefiles,args=(10,)) # args refers to the size threshold above which to divide large units into their smaller sub-components, e.g. 10km2. Make sure this is consistent with Madrid