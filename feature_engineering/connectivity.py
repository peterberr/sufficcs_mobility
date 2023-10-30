# load in required packages
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.wkb import loads,dumps
from shapely.geometry import MultiPolygon, MultiPoint
from scipy.spatial import cKDTree, distance
from shapely.geometry import Point, LineString, Polygon
from shapely.wkt import loads,dumps
from shapely import speedups
speedups.enabled
from pyproj import CRS
from pysal.lib import weights
import sys
import networkx as nx
import pickle
import osmnx as ox
from citymob import import_csv_w_wkt_to_gdf, remove_holes, remove_invalid_geoms

crs0= 3035 # crs for boundary 
crs_osm = 4326

cities_all=['Berlin','Dresden','Düsseldorf','Frankfurt am Main','Kassel','Leipzig','Magdeburg','Potsdam','Clermont','Dijon','Lille','Lyon','Montpellier','Nantes','Nimes','Paris','Toulouse','Madrid','Wien']
countries=['Germany','Germany','Germany','Germany','Germany','Germany','Germany','Germany','France','France','France','France','France','France','France','France','France','Spain','Austria']

# (german) city postcodes
city_plz_fp='../dictionaries/city_postcode_DE.pkl'
a_file = open(city_plz_fp, "rb")
city_plz_dict = pickle.load(a_file)

# select connectivity stats of interest
metrics=['n','m','k_avg','intersection_density_km','clean_intersection_density_km','street_length_total','street_density_km','streets_per_node_avg','street_length_avg']
# n: count of nodes in graph
# m: count edges in graph
# k-avg:  graph’s average node degree (in-degree and out-degree)
# intersection_density_km: intersection_count per sq km
# clean_intersection_density_km: clean_intersection_count per sq km
# street_length_total -total street lenght in graph
# street_density_km - street_length_total per sq km
# streets_per_node_avg: average count of streets per node
# street_length_avg: average street length = street_length_total / street_segment_count

# see here for the explanation of 'clean' stats: that is what we want. https://github.com/gboeing/osmnx-examples/blob/main/notebooks/04-simplify-graph-consolidate-nodes.ipynb

# select network type, options are (string {"all_private", "all", "bike", "drive", "drive_service", "walk"}) , descriptions available here https://geoffboeing.com/2016/11/osmnx-python-street-networks/
nw_type='drive'

def network_plz(city_plz,metrics,nw_type,op):
    for i in range(len(op)):
        print(i)

        poly = city_plz.iloc[i].geometry
        try:
            graph_plz=ox.graph_from_polygon(poly,simplify=True,network_type=nw_type,retain_all=True)
            graph_proj=ox.project_graph(graph_plz)

            # get area
            graph_area=city_plz.loc[i,'geom_area']

            # get basic stats
            stats = ox.basic_stats(graph_proj,area=graph_area,clean_int_tol=10)

            # restrict to stats we are interested in
            stats1=dict((k, stats[k]) for k in metrics if k in stats)

            op.iloc[i,1:10]=pd.Series(stats1)

            # intersection node density. calculated as average node degree * nodes / area = total edges / area, where a single two-way street is counted as two edges. in other words the total nodal degree over the whole area.
            # calculated based on the description in section 2.3 of the SI of Stokes and Seto (2019) doi.org/10.1088/1748-9326/aafab8
            op.iloc[i,10]=stats['k_avg']*stats['n']/graph_area

            # bike lane share, calculated as a share of drive lane length, if there are any bike lanes
            cf = '["cycleway"]'
            # this might throw an error if no graph is found within the polygon
            graph_bike = ox.graph_from_polygon(poly, custom_filter=cf,retain_all=True)
            if len(graph_bike.edges)>0:
                graph_proj2=ox.project_graph(graph_bike)
                stats = ox.basic_stats(graph_proj2,area=graph_area,clean_int_tol=10)
                stats2=dict((k, stats[k]) for k in metrics if k in stats)
                op.iloc[i,11]=round(stats2['street_length_total']/stats1['street_length_total'],4)
            else:
                op.iloc[i,11]=0
        except:
            print('Graph error')

    return op


def conn(city):
    print(city)

    country=countries[cities_all.index(city)]
	# # 1. Read in boundaries

    if country=='Germany':
        # set the osmnx date configuration for German cities as end of 2018
        ox.config(overpass_settings='[out:json][timeout:90][date:"2018-12-31T23:59:00Z"]')
        fp = "../shapefiles/plz-5stellig.shp/plz-5stellig.shp"
        de_plz = gpd.read_file(fp)

        city_poly=de_plz.loc[(de_plz['plz'].isin(city_plz_dict[city]))]
        # make sure the city_plz is in the correct csv
        city_poly.to_crs(crs_osm,inplace=True)
        # change geocode label 
        if 'plz' in city_poly.columns:
            city_poly.rename(columns={'plz':'geocode'},inplace=True)

    elif city =='Wien':
        # set the osmnx date configuration for Wien as end of 2014
        ox.config(overpass_settings='[out:json][timeout:90][date:"2014-12-31T23:59:00Z"]')
        fp = '../outputs/density_geounits/'+city+'_pop_density.csv'
        city_poly=import_csv_w_wkt_to_gdf(fp,crs0,gc='geocode')


    elif city =='Madrid':
        # set the osmnx date configuration for Madrid as end of 2018
        ox.config(overpass_settings='[out:json][timeout:90][date:"2018-12-31T23:59:00Z"]')
        # for Madrid there was some decision required regarding whether to calc conn stats at high or mixed res. now i do it high res, consistent with FR cities outside Paris
        fp = '../outputs/density_geounits/'+city+'_pop_density_highres.csv' 
        city_poly=import_csv_w_wkt_to_gdf(fp,crs0,gc='geo_unit_highres')
        city_poly['geocode']=city_poly['geo_unit_highres'].astype('str')
        city_poly.drop(columns='geo_unit_highres',inplace=True)
    
    else:
        if city=='Montpellier':
            # set the osmnx date configuration for Montpellier as end of 2014
            ox.config(overpass_settings='[out:json][timeout:90][date:"2014-12-31T23:59:00Z"]')

            fp='../../MSCA_data/FranceRQ/lil-0937_Montpellier.csv/Doc/SIG/EDGT Montpellier_EDVM Beziers_Zones fines.mid'
            gdf_hi=gpd.read_file(fp)
            # boundary of gdf is set below for Montpellier
            gdf_hi=gdf_hi.to_crs(crs0)
            gdf_hi=remove_invalid_geoms(gdf_hi,crs0,'gdf',city)
            gdf_hi=remove_holes(gdf_hi,100,city)
            gdf_hi['area']=gdf_hi.area*1e-6

            # load a shapefile consisting only of the larger sector geo units
            fp2='../../MSCA_data/FranceRQ/lil-0937_Montpellier.csv/Doc/SIG/EDGT Montpellier_EDVM Beziers_DTIR.mid'
            gdf2=gpd.read_file(fp2)
            geo_unit=gdf2.loc[gdf2['D5_D10']=='01', 'DTIR'].sort_values().unique()
            # restrict surveys to the 'Montpellier Agglomération', with face to face interviews
            gdf2=gdf2.loc[(gdf2['ID_ENQ']==1) & (gdf2['DTIR'].isin(geo_unit)),]
            gdf_hi=gdf_hi.loc[(gdf_hi['ID_ENQ']==1) & (gdf_hi['NUM_SECTEUR'].isin(geo_unit)),]
            gdf_hi.rename(columns={'NUM_ZF_2013':'geocode'},inplace=True)

        if city=='Clermont':
            # set the osmnx date configuration for Montpellier as end of 2013 (even though survey is from 2012. But older data is less reliable from osm/osmnx)
            ox.config(overpass_settings='[out:json][timeout:90][date:"2013-12-31T23:59:00Z"]')

            fp='../../MSCA_data/FranceRQ/lil-0924_Clermont.csv/Doc/SIG/EDGT Clermont2012_DTIR.mid'
            gdf_hi=gpd.read_file(fp)
            gdf_hi.to_crs(crs0,inplace=True)
            geo_unit=gdf_hi['NUM_DTIR'].sort_values().unique()
            # limit to DTIRs 101:119, this corresponds to the PTU SMTC (Périmètre des Transports Urbains, Syndicat Mixte des Transports en Commun),
            # or 21 communes of  Clermont Communauté (replaced by Clermont Auvergne Métropole in 2018, but survey is from 2012)
            geo_unit=geo_unit[0:19]
            gdf_hi=gdf_hi.loc[gdf_hi['NUM_DTIR'].isin(geo_unit),] 
            # clean up (remove holes and make invalid geoms valid)
            gdf_hi=remove_invalid_geoms(gdf_hi,crs0,'gdf_hi',city)
            gdf_hi=remove_holes(gdf_hi,100,city)
            # calculate area of geo units for each gdf_hi
            gdf_hi['area']=gdf_hi.area*1e-6 # in km2
            gdf_hi.rename(columns={'DFIN':'geocode'},inplace=True)

        if city=='Lyon': # think i use the .tab file https://stackoverflow.com/questions/22218069/how-to-load-mapinfo-file-into-geopandas for mapinfo gis file
            # set the osmnx date configuration for Lyon as end of 2015 
            ox.config(overpass_settings='[out:json][timeout:90][date:"2015-12-31T23:59:00Z"]')

            fp='../../MSCA_data/FranceRQ/lil-1023_Lyon.csv/Doc/SIG/EDGT_AML2015_ZF_GT.TAB'
            gdf_hi=gpd.read_file(fp)
            # restrict to D12 zones 01 to 04 (DTIR<258), these are sufficiently close to the center of Lyon, and combined make up an area of 841km2 (quite largr).
            # It corresponds to all of Métropole de Lyon plus some additional nearby regions: Sepal, + a little bit of Ouest Rhône
            #geo_unit=gdf_hi.loc[gdf_hi['DTIR'].astype('int')<258,'DTIR'].sort_values().unique()
            geo_unit=gdf_hi.loc[gdf_hi['D10'].isin(['D12-01','D12-02']),'DTIR'].sort_values().unique()
            gdf_hi=gdf_hi.loc[gdf_hi['DTIR'].isin(geo_unit),] 
            gdf_hi=gdf_hi.to_crs(crs0)
            gdf_hi=remove_invalid_geoms(gdf_hi,crs0,'gdf_hi',city)
            gdf_hi=remove_holes(gdf_hi,100,city)
            gdf_hi['area']=gdf_hi.area*1e-6
            gdf_hi.rename(columns={'ZF2015_Nouveau_codage':'geocode'},inplace=True)
        
        if city=='Toulouse':
            # set the osmnx date configuration for Toulouse as end of 2013 
            ox.config(overpass_settings='[out:json][timeout:90][date:"2013-12-31T23:59:00Z"]')

            fp='../../MSCA_data/FranceRQ/lil-0933_Toulouse.csv/Doc/SIG/ZONE_FINE_EMD2013_FINAL4.mid' # hires gdf_hi
            gdf_hi=gpd.read_file(fp)
            gdf_hi.to_crs(crs0,inplace=True)
            geo_unit=gdf_hi['SECTEUR_EMD2013'].sort_values().unique()
            # restrict to Toulouse and it's near periphery, as shown on on pg2 of this document https://www.tisseo.fr/sites/default/files/Enquete_menage_deplacement.pdf, which makes up 781km2
            geo_unit=geo_unit[0:56]
            gdf_hi=gdf_hi.loc[gdf_hi['SECTEUR_EMD2013'].isin(geo_unit),] 
            # clean up
            gdf_hi=remove_invalid_geoms(gdf_hi,crs0,'gdf_hi',city)
            gdf_hi=remove_holes(gdf_hi,100,city)
            gdf_hi['area']=gdf_hi.area*1e-6
            gdf_hi.rename(columns={'ZF_SEC_EMD2013':'geocode'},inplace=True)

        if city=='Nantes': 
            # set the osmnx date configuration for Nantes as end of 2015
            ox.config(overpass_settings='[out:json][timeout:90][date:"2015-12-31T23:59:00Z"]')

            fp='../../MSCA_data/FranceRQ/lil-1024_Nantes.csv/Doc/SIG/EDGT44_2015_ZF.TAB'
            gdf_hi=gpd.read_file(fp)
            # restrict to Nantes Metropole
            geo_unit=gdf_hi.loc[gdf_hi['NOM_D10']=='Nantes Métropole','NUM_DTIR'].sort_values().unique()
            gdf_hi=gdf_hi.loc[gdf_hi['NUM_DTIR'].isin(geo_unit),] 
            gdf_hi=gdf_hi.to_crs(crs0)
            gdf_hi=remove_invalid_geoms(gdf_hi,crs0,'gdf_hi',city)
            gdf_hi=remove_holes(gdf_hi,100,city)
            gdf_hi['area']=gdf_hi.area*1e-6
            gdf_hi.rename(columns={'Id_zf_cerema':'geocode'},inplace=True)

        if city=='Nimes':
            # set the osmnx date configuration for Nimes as end of 2015
            ox.config(overpass_settings='[out:json][timeout:90][date:"2015-12-31T23:59:00Z"]')
            
            fp='../../MSCA_data/FranceRQ/lil-1135_Nimes.csv/Doc/SIG/EMD_Nimes_2014_2015_ZF.TAB'
            gdf_hi=gpd.read_file(fp)
            # restrict to Nimes city, which covers dtir 1-20. This is small (161km2). Could optionally extend to Communauté d'agglomération Nîmes Métropole, but that would leave us with a very low density (326/km2) and it is questionable whether that area is 'city'
            geo_unit=gdf_hi.loc[gdf_hi['NOM_DTIR']=='NIMES','NUM_DTIR'].sort_values().unique()  
            gdf_hi=gdf_hi.loc[gdf_hi['NUM_DTIR'].isin(geo_unit),] 
            gdf_hi=gdf_hi.to_crs(crs0)
            gdf_hi=remove_invalid_geoms(gdf_hi,crs0, 'gdf_hi',city)
            gdf_hi=remove_holes(gdf_hi,100,city)
            gdf_hi['area']=gdf_hi.area*1e-6
            gdf_hi.rename(columns={'NUM_ZF_2013':'geocode'},inplace=True)

        if city=='Dijon':
            # set the osmnx date configuration for Dijon as end of 2016
            ox.config(overpass_settings='[out:json][timeout:90][date:"2016-12-31T23:59:00Z"]')

            fp='../../MSCA_data/FranceRQ/lil-1214_Dijon.csv/Doc/SIG/EDGT_DIJON_2016_ZF.TAB'
            gdf_hi=gpd.read_file(fp)
            # restrict  to the Ville de Dijon and Grand Dijon hypercentre, which covers dtir 1-20. These all had face to face interviews.
            geo_unit=gdf_hi.loc[gdf_hi['NUM_D2']=='01','NUM_DTIR'].sort_values().unique() 
            gdf_hi=gdf_hi.loc[gdf_hi['NUM_DTIR'].isin(geo_unit),] 
            gdf_hi=gdf_hi.to_crs(crs0)
            gdf_hi=remove_invalid_geoms(gdf_hi,crs0, 'gdf_hi', city)
            gdf_hi=remove_holes(gdf_hi,100,city)
            gdf_hi['area']=gdf_hi.area*1e-6
            gdf_hi.rename(columns={'NUM_ZF':'geocode'},inplace=True)

        if city=='Lille':
            # set the osmnx date configuration for Lille as end of 2016
            ox.config(overpass_settings='[out:json][timeout:90][date:"2016-12-31T23:59:00Z"]')

            fp='../../MSCA_data/FranceRQ/lil-1152_Lille.csv/Doc/SIG/EDGT_LILLE_2016_ZF.TAB'
            gdf_hi=gpd.read_file(fp)
            # restrict to the métropole européenne de lille, which covers the French part of the eurumetripole, and includes also the cities of Tourcoing and Roubaix.
            geo_unit=gdf_hi.loc[gdf_hi['ST']<158,'ST'].sort_values().unique() 
            gdf_hi=gdf_hi.loc[gdf_hi['ST'].isin(geo_unit),] 
            gdf_hi=gdf_hi.to_crs(crs0)
            # clean up and add area
            gdf_hi=remove_invalid_geoms(gdf_hi,crs0, 'gdf_hi', city)
            gdf_hi=remove_holes(gdf_hi,100,city)
            gdf_hi['area']=gdf_hi.area*1e-6
            gdf_hi.rename(columns={'ZFIN2016F':'geocode'},inplace=True)

        if city == 'Paris':
            # set the osmnx date configuration for Paris as end of 2013 (even though survey is from 2010. But older data is less reliable from osm/osmnx)
            ox.config(overpass_settings='[out:json][timeout:90][date:"2013-12-31T23:59:00Z"]')

            fp = '../outputs/density_geounits/'+city+'_pop_density_lowres.csv'
            gdf_hi=import_csv_w_wkt_to_gdf(fp,crs0,gc='geo_unit')
            gdf_hi.rename(columns={'geo_unit':'geocode'},inplace=True)
            
        city_poly=gdf_hi.loc[:,['geocode','geometry','area']]

        # change geocode label 
        if 'geo_unit' in city_poly.columns: # for Wien and Paris at least
            city_poly.rename(columns={'geo_unit':'geocode'},inplace=True)
        city_poly['geocode']=city_poly['geocode'].astype('str').map(lambda x: x.replace('.','').replace(' ',''))

    if (country in ['Germany','Austria']) | (city=='Paris'):
        fp='../outputs/CenterSubcenter/'+city+'_dist.csv'
    else:
        fp='../outputs/CenterSubcenter/'+city+'_dist_hires.csv'
    dists=import_csv_w_wkt_to_gdf(fp,crs=3035,geometry_col='wgt_center',gc='geocode')
    dists['catch']=dists.geometry.buffer(1250)
    dists['geocode']=dists['geocode'].astype(str)

    city_poly=city_poly.merge(dists.loc[:,['geocode','wgt_center','catch']])
    city_poly.rename(columns={'geometry':'geoshape'},inplace=True)
    city_poly.rename(columns={'catch':'geometry'},inplace=True)
    city_poly.set_geometry('geometry')
    city_poly['geom_area']=city_poly.area # calculalte area to use as denominator for densities
    city_poly.to_crs(crs_osm,inplace=True)
         
    op=city_poly.loc[:, ['geocode']].copy()
    op[metrics]=np.nan
    # add in the 2 custom metrics of intersection node density and bike lane share
    op['int_node_dens']=np.nan 
    op['bike_lane_share']=np.nan

    print(city, len(op), ox.settings.overpass_settings)
    op2=network_plz(city_poly,metrics=metrics,nw_type='drive',op=op)

    op2.to_csv('../outputs/Connectivity/connectivity_stats_' + city + '.csv',index=False)
cities=pd.Series(['Nantes','Toulouse','Montpellier'])
#cities=pd.Series(cities_all)

cities.apply(conn) 