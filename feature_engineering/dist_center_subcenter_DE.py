# script to calculate distances from postcodes to city centers and subcenters in German cities
# last update Peter Berrill Nov 25 2022

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.wkt import loads,dumps
from shapely import speedups
from shapely.geometry import Point, LineString, Polygon
speedups.enable()
from pyproj import CRS
from pysal.lib import weights
import pickle
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

def import_csv_w_wkt_to_gdf(path,crs,geometry_col='geometry'):
	'''
	Import a csv file with WKT geometry column into a GeoDataFrame

    Last modified: 12/09/2020. By: Nikola

    '''

	df = pd.read_csv(path,encoding='latin-1')
	gdf = gpd.GeoDataFrame(df, 
						geometry=df[geometry_col].apply(wkt.loads),
						crs=crs)
	return(gdf)

# inputs:
# 1km Germany INSPIRE grid. possible/equivalent source: https://gdz.bkg.bund.de/index.php/default/geographische-gitter-fur-deutschland-in-lambert-projektion-geogitter-inspire.html
# German postcode shapefiles. source https://www.suche-postleitzahl.org/plz-karte-erstellen
# German city-postcode dictionaries, loaded as pickle files
# city center point estimates
# city boundary shapefiles
# EUBUCCO data of building geometries and types. source https://eubucco.com/data/

# outputs:
# distance from the building-weighted centroid of each postcode to city center and subcenters, and built up volume per postcode as csv
# coordinates of the center and subcetenters
# figures showing the variation of building volume density (m2/km2) and identified subcenters, and distances from postcodes to centers and subcenters

crs0=3035

# load in the 1km grid from Inspire for Germany, and use this grid to estimate the subcenters.
grid=gpd.read_file('C:/Users/peter/Documents/general_GIS_files/Germany/Inspire_1km_grid/DE_Gitter_ETRS89_LAEA_1km.shp') 
# ensure grid is in the common crs
grid=grid.to_crs(crs0)
# calculate grid cell areas
grid['area_cell']=grid['geometry'].area
grid['grid_id'] = grid.index

# read in German postcotde
fp = "C:/Users/peter/Documents/projects/city_mobility/shapefiles/plz-5stellig.shp/plz-5stellig.shp"
plz = gpd.read_file(fp)
plz=plz.to_crs(crs0)

plz_poly=plz.copy()
plz_cent=plz.copy()
plz_cent['geometry']=plz_cent.centroid
del plz

# load the dictionary of city postcodes, and select only postcodes within the selected city
city_plz_fp='C:/Users/peter/Documents/projects/city_mobility/dictionaries/city_postcode_DE.pkl'
a_file = open(city_plz_fp, "rb")
city_plz_dict = pickle.load(a_file)

# read in shapefile of user-defined city centers
centers=import_csv_w_wkt_to_gdf('../source/citycenters/centers.csv',crs=4326)
centers.to_crs(crs0,inplace=True)

def subcenters(city):
    print(city)
    city_poly=plz_poly.loc[(plz_poly['plz'].isin(city_plz_dict[city])),:].copy()
    fp='../../MSCA_data/BuildingsDatabase/germany/' + city + '_geom.csv' 
    # fp='C:/Users/peter/Documents/projects/MSCA_data/BuildingsDatabase/AllCities/share/' + city + '/' + city + '/' + city + '_geom.csv'  
    buildings_gdf=import_csv_w_wkt_to_gdf(fp,crs=crs0)
    # load in the attributes file containing building heights
    fp='../../MSCA_data/BuildingsDatabase/germany/' + city + '_attrib.csv'  
    at=pd.read_csv(fp)
    # read in file of city boundaries
    fp='../outputs/city_boundaries/' + city + '.csv'  
    gdf_boundary = import_csv_w_wkt_to_gdf(fp,crs=crs0,geometry_col='geometry')
    # check how many buildings are inside the boundary
    check=buildings_gdf.within(gdf_boundary.at[0,'geometry'])
    # see how many buildings are not inside the city boundary
    print('N buildings: ',len(check))
    print('Buildings within boundary: ',sum(check))
    print('Percent of buildings within boundary: ', sum(check)/len(check))


    # calculate the area of the buidlings from the database
    buildings_gdf["area"] = buildings_gdf["geometry"].area
    # calculate the centerpoint of each building geometry, 
    buildings_gdf["center"] = buildings_gdf["geometry"].centroid
    # For now, from the attributes file, we are currently only interested in building id and height. If type is fully available later we will include that
    at=at.loc[:,('id','height')]
    # merge the building attributes and heights
    df=pd.DataFrame(buildings_gdf.merge(at,on='id'))
    # calculate building volumes
    df['volume']=df['area']*df['height']
    # convert to Geodataframe
    gdf=gpd.GeoDataFrame(df,geometry='geometry',crs=crs0)

    #### Now we move onto the next step, sum up values by grid cell and then apply thresholds to identify sub-centers ###########
    # spatial join the geodataframe containing building geometries and locations with the grid
    jn=gpd.sjoin(gdf,grid,how="left",predicate="intersects")

    # clean up the joined gdf a bit
    #jn=jn.rename(columns={"index_right": "grid_id"})
    print(jn.columns)
    # keep only desired columns
    jn=jn.loc[:,('id','geometry','area','center','height','volume','grid_id','ID_1km','area_cell')]

    # sum volumes by grid cell
    sum_vol=jn.groupby('grid_id')['volume'].sum()
    # sum number of buildings by crid cell
    sum_bld=jn.groupby('grid_id')['id'].count()
    # sum building footprints by grid cell
    sum_fp=jn.groupby('grid_id')['area'].sum()

    # make a 'result' data frame summing up the number of buildings, footprint, and built-up volume in each grid cell
    result = pd.DataFrame({'grid_id': sum_vol.index, 'Num. Bldgs': sum_bld, 'Sum Bldg Footprint (m2)': sum_fp, 'Sum Bldg Volume (m3)': sum_vol })
    result.reset_index(drop=True, inplace=True)

    # add an id colum for grid, to enable merging with result
    #grid['grid_id'] = grid.index

    # make a geodataframe with sum of buildings, volume, and footprints per each grid cell. 'geometry' here refers to the grid cells
    result_gdf=gpd.GeoDataFrame(result.merge(grid.loc[:,('grid_id','geometry','area_cell')],on='grid_id'),geometry='geometry',crs=crs0)
    #result_gdf.drop(columns=['id'],axis=1,inplace=True)

    # define centres of each grid cell
    result_gdf['centerpoint']=result_gdf['geometry'].centroid
    # sort grid cells by volume
    result_sorted=result_gdf.sort_values('Sum Bldg Volume (m3)',ascending=False)
    result_sorted['volume density']=result_sorted['Sum Bldg Volume (m3)']/result_sorted['area_cell']

    cp0=centers.loc[centers['City']==city,:]
    center_cell=gpd.sjoin(result_gdf, cp0)
    # define id and index of center
    center_id=center_cell['grid_id'].values[0]
    center_ix=center_cell['grid_id'].index[0]


    # we don't use the cell of max density to identify the city center anymore, instead we use the manually identified city centers, in cp0
    # cp=result_sorted.drop('geometry',axis=1)
    # cp=cp.rename(columns={"centerpoint": "geometry"})
    # # identify the centerpoint of the cell with highest building volume - i.e. the city center
    # cp=cp.iloc[0:1] # extract only the first row

    # calculate distance to center based on the distance of each cell from the center of the cell containing the manually identified center
    result_gdf['dist2center']=result_gdf.geometry.apply(lambda g: result_gdf.loc[result_gdf['grid_id']==center_id,'centerpoint'].distance(g))
    result_gdf['DistGroup']='uncategorized'

    result_gdf.loc[result_gdf['dist2center']<1001,'DistGroup']='0-1'
    result_gdf.loc[(result_gdf['dist2center']<2001) & (result_gdf['dist2center']>1000),'DistGroup']='1-2'
    result_gdf.loc[(result_gdf['dist2center']<3001) & (result_gdf['dist2center']>2000),'DistGroup']='2-3'
    result_gdf.loc[(result_gdf['dist2center']<4001) & (result_gdf['dist2center']>3000),'DistGroup']='3-4'
    result_gdf.loc[(result_gdf['dist2center']<5001) & (result_gdf['dist2center']>4000),'DistGroup']='4-5'
    result_gdf.loc[(result_gdf['dist2center']<6001) & (result_gdf['dist2center']>5000),'DistGroup']='5-6'
    result_gdf.loc[(result_gdf['dist2center']<7001) & (result_gdf['dist2center']>6000),'DistGroup']='6-7'
    result_gdf.loc[(result_gdf['dist2center']<8001) & (result_gdf['dist2center']>7000),'DistGroup']='7-8'
    result_gdf.loc[(result_gdf['dist2center']<9001) & (result_gdf['dist2center']>8000),'DistGroup']='8-9'
    result_gdf.loc[(result_gdf['dist2center']<10001) & (result_gdf['dist2center']>9000),'DistGroup']='9-10'
    result_gdf.loc[(result_gdf['dist2center']<11001) & (result_gdf['dist2center']>10000),'DistGroup']='10-11'
    result_gdf.loc[(result_gdf['dist2center']<12001) & (result_gdf['dist2center']>11000),'DistGroup']='11-12'
    result_gdf.loc[(result_gdf['dist2center']<13001) & (result_gdf['dist2center']>12000),'DistGroup']='12-13'
    result_gdf.loc[(result_gdf['dist2center']<14001) & (result_gdf['dist2center']>13000),'DistGroup']='13-14'
    result_gdf.loc[(result_gdf['dist2center']<15001) & (result_gdf['dist2center']>14000),'DistGroup']='14-15'
    result_gdf.loc[(result_gdf['dist2center']<16001) & (result_gdf['dist2center']>15000),'DistGroup']='15-16'
    result_gdf.loc[(result_gdf['dist2center']<17001) & (result_gdf['dist2center']>16000),'DistGroup']='16-17'
    result_gdf.loc[(result_gdf['dist2center']<18001) & (result_gdf['dist2center']>17000),'DistGroup']='17-18'
    result_gdf.loc[(result_gdf['dist2center']<19001) & (result_gdf['dist2center']>18000),'DistGroup']='18-19'
    result_gdf.loc[(result_gdf['dist2center']<20001) & (result_gdf['dist2center']>19000),'DistGroup']='19-20'
    result_gdf.loc[result_gdf['dist2center']>20000,'DistGroup']='20+'

    # make a grouped dataframe, grouping by distance to center
    grouped_gdf=result_gdf.groupby('DistGroup')
    # calculate the cutoffs per group and identify in the results_gdf whether each cell is a center, or not.
    # the .transform() method is key here for calculating mean and SD for each distance band

    # here we define which 'CO' multiplier we use, this is the factor we multiply the standard deviation by to calculate cutoff thresholds which determine whether individual cells are subcenters or not
    # this is a crucial subjective assumption which will influence the identification of sub-centers.
    # The threshold for identifiying cutoffs is: cutoff = group_mean + CO*group_SD
    # See this paper by Taubenböck et al for the origin of this distance based threshold for identifying city subcenters (http://dx.doi.org/10.1016/j.compenvurbsys.2017.01.005)

    CO=1 

    # First calculate distance based sub-centers, i.e. points of high concentration of built-up volumne
    # That means, within each distance group, identify which cells are above the cutoff (defined as group mean + 1SD) of built-up volume per km2
    result_gdf['group_mean'] = grouped_gdf['Sum Bldg Volume (m3)'].transform('mean')
    result_gdf['group_SD'] = grouped_gdf['Sum Bldg Volume (m3)'].transform('std')
    result_gdf['cutoff'] = result_gdf['group_mean'] + CO*result_gdf['group_SD'] 
    result_gdf['isCenter_dist'] = result_gdf['Sum Bldg Volume (m3)'] > result_gdf['cutoff']
    # # make sure the centerpoint is included as a center by the distance based method, now we don't do this
    # result_gdf.loc[result_gdf['grid_id']==center_id,'isCenter_dist']=True

    # Second calulate a city-wide threshold for identifying subcenter cells.
    # This is calculated as the mean + CO*SD of the built up volume of all cells throughout the city
    cut_regional=CO*sum_vol.std()+sum_vol.mean()
    result_gdf['isCenter_regional'] = result_gdf['Sum Bldg Volume (m3)'] > cut_regional
    # finally, identify which cells are identified as centers by both the regional and distance based approaches.
    result_gdf['isCenter_regional_dist'] = result_gdf['isCenter_regional'] & result_gdf['isCenter_dist']

    # set a 1km buffer around the centerpoint and no grid cells intersecting with that buffer can be identified as a subcenter
    cen_buff=cp0.loc[:,('City', 'geometry')].copy()
    cen_buff['buff']=cen_buff.buffer(1000)
    cen_buff.set_geometry('buff',inplace=True)
    cen_buff.drop(columns='geometry',inplace=True)
    remove=gpd.sjoin(result_gdf,cen_buff)['grid_id']
    result_gdf.loc[result_gdf['grid_id'].isin(remove), ('isCenter_dist','isCenter_regional','isCenter_regional_dist')]=False

    # state how many cells have cleared the cutoff by regional, distance, and combined regional and distance based approaches.
    print('regional centers: ', sum(result_gdf['isCenter_regional']))
    print('dist-based centers: ', sum(result_gdf['isCenter_dist']))
    print('regional &  dist-based centers: ', sum(result_gdf['isCenter_regional_dist']))
    # create dataframes contaning only the cells which pass the different thresholds
    centers_regional_dist=result_gdf.loc[result_gdf['isCenter_regional_dist']==True,]
    centers_dist=result_gdf.loc[result_gdf['isCenter_dist']==True,]
    centers_regional=result_gdf.loc[result_gdf['isCenter_regional']==True,]

    ### So now we have identified the grid cell identified as centers using regional and distance-based approaches, and by both. ######
    # Next step is to turn the subcenter areas into individual points. 
    # First I have to turn contiguous cells into single areas, and then identify a single point identifying that sub-center.
    # I will do this using queen based contiguity.
    wqrd=weights.Queen.from_dataframe(centers_regional_dist)
    wqd=weights.Queen.from_dataframe(centers_dist)
    wqr=weights.Queen.from_dataframe(centers_regional)

    # classify each center cell into the group it is a member of, based on it's proximity a la Queen contiguity
    # This throws a 'value is trying to be set on a copy of a slice from a DataFrame' error which I cannot avoid.
    centers_regional_dist['contig_group']=wqrd.component_labels
    centers_dist['contig_group']=wqd.component_labels
    centers_regional['contig_group']=wqr.component_labels

    # retrieve the index id of the cell with the greatest volume in each contiguous group - this will be the 'center' cell of the city subcenter
    max_ids_RD=centers_regional_dist.groupby('contig_group')['Sum Bldg Volume (m3)'].transform('idxmax').unique()
    max_ids_D=centers_dist.groupby('contig_group')['Sum Bldg Volume (m3)'].transform('idxmax').unique()
    max_ids_R=centers_regional.groupby('contig_group')['Sum Bldg Volume (m3)'].transform('idxmax').unique()
    # Create GeoDataFrames of the centerpoints of each city subcenter calculated with the three different methods
    cp_RD=centers_regional_dist.loc[max_ids_RD].set_geometry('centerpoint')
    cp_D=centers_dist.loc[max_ids_D].set_geometry('centerpoint')
    cp_R=centers_regional.loc[max_ids_R].set_geometry('centerpoint')

    # now calculate the centre of mass (or volume) of each postcode, as the weighted mean xy coordinates, with built-up volumne used as the weights.
    # first make a grid of 100m side
    xmin,ymin,xmax,ymax =  gdf_boundary.total_bounds
    # define grid with cells side 100 (m)
    width = 100
    height = 100
    rows = int(np.ceil((ymax-ymin) /  height))
    cols = int(np.ceil((xmax-xmin) / width))
    XleftOrigin = xmin
    XrightOrigin = xmin + width
    YtopOrigin = ymax
    YbottomOrigin = ymax- height
    polygons = []
    for i in range(cols):
        Ytop = YtopOrigin
        Ybottom =YbottomOrigin
        for j in range(rows):
            polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])) 
            Ytop = Ytop - height
            Ybottom = Ybottom - height
        XleftOrigin = XleftOrigin + width
        XrightOrigin = XrightOrigin + width

    grid_100 = gpd.GeoDataFrame({'geometry':polygons},crs=crs0)
    grid_100['id_grid']=grid_100.index
    grid_100['area_cell']=grid_100.area

    # spatial join the geodataframe containing building geometries and locations with the grid
    jn2=gpd.sjoin(gdf,grid_100,how="left",predicate="intersects")
    # keep only desired columns
    jn2=jn2.loc[:,('id_grid','geometry','area','center','height','volume','area_cell')].copy()

    # sum volumes by grid cell
    sum_vol_g2=pd.DataFrame(jn2.groupby('id_grid')['volume'].sum())
    sum_vol_g2.reset_index(inplace=True)

    grid_100=grid_100.merge(sum_vol_g2)
    grid_100['centroid']=grid_100.centroid
    grid_100.rename(columns={'geometry':'cell_geom'},inplace=True)
    grid_100.set_geometry('centroid',inplace=True,crs=crs0)

    #city_poly=plz_poly.loc[(plz_poly['plz'].isin(city_plz_dict[city])),:].copy()
    # work-around (temporary) to adjust for the fact that we currently only have buildisng stock data for postcodes within official city boundaries
    city_poly.loc[:,'bldg_data']='Yes'
    if city=='Berlin':
        city_poly.loc[city_poly['plz'].isin(['15366','14513','14612']),'bldg_data']='No'
    if city=='Dresden':
        city_poly.loc[city_poly['plz'].isin(['01445','01705']),'bldg_data']='No'
    if city=='Düsseldorf':
        city_poly.loc[city_poly['plz'].isin(['40699','41460','40878','40721','40296','40789','40764','40880','40667']),'bldg_data']='No'
    if city=='Frankfurt am Main':
        city_poly.loc[city_poly['plz'].isin(['61118','65760','63263','61348','61352','65843']),'bldg_data']='No'
    if city=='Kassel':
        city_poly.loc[city_poly['plz'].isin(['34246','34225']),'bldg_data']='No'
    if city=='Leipzig':
        city_poly.loc[city_poly['plz'].isin(['04416']),'bldg_data']='No'
    if city=='Magdeburg':
        city_poly.loc[city_poly['plz'].isin(['39179']),'bldg_data']='No'
    # spatial join the geodataframe containing building volumnes by 100m grid cells with the polygons of postcodes
    jn3=gpd.sjoin(grid_100,city_poly,how="left",predicate="intersects")
    jn3=jn3.loc[:,('id_grid','cell_geom','centroid','volume','plz','note','bldg_data')].copy()

    # calculating weighted center
    jn3['x_center']=jn3.centroid.map(lambda p: p.x)
    jn3['y_center']=jn3.centroid.map(lambda p: p.y)
    jn3['x_wgt']=jn3['x_center']*jn3['volume']
    jn3['y_wgt']=jn3['y_center']*jn3['volume']

    plz_vol=pd.DataFrame(jn3.groupby('plz')['volume'].sum())#
    plz_vol.reset_index(drop=False,inplace=True)

    # create gdf of weighted centers, defined by the points x_mean, y_mean
    data=[]
    for pc in city_poly['plz']:
        if (city_poly.loc[city_poly['plz']==pc,'bldg_data']=='Yes').bool():
                x_mean=jn3.loc[jn3['plz']==pc,'x_wgt'].sum()/jn3.loc[jn3['plz']==pc,'volume'].sum()
                y_mean=jn3.loc[jn3['plz']==pc,'y_wgt'].sum()/jn3.loc[jn3['plz']==pc,'volume'].sum()
                p=Point([x_mean,y_mean])
        if (city_poly.loc[city_poly['plz']==pc,'bldg_data']=='No').bool():
            p=city_poly.loc[city_poly['plz']==pc,].centroid.values[0]

        data.append([pc,p])

    # create dataframe containing the weighted centers, volume, and then calculate volume density for each postcode
    plz_wgt_cent=gpd.GeoDataFrame(data,crs=crs0,columns =['plz','geometry'])
    plz_wgt_cent.rename(columns={'geometry':'wgt_center'},inplace=True)
    plz_poly2=plz_poly.merge(plz_wgt_cent).copy() #https://stackoverflow.com/questions/10851906/python-3-unboundlocalerror-local-variable-referenced-before-assignment
    plz_poly2=plz_poly2.merge(plz_vol)
    plz_poly2['area']=plz_poly2.area*1e-6
    # here calculate building volume density (m3/m2) for each postcode
    plz_poly2['build_vol_density']=plz_poly2['volume']/plz_poly2['area']

    ### Now calculate the distance from the weighted center of every postcode to the closest subcenter
    # city_plz=plz_cent.loc[(plz_cent['plz'].isin(city_plz_dict[city]))] # we don't use this anymore, we use plz_wgt_cent['wgt_center'] instead
    subcenters_coord=cp_RD['centerpoint']

    # gdf_1=city_plz['geometry'].to_crs(crs0) # this is the old approach of using postcode centroids
    gdf_1=plz_wgt_cent['wgt_center'] # here i instead use a gdf with geometry = points of weighted bld vol center for each postcode
    gdf_2=subcenters_coord.to_crs(crs0)

    # # create distance to city centre, measured in km
    d2sc=gdf_1.geometry.apply(lambda g: 0.001*gdf_2.distance(g))

    # # remove the center from the subcenters dataframe
    # if center_ix in d2sc.columns:
    #     d2sc.drop(center_ix,inplace=True,axis=1)

    # # merge the postcode into the df of distance to sub-centers by index
    d2sc=d2sc.merge(plz_wgt_cent['plz'],left_index=True,right_index=True)

    # # pre-define min distance to subcenter as 0
    d2sc['minDist_subcenter']=0
    d2sc.reset_index(drop=True,inplace=True)

    # # then remove the center from the subcenters dataframe
    #d2sc.drop(center_ix,inplace=True,axis=1)
    # And iterate over the rows to find the min distance to subcenters
    for idx, row in d2sc.iterrows():
        d2sc.loc[idx,'minDist_subcenter']=d2sc.iloc[idx,np.arange(0,len(d2sc.columns)-2)].min()

    # keep just the postcode and min distance to subcenter
    d2sc=d2sc.loc[:,('plz','minDist_subcenter')]

    # calculate distance to the main city center, identified by the cp.index
    gdf_1=plz_wgt_cent['wgt_center'] # here i instead use a gdf with geometry = points of weighted bld vol center for each postcode
    gdf_2=result_gdf.loc[result_gdf['grid_id']==center_id,'centerpoint'].to_crs(crs0)

    # # create distance to city centre, measured in km
    d2c=gdf_1.geometry.apply(lambda g: 0.001*gdf_2.distance(g))
    d2c.columns=['Distance2Center']
    # # merge the postcode into the df of distance to center by index
    d2c=d2c.merge(plz_wgt_cent['plz'],left_index=True,right_index=True)
    d2c=d2c.loc[:,('plz','Distance2Center')].copy()

    # merge the distance to center and distance to subcenter by postcode
    d2=d2sc.merge(d2c)
    # merge the building volume density into the distance to center dataframe for saving
    d2=d2.merge(plz_poly2.loc[:,('plz','build_vol_density')])

    # save distance to (sub)centers, and built up volume per postcode as csv
    d2['plz']=d2['plz'].astype('str')
    fp = '../outputs/CenterSubcenter/'+city+'_dist.csv'
    d2.to_csv(fp,index=False)

    # convert to gdf by combining with the postcode polygons
    d2_poly=gpd.GeoDataFrame(plz_poly2.merge(d2),crs=crs0)

    # save the coordinates of the center and subcetenters
    cp_RD['type']='subcenter'
    cp_RD.loc[cp_RD['grid_id']==center_id,'type']='center'
    fp = '../outputs/CenterSubcenter/'+city+'_coords.csv'
    cp_RD.to_crs(4326).to_csv(fp,index=False)

    # # avoid identifying the center as a subcenter when plotting; drop it from the relevant gdfs
    # if center_ix in cp_RD.index:
    #     cp_RD.drop(center_ix,axis=0,inplace=True)
    # if center_ix in cp_R.index:
    #     cp_R.drop(center_ix,axis=0,inplace=True)
    # if center_ix in cp_D.index:
    #     cp_D.drop(center_ix,axis=0,inplace=True)

    # make some plots to save

    fig, ax = plt.subplots(figsize=(11,11))
    gdf_boundary.plot(ax=ax,facecolor='w',edgecolor='b')
    ax.add_artist(ScaleBar(1,font_properties={'size':20}))
    centers_regional_dist.plot(ax=ax, 
        edgecolor='black',
        facecolor='purple',
        alpha = 0.7) 

    plz_poly2.plot(ax=ax,facecolor='None',edgecolor='black',alpha=0.2)
    cp_RD.plot(ax=ax,facecolor='black') 
    cp0.plot(ax=ax,facecolor='red')
                        
    plt.title("Grid cells with a threhold of " + str(CO) + "SD above mean, using regional AND distance-based thresholds \n Black dots identify the highest density cells of each contiguous group. City: " + city)
    plt.savefig('../figures/CenterSubcenter/'+ city + '_centercells.png',facecolor='w')

    fig, ax = plt.subplots(figsize=(11,11))
    gdf_boundary.plot(ax=ax,facecolor='w',edgecolor='b')
    ax.add_artist(ScaleBar(1,font_properties={'size':20}))

    d2_poly.plot(ax=ax, column='Distance2Center',legend=True) 
    cp_RD.plot(ax=ax,facecolor='black') 
    cp0.plot(ax=ax,facecolor='red')
                        
    plt.title("Center and Subcenters and distance to center for each postcode, City: " + city)
    plt.savefig('../figures/CenterSubcenter/'+ city + '_dist2center.png',facecolor='w')

    fig, ax = plt.subplots(figsize=(11,11))
    gdf_boundary.plot(ax=ax,facecolor='w',edgecolor='b')
    ax.add_artist(ScaleBar(1,font_properties={'size':20}))

    d2_poly.plot(ax=ax, column='minDist_subcenter',legend=True) 
    cp_RD.plot(ax=ax,facecolor='black') 
    cp0.plot(ax=ax,facecolor='red')
                        
    plt.title("Center and Subcenters and distance to closest subcenter for each postcode, City: " + city)
    plt.savefig('../figures/CenterSubcenter/'+ city + '_dist2subcenter.png',facecolor='w')

    fig, ax = plt.subplots(figsize=(10,10))
    gdf_boundary.plot(ax=ax,facecolor='w',edgecolor='b')
    ax.add_artist(ScaleBar(1,font_properties={'size':20}))

    buildings_gdf.plot(ax=ax, 
        edgecolor='black',
        alpha = 0.7)
    cp_RD.plot(ax=ax,facecolor='cyan') 
    cp0.plot(ax=ax,facecolor='red')   
    plt.title("City boundary, buildings, center, and region+distance-based subcenters: " + city)   
    plt.savefig('../figures/CenterSubcenter/'+ city + '_bldgs_subcenters.png',facecolor='w')

    ## Plot and save sum building volumes by grid cell
    fig, ax = plt.subplots(figsize=(11,11))
    gdf_boundary.plot(ax=ax,facecolor='w',edgecolor='black')
    ax.add_artist(ScaleBar(1,font_properties={'size':20}))
    result_gdf.plot(ax=ax, column='Sum Bldg Volume (m3)', cmap='Spectral_r', legend=True)
    cp_RD.plot(ax=ax,facecolor='black') 
    cp0.plot(ax=ax,facecolor='red')
    d2_poly.plot(ax=ax,alpha=0.25,edgecolor='black',facecolor='None') 
    plt.title('Building volume by grid cell with postcodes, centers and subcenters, ' + city)  
    plt.savefig('../figures/CenterSubcenter/'+ city + '_vol_gridcell.png',facecolor='w')

    print('Finished calculating center and subcenters for ' + city)

#cities=pd.Series(['Berlin','Dresden','Leipzig'])
cities=pd.Series(['Berlin','Dresden','Leipzig','Magdeburg','Potsdam'])
#cities=pd.Series(['Magdeburg','Potsdam'])
cities.apply(subcenters)