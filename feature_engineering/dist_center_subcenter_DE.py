# script to calculate location of subcenters, and distances from postcodes to center and subcenters
# last update Peter Berrill June 6 2023

# load required libraries
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
from citymob import import_csv_w_wkt_to_gdf

crs0=3035

# load in the 1km grid from Inspire for Germany, and use this grid to estimate the subcenters.
grid=gpd.read_file('../../../general_GIS_files/Germany/Inspire_1km_grid/DE_Gitter_ETRS89_LAEA_1km.shp') 
# ensure grid is in the common crs
grid=grid.to_crs(crs0)
# calculate grid cell areas
grid['area_cell']=grid['geometry'].area
grid['grid_id'] = grid.index

# read in German postcotde
fp = "../shapefiles/plz-5stellig.shp/plz-5stellig.shp"
plz = gpd.read_file(fp)
plz=plz.to_crs(crs0)

plz_poly=plz.copy()
plz_cent=plz.copy()
plz_cent['geometry']=plz_cent.centroid
del plz

# load the dictionary of city postcodes, and select only postcodes within the selected city
city_plz_fp='../dictionaries/city_postcode_DE.pkl'
a_file = open(city_plz_fp, "rb")
city_plz_dict = pickle.load(a_file)

# read in shapefile of user-defined city centers
centers=import_csv_w_wkt_to_gdf('../source/citycenters/centers.csv',crs=4326)
centers.to_crs(crs0,inplace=True)
def subcenters(city):
    print(city)
    city_poly=plz_poly.loc[(plz_poly['plz'].isin(city_plz_dict[city])),:].copy()
    if city == 'Frankfurt am Main':
        fp='../../MSCA_data/BuildingsDatabase/clips/eubucco_frankfurt_am_main.shp' 
    else:
        fp='../../MSCA_data/BuildingsDatabase/clips/eubucco_' + city + '.shp' 

    buildings_gdf=gpd.read_file(fp)

    # read in file of city boundaries
    fp='../outputs/city_boundaries/' + city + '.csv'  
    gdf_boundary = import_csv_w_wkt_to_gdf(fp,crs=crs0,geometry_col='geometry')

    # calculate the area of the buidlings from the database
    buildings_gdf["area"] = buildings_gdf["geometry"].area
    # calculate the centerpoint of each building geometry, 
    buildings_gdf["center"] = buildings_gdf["geometry"].centroid

    # merge the building attributes and heights
    gdf=buildings_gdf.copy()
    # calculate building volumes
    gdf['volume']=gdf['area']*gdf['height']

    #### Now we move onto the next step, sum up values by grid cell and then apply thresholds to identify sub-centers ###########
    # spatial join the geodataframe containing building geometries and locations with the grid
    jn=gpd.sjoin(gdf,grid,how="left",predicate="intersects")
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

    # define cut-off
    if city in (['Frankfurt am Main','Kassel']):
        # Higher threshold for area-based subcenters
        CO=1.25
        # First calculate distance based sub-centers, i.e. points of high concentration of built-up area
        # That means, within each distance group, identify which cells are above the cutoff (defined as group mean + 1SD) of bldg footprint per km2
        result_gdf['group_mean'] = grouped_gdf['Sum Bldg Footprint (m2)'].transform('mean')
        result_gdf['group_SD'] = grouped_gdf['Sum Bldg Footprint (m2)'].transform('std')
        result_gdf['cutoff'] = result_gdf['group_mean'] + CO*result_gdf['group_SD'] 
        result_gdf['isCenter_dist'] = result_gdf['Sum Bldg Footprint (m2)'] > result_gdf['cutoff']

        # Second calulate a city-wide threshold for identifying subcenter cells.
        # This is calculated as the mean + CO*SD of the built up volume of all cells throughout the city
        cut_regional=CO*sum_fp.std()+sum_fp.mean()
        result_gdf['isCenter_regional'] = result_gdf['Sum Bldg Footprint (m2)'] > cut_regional
        # finally, identify which cells are identified as centers by both the regional and distance based approaches.
        result_gdf['isCenter_regional_dist'] = result_gdf['isCenter_regional'] & result_gdf['isCenter_dist']
    else:
        CO=1
        # First calculate distance based sub-centers, i.e. points of high concentration of built-up volumne
        # That means, within each distance group, identify which cells are above the cutoff (defined as group mean + 1SD) of built-up volume per km2
        result_gdf['group_mean'] = grouped_gdf['Sum Bldg Volume (m3)'].transform('mean')
        result_gdf['group_SD'] = grouped_gdf['Sum Bldg Volume (m3)'].transform('std')
        result_gdf['cutoff'] = result_gdf['group_mean'] + CO*result_gdf['group_SD'] 
        result_gdf['isCenter_dist'] = result_gdf['Sum Bldg Volume (m3)'] > result_gdf['cutoff']

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

    # classify each center cell into the group it is a member of, based on it's proximity a la Queen contiguity
    # This throws a 'value is trying to be set on a copy of a slice from a DataFrame' error which I cannot avoid.
    centers_regional_dist['contig_group']=wqrd.component_labels

    # retrieve the index id of the cell with the greatest volume in each contiguous group - this will be the 'center' cell of the city subcenter
    if city in (['Frankfurt am Main','Kassel']):
        max_ids_RD=centers_regional_dist.groupby('contig_group')['Sum Bldg Footprint (m2)'].transform('idxmax').unique()
    else:
        max_ids_RD=centers_regional_dist.groupby('contig_group')['Sum Bldg Volume (m3)'].transform('idxmax').unique()

    # Create GeoDataFrames of the centerpoints of each city subcenter calculated with the three different methods
    cp_RD=centers_regional_dist.loc[max_ids_RD].set_geometry('centerpoint')

    # now calculate the centre of mass (or volume or area) of each postcode, as the weighted mean xy coordinates, with built-up volume/area used as the weights.
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

    if city in (['Frankfurt am Main','Kassel']):
        # sum area by grid cell
        sum_fp_g2=pd.DataFrame(jn2.groupby('id_grid')['area'].sum())
        sum_fp_g2.reset_index(inplace=True)
        grid_100=grid_100.merge(sum_fp_g2)
    else:
        # sum volumes by grid cell
        sum_vol_g2=pd.DataFrame(jn2.groupby('id_grid')['volume'].sum())
        sum_vol_g2.reset_index(inplace=True)
        grid_100=grid_100.merge(sum_vol_g2)

    grid_100['centroid']=grid_100.centroid
    grid_100.rename(columns={'geometry':'cell_geom'},inplace=True)
    grid_100.set_geometry('centroid',inplace=True,crs=crs0)

    # spatial join the geodataframe containing building volumes/areas by 100m grid cells with the polygons of postcodes
    jn3=gpd.sjoin(grid_100,city_poly,how="left",predicate="intersects")
    if city in (['Frankfurt am Main','Kassel']):
        jn3=jn3.loc[:,('id_grid','cell_geom','centroid','area','plz','note')].copy()
        if city=='Frankfurt am Main': # manually allocate grid id 32399 to plz 60308 in Frankfurt. This is because no grid cells gets allocated to 60308, because it is so small. But 32399 is closest, and we need to keep 60308 in 
            for c in ['plz','note','einwohner','qkm']:
                jn3.loc[jn3['id_grid']==32399,c]=city_poly.loc[city_poly['plz']=='60308',c].iloc[0]
    else:
        jn3=jn3.loc[:,('id_grid','cell_geom','centroid','volume','plz','note')].copy()

    # calculating weighted center
    jn3['x_center']=jn3.centroid.map(lambda p: p.x)
    jn3['y_center']=jn3.centroid.map(lambda p: p.y)

    if city in (['Frankfurt am Main','Kassel']):
        jn3['x_wgt']=jn3['x_center']*jn3['area']
        jn3['y_wgt']=jn3['y_center']*jn3['area']

        plz_vol=pd.DataFrame(jn3.groupby('plz')['area'].sum())# use area for Hessen cities
        plz_vol.reset_index(drop=False,inplace=True)
        plz_vol.rename(columns={'area':'footprint'},inplace=True)
        
        # create gdf of weighted centers
        data=[]
        for pc in city_poly['plz']:
                x_mean=jn3.loc[jn3['plz']==pc,'x_wgt'].sum()/jn3.loc[jn3['plz']==pc,'area'].sum()
                y_mean=jn3.loc[jn3['plz']==pc,'y_wgt'].sum()/jn3.loc[jn3['plz']==pc,'area'].sum()
                p=Point([x_mean,y_mean])
                data.append([pc,p])
    else:
        jn3['x_wgt']=jn3['x_center']*jn3['volume']
        jn3['y_wgt']=jn3['y_center']*jn3['volume']

        plz_vol=pd.DataFrame(jn3.groupby('plz')['volume'].sum())#
        plz_vol.reset_index(drop=False,inplace=True)

        # create gdf of weighted centers
        data=[]
        for pc in city_poly['plz']:
                x_mean=jn3.loc[jn3['plz']==pc,'x_wgt'].sum()/jn3.loc[jn3['plz']==pc,'volume'].sum()
                y_mean=jn3.loc[jn3['plz']==pc,'y_wgt'].sum()/jn3.loc[jn3['plz']==pc,'volume'].sum()
                p=Point([x_mean,y_mean])
                
                data.append([pc,p])

    # create dataframe containing the weighted centers, volume, and then calculate volume density for each postcode
    plz_wgt_cent=gpd.GeoDataFrame(data,crs=crs0,columns =['plz','geometry'])
    plz_wgt_cent.rename(columns={'geometry':'wgt_center'},inplace=True)
    plz_poly2=plz_poly.merge(plz_wgt_cent).copy() 
    plz_poly2=plz_poly2.merge(plz_vol)
    plz_poly2['area']=plz_poly2.area*1e-6 # area of postcode in km2

    # calculate building volume density (m3/km2) for each postcode
    if city in (['Frankfurt am Main','Kassel']):
        plz_poly2['build_vol_density']=plz_poly2['footprint']/plz_poly2['area']*7 # estimate of average height, to make consistent with other cities estimate of building vol density, check with Nikola
    else:
        plz_poly2['build_vol_density']=plz_poly2['volume']/plz_poly2['area']

    ### Now calculate the distance from the weighted center of every postcode to the closest subcenter
    subcenters_coord=cp_RD['centerpoint']

    gdf_1=plz_wgt_cent['wgt_center'] # points of weighted bld vol center for each postcode
    gdf_2=subcenters_coord.to_crs(crs0)

    # create distance to city centre, measured in km
    d2sc=gdf_1.geometry.apply(lambda g: 0.001*gdf_2.distance(g))

    # # merge the postcode into the df of distance to sub-centers by index
    d2sc=d2sc.merge(plz_wgt_cent['plz'],left_index=True,right_index=True)

    # pre-define min distance to subcenter as 0
    d2sc['minDist_subcenter']=0
    d2sc.reset_index(drop=True,inplace=True)

    # iterate over the rows to find the min distance to subcenters
    for idx, row in d2sc.iterrows():
        d2sc.loc[idx,'minDist_subcenter']=d2sc.iloc[idx,np.arange(0,len(d2sc.columns)-2)].min()
    # keep just the postcode and min distance to subcenter
    d2sc=d2sc.loc[:,('plz','minDist_subcenter')]

    # calculate distance to the main city center, identified by the cp.index
    gdf_2=result_gdf.loc[result_gdf['grid_id']==center_id,'centerpoint'].to_crs(crs0)

    # # create distance to city centre, measured in km
    d2c=gdf_1.geometry.apply(lambda g: 0.001*gdf_2.distance(g))
    d2c.columns=['Distance2Center']
    # # merge the postcode into the df of distance to center by index
    d2c=d2c.merge(plz_wgt_cent,left_index=True,right_index=True)
    d2c=d2c.loc[:,('plz','Distance2Center','wgt_center')].copy()

    # merge the distance to center and distance to subcenter by postcode
    d2=d2sc.merge(d2c)
    # merge the building volume density into the distance to center dataframe for saving
    d2=d2.merge(plz_poly2.loc[:,('plz','build_vol_density')])

    # save distance to (sub)centers, and built up volume per postcode as csv
    d2.rename(columns={'plz':'geocode'},inplace=True)
    d2['geocode']=d2['geocode'].astype(str)
    fp = '../outputs/CenterSubcenter/'+city+'_dist.csv'
    d2.to_csv(fp,index=False)
    # convert to gdf by combining with the postcode polygons
    d2_poly=gpd.GeoDataFrame(plz_poly2.merge(d2),crs=crs0)

    # save the coordinates of the center and subcetenters
    cp_RD['type']='subcenter'
    fp = '../outputs/CenterSubcenter/'+city+'_coords.csv'
    cp_RD.to_crs(4326).to_csv(fp,index=False)
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
    plt.savefig('../outputs/CenterSubcenter/'+ city + '_centercells.png',facecolor='w')

    fig, ax = plt.subplots(figsize=(10,10))
    gdf_boundary.plot(ax=ax,facecolor='w',edgecolor='b')
    ax.add_artist(ScaleBar(1,font_properties={'size':20}))

    buildings_gdf.plot(ax=ax, 
        edgecolor='black',
        alpha = 0.7)
    cp_RD.plot(ax=ax,facecolor='cyan') 
    cp0.plot(ax=ax,facecolor='red')   
    plt.title("City boundary, buildings, center, and region+distance-based subcenters: " + city)   
    plt.savefig('../outputs/CenterSubcenter/'+ city + '_bldgs_subcenters.png',facecolor='w')

    ## Plot and save sum building volumes by grid cell
    fig, ax = plt.subplots(figsize=(11,11))
    gdf_boundary.plot(ax=ax,facecolor='w',edgecolor='black')
    ax.add_artist(ScaleBar(1,font_properties={'size':20}))
    result_gdf.plot(ax=ax, column='Sum Bldg Volume (m3)', cmap='Spectral_r', legend=True)
    cp_RD.plot(ax=ax,facecolor='black') 
    cp0.plot(ax=ax,facecolor='red')
    d2_poly.plot(ax=ax,alpha=0.25,edgecolor='black',facecolor='None') 
    plt.title('Building volume by grid cell with postcodes, centers and subcenters, ' + city)  
    plt.savefig('../outputs/CenterSubcenter/'+ city + '_vol_gridcell.png',facecolor='w')

    print('Finished calculating center and subcenters for ' + city)

cities=pd.Series(['Berlin','Dresden','Düsseldorf','Frankfurt am Main','Kassel','Leipzig','Magdeburg','Potsdam'])
cities.apply(subcenters)