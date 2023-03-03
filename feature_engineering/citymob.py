import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid


def remove_holes(gdf, T,loc):
    gdf_copy=gdf.copy()
    gdf_copy['valid']=gdf_copy.is_valid

    # pc=round(100*sum(gdf_copy.is_valid==False)/len(gdf_copy))
    # print(str(pc) + 'percent of geometries are invalid in ' + loc)

    # turn Geometry Collections into Polygons or Multipolygons (remove lines and points). needs to be expanded to include geometry collections including multipolygons
    for i, row in gdf_copy.iterrows():
        if row.geometry.geom_type == 'GeometryCollection':
            print(i, 'we got a GC')

            # get all polygons
            shapes = []

            for shape in row.geometry.geoms:
                if shape.geom_type == 'Polygon':
                    print('got a poly')
                    shapes.append(shape)
                elif shape.geom_type == 'MultiPolygon':
                    print('got an mp in a gc')
                    #list_parts=[]
                    for poly in shape.geoms:
                        if poly.geom_type == 'Polygon':
                            print('got a poly in a mp in a gc')
                            shapes.append(poly)
            if len(shapes)>1:
                print('its a MP now')
                gdf_copy.at[i, 'geometry'] = MultiPolygon(shapes)
            else:
                print('its a poly now')
                gdf_copy.at[i, 'geometry'] = shapes[0]

    gdf_copy['geo_old']=gdf_copy['geometry']
    list_geos=[]
    for geo in gdf_copy['geo_old']:
        if geo.geom_type == 'Polygon': # if polygon
            if len(geo.interiors)==0: # if no interiors, add the polygon directly without changes
                list_geos.append(geo)
            else:
                if max([Polygon(a).area for a in geo.interiors])<T: # if max interior area is small, remove all interiors
                    p = Polygon(geo.exterior.coords)
                    list_geos.append(p)
                else: # if max interior area is large, check which interiors are bigger than T, and keep those
                    list_interiors=[]
                    for interior in geo.interiors:
                        pi = Polygon(interior)
                        if pi.area>T:
                            list_interiors.append(interior)
                    new_poly=Polygon(geo.exterior.coords, holes=list_interiors)
                    list_geos.append(new_poly)


        if geo.geom_type == 'MultiPolygon': # if multipolygon
            list_parts=[]
            for polygon in geo.geoms:

                # filter out polygons with area below threshold
                if polygon.area>T:
                    
                    # if interiors exist remove interiors with area below threshold
                    if len(polygon.interiors)>0: # if interiors exist
                        if max([Polygon(a).area for a in polygon.interiors])<T: # if max interior area is small, remove all interiors
                            p = Polygon(polygon.exterior.coords)
                            list_parts.append(p)

                        else: # if max interior area is large, check which interiors are bigger than T, and keep those
                            list_interiors=[]
                            for interior in polygon.interiors:
                                pi = Polygon(interior)
                                if pi.area>T:
                                    list_interiors.append(interior)
                            new_poly=Polygon(polygon.exterior.coords, holes=list_interiors)
                            list_parts.append(new_poly)

                    else: # if no interiors exist, add polygon as is 
                        list_parts.append(polygon)

            if len(list_parts)>1:
                new_geom=MultiPolygon(list_parts)
            else: 
                new_geom=list_parts[0]

            list_geos.append(new_geom)
        if geo.geom_type == 'Point': # if Point, do nothing, just add it into list_geos without changes
            list_geos.append(geo)
            
    gdf_copy['geometry']=list_geos
    gdf_copy.drop(columns=['geo_old'],inplace=True)
    # pc=round(100*sum(gdf_copy.is_valid==False)/len(gdf_copy))
    # print(str(pc) + 'percent of geometries are invalid after removing holes in ' + loc)
    gdf_copy.drop(columns=['valid'],inplace=True)

    return(gdf_copy)


def remove_invalid_geoms(gdf,crs0,gdf_name, city):
    '''
    Identify invalid geometries, and replace them with valid geometries using the Shapely `make_valid` method.
    '''
    gdf['valid']=gdf.is_valid
    if any(gdf.is_valid==False):
        pc=round(100*sum(gdf.is_valid==False)/len(gdf))
        print(str(pc) + ' percent of geometries are invalid in ' + gdf_name + ' for ' + city)
    else:
         print('0 percent of geometries are invalid in ' + gdf_name + ' for ' + city)
    if any(gdf.is_valid==False):

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
            print('Unable to make valid all geometries in ' + city)
    
    return(gdf)