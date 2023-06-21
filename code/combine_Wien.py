# script to combine and harmonize survey data for Wien, and calculate some summary statistics
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

city = 'Wien'

# inputs:
# Wien dictionary pickle files, to create harmonised variables from the raw survey data
# SrV survey data files (houeshold, population, trips) for Wien 
# population density by spatial unit for Wien
# other pre-calculated urban form features (building density, distance to center, land-use data, connectivity metrics) by spatial unit 

fn_hh='../../MSCA_data/Austria_OnTheRoad/ou_data/ou_2013-2014/Data/Haushaltsdatensatz.csv'
fn_p='../../MSCA_data/Austria_OnTheRoad/ou_data/ou_2013-2014/Data/Personendatensatz.csv'
fn_t='../../MSCA_data/Austria_OnTheRoad/ou_data/ou_2013-2014/Data/Wegedatensatz.csv'

sH=pd.read_csv(fn_hh,sep=';',encoding='latin-1')
sP=pd.read_csv(fn_p,sep=';',encoding='latin-1')
sW=pd.read_csv(fn_t,sep=';',encoding='latin-1')

# load pickled dictionary files
with open('../dictionaries/Austria_var.pkl','rb') as f:
    var_all = pickle.load(f)

with open('../dictionaries/Austria_val.pkl','rb') as f:
    value_all = pickle.load(f)

with open('../dictionaries/Austria_na.pkl','rb') as f:
    na_all = pickle.load(f)


for k in var_all['HH'].keys():
    if len(value_all['HH'][k])>0:
        sH[k]=sH[var_all['HH'][k]].map(value_all['HH'][k])
    elif len(value_all['HH'][k])==0:
        sH[k]=sH[var_all['HH'][k]]

for k in var_all['P'].keys():
    if len(value_all['P'][k])>0:
        sP[k]=sP[var_all['P'][k]].map(value_all['P'][k])
    elif len(value_all['P'][k])==0:
        sP[k]=sP[var_all['P'][k]]

for k in var_all['W'].keys():
    if len(value_all['W'][k])>0:
        sW[k]=sW[var_all['W'][k]].map(value_all['W'][k])
    elif len(value_all['W'][k])==0:
        sW[k]=sW[var_all['W'][k]]

# fill in na's as necessary
sH.fillna(value=na_all['HH'],inplace=True)
sP.fillna(value=na_all['P'],inplace=True)
sW.fillna(value=na_all['W'],inplace=True)

# keep only variables needed, i.e. the variables that are included as dictionary keys
sH=sH[list(value_all['HH'].keys())]
sP=sP[list(value_all['P'].keys())]
sW=sW[list(value_all['W'].keys())]

# define unique person and trip numbers
sP['HH_PNR']=sP['HHNR'].astype('str')+'_'+sP['Person'].astype('str')
sW['HH_PNR']=sW['HHNR'].astype('str')+'_'+sW['Person'].astype('str')
sW['HH_P_WNR']=sW['HH_PNR']+'_'+sW['Trip'].astype('str')

# bring HHNR and geo_unit to the left side of the HH df
cols=sH.columns.tolist()
cols_new = ['HHNR'] + [value for value in cols if value not in {'HHNR'}]
sH=sH[cols_new]

# bring HHNR, HH_PNR, to the left side of the Per df
cols=sP.columns.tolist()
cols_new = ['HHNR','HH_PNR'] + [value for value in cols if value not in {'HHNR', 'HH_PNR'}]
sP=sP[cols_new]

# bring HHNR, HH_PNR, HH_P_WNR to the left side of the W df
cols=sW.columns.tolist()
cols_new = ['HHNR','HH_PNR','HH_P_WNR'] + [value for value in cols if value not in {'HHNR', 'HH_PNR','HH_P_WNR'}]
sW=sW[cols_new]

# define trip time of day
# first remove observations with no timestamp
sW=sW.loc[sW['Time']!=' ',]
sW['Hour']=[int(elem.split(':')[0]) for elem in sW.Time]
# break the trip departure times into grouped times of day
sW['Trip_Time']='Nighttime Off-Peak'
sW.loc[sW['Hour'].isin([6,7,8,9]),'Trip_Time']='AM_Rush'
sW.loc[sW['Hour'].isin([12,13]),'Trip_Time']='Lunch'
sW.loc[sW['Hour'].isin([16,17,18]),'Trip_Time']='PM Rush'
sW.loc[sW['Hour'].isin([19,20,21]),'Trip_Time']='Evening'
sW.loc[sW['Hour'].isin([10,11,14,15]),'Trip_Time']='Daytime Off-Peak'
sW['Trip_Time'].value_counts()

# trip type/purpose 
sW['trip_type_all']=sW['Ori_Reason']+'-'+sW['Des_Reason']

sW['Trip_Purpose']='Other'
sW.loc[(sW['Ori_Reason'].isin(['Home','Personal'])) & (sW['Des_Reason'].isin(['Home','Personal'])),'Trip_Purpose']='Home↔Personal' #
sW.loc[(sW['Ori_Reason'].isin(['Home','Companion'])) & (sW['Des_Reason'].isin(['Home','Companion'])),'Trip_Purpose']='Home↔Companion' #
sW.loc[(sW['Ori_Reason'].isin(['Work','Companion'])) & (sW['Des_Reason'].isin(['Work','Companion'])),'Trip_Purpose']='Work↔Companion' #
sW.loc[(sW['Ori_Reason'].isin(['Home','Other'])) & (sW['Des_Reason'].isin(['Home','Other'])),'Trip_Purpose']='Other↔Home'
sW.loc[sW['trip_type_all']=='Home-Shopping','Trip_Purpose']='Home-Shopping'
sW.loc[sW['trip_type_all']=='Shopping-Home','Trip_Purpose']='Shopping-Home'
sW.loc[sW['trip_type_all']=='Home-School','Trip_Purpose']='Home-School'
sW.loc[sW['trip_type_all']=='School-Home','Trip_Purpose']='School-Home'
sW.loc[sW['trip_type_all']=='Home-Work','Trip_Purpose']='Home-Work'
sW.loc[sW['trip_type_all']=='Work-Home','Trip_Purpose']='Work-Home'
sW.loc[sW['trip_type_all']=='Home-Leisure','Trip_Purpose']='Home-Leisure'
sW.loc[sW['trip_type_all']=='Leisure-Home','Trip_Purpose']='Leisure-Home'
sW.loc[sW['trip_type_all']=='Shopping-Shopping','Trip_Purpose']='Shopping'
sW.loc[sW['trip_type_all']=='Work-Work','Trip_Purpose']='Work'
sW.loc[sW['trip_type_all']=='Work-Leisure','Trip_Purpose']='Work-Leisure'
sW.loc[sW['trip_type_all']=='Work-Work','Trip_Purpose']='Work'
sW.loc[sW['trip_type_all']=='Work-Leisure','Trip_Purpose']='Work-Leisure'
sW.loc[sW['trip_type_all']=='Work-Shopping','Trip_Purpose']='Work-Shopping'
sW.loc[sW['trip_type_all']=='Leisure-Work','Trip_Purpose']='Leisure-Work'
sW.loc[sW['trip_type_all']=='Leisure-Leisure','Trip_Purpose']='Leisure'
sW.loc[sW['trip_type_all']=='Companion-Companion','Trip_Purpose']='Companion'

# make the aggregated trip purpose
sW['Trip_Purpose_Agg']='Other'
sW.loc[sW['Trip_Purpose'].isin(['Home-Work','Work-Home']),'Trip_Purpose_Agg']='Home↔Work'
sW.loc[sW['Trip_Purpose'].isin(['Home-School','School-Home']),'Trip_Purpose_Agg']='Home↔School'
sW.loc[sW['Trip_Purpose'].isin(['Home-Shopping','Shopping-Home']),'Trip_Purpose_Agg']='Home↔Shopping'
sW.loc[sW['Trip_Purpose'].isin(['Home↔Companion']),'Trip_Purpose_Agg']='Home↔Companion'
sW.loc[sW['Trip_Purpose'].isin(['Home-Leisure','Leisure-Home','Home↔Personal']),'Trip_Purpose_Agg']='Home↔Leisure'

# split students into school and 3rd level by age, with a threshold of 18
sP.loc[(sP['Age']<19) & (sP['Occupation']=='Student'),'Occupation']='Student_School'
sP.loc[(sP['Age']>18) & (sP['Occupation']=='Student'),'Occupation']='Student_3rdLevel'

# address the NA values for Education and Occupation for children under 6
sP.loc[sP['Age']<7,['Education','Occupation']]='Pre-School'
# address inconsistencies with the Person 'Education' variable, arising from respondents who have completed a certain level of education/training responding no dimploma yet, even though they have lower diplomas than the one they are currently studying for
# these assumptions are based on Spanish law, in which it is mandatory to go to school until age 15 (end of secondary school), so anyone in an occupation post-15 is very likely to have some education. https://www.expatica.com/at/education/children-education/education-in-austria-87120/

sP.loc[(sP['Age']>11) & (sP['Age']<16) & (sP['Education'].isin(['No diploma yet','Unkown'])),'Education']="Elementary" # if aged between 12 and 15, assume at least an Elementary education
sP.loc[(sP['Age']>15) & (sP['Age']<20) & (sP['Education'].isin(['No diploma yet','Unkown'])),'Education']="Secondary" # if aged between 16 and 19, assume at least a Secondary education.


# drop unneccessary columns
#sW.drop(columns=['Ori_Reason','Des_Reason','HHNR','Person','Time','Hour'],inplace=True)
sW.drop(columns=['HHNR','Person','Time','Hour'],inplace=True)
sH.drop(columns=['BikeAvailable','2_3WAvailable','CarAvailable'],inplace=True)


# convert comma decimal points to points
sW['Trip_Distance']=sW['Trip_Distance'].map(lambda x: round(float(x.replace(',','.')),2))
sW['Trip_Duration']=sW['Trip_Duration'].map(lambda x: round(float(x.replace(',','.')),2))
sW['Trip_Weight']=sW['Trip_Weight'].map(lambda x: float(x.replace(',','.')))

sH['HH_Weight']=sH['HH_Weight'].map(lambda x: float(x.replace(',','.')))
sP['Per_Weight']=sP['Per_Weight'].map(lambda x: float(x.replace(',','.')))

# merge together the household, person, and trip files
sHP=sH.merge(sP,on='HHNR')

sHPW=sHP.merge(sW,on='HH_PNR')

# make a generic variable for parking available at work/study destinations
sHPW['ParkingAvailable_Dest']=0
sHPW.loc[(sHPW['Des_Reason'].isin(['Work','School'])) & (sHPW['Work/Study_CarParkAvailable']==1) ,'ParkingAvailable_Dest']=1
sHPW.drop(columns=['Ori_Reason','Des_Reason','Work/Study_CarParkAvailable'],inplace=True)

# bring HHNR, HH_PNR, HH_P_WNR to the left side of the combined df
cols=sHPW.columns.tolist()
cols_new = ['HHNR','HH_PNR','HH_P_WNR','Res_geocode','Ori_geocode','Des_geocode','Trip_Time','Trip_Purpose'] + [value for value in cols if value not in {'HHNR','HH_PNR','HH_P_WNR','Res_geocode','Ori_geocode','Des_geocode','Trip_Time','Trip_Purpose','Mode', 'Trip_Distance'}] +['Mode', 'Trip_Distance']
sHPW=sHPW[cols_new]

# create a dictionary to map the old municipality ids to the new ones which took effect after the dissolution of the Bezirk Wien-Umgebung https://de.wikipedia.org/wiki/Bezirk_Wien-Umgebung
# this is not a comprehensive list, only those caught by the buffer around Wien are included
dict={'32403':'31949', # Gablits
'32404':'31235', # Gerasdorf bei Wien
'32406':'30732', # Himberg
'32408':'32144', # Klosterneuburg
'32409':'30734', # lanzendorf
'32410':'30735', # Leopoldsdorf
'32411':'30736',  # Maria-Lanzendorf	
'32412':'31950', # Mauerbach
'32416':'31952', # purkersdorf
'32419':'30740', # Schwechat
'32424':'30741' # Zwölfaxing
}

sHPW['Ori_geocode']=sHPW['Ori_geocode'].astype(str).replace(dict)
sHPW['Des_geocode']=sHPW['Des_geocode'].astype(str).replace(dict)
sHPW['Res_geocode']=sHPW['Res_geocode'].astype(str).replace(dict)

# load in dictionary of  geocodes, which we will use for merging with the urban form features
with open('../dictionaries/city_postcode_AT.pkl','rb') as f:
    code_dict = pickle.load(f)
# limit to trips starting within our definition of Wien
sHPW=sHPW.loc[sHPW['Ori_geocode'].astype('str').isin(code_dict['Wien']),:]

# further cleaning to retrict the trips to those between 0.1km and 50km and remove trips w/o mode information
sHPW['Trip_Distance']=sHPW.loc[:,'Trip_Distance']*1000
sHPW=sHPW.loc[(sHPW['Trip_Distance']>=50) & (sHPW['Trip_Distance']<=50000),:] 
sHPW=sHPW.loc[sHPW['Mode']!='Other',:]

# restrict to weekdays
sHPW=sHPW.loc[sHPW['Day']<6,:]
# identify people who still have 2 reporting days, and drop their second reporting day
per_days=pd.DataFrame(sHPW.loc[:,('HH_PNR','Day')].drop_duplicates()['HH_PNR'].value_counts()).reset_index()
per_days=per_days.loc[per_days['HH_PNR']>1,:]
per_days.rename(columns={'index':'HH_PNR','HH_PNR':'count'},inplace=True)
double=per_days['HH_PNR'].values
sHPW=sHPW.loc[~((sHPW['HH_PNR'].isin(double)) & (sHPW['ReportingDay']==2)) ,:]

# classify moped as 2-3wheeler
sHPW.loc[sHPW['Mode_Detailed']=='Moped','Mode']='2_3_Wheel'

weighted=sHPW.loc[:,('Trip_Weight','Mode','Trip_Distance')]
weighted['Dist_Weighted_P']=weighted['Trip_Weight']*weighted['Trip_Distance']

# weight_daily_travel=pd.DataFrame(0.001*weighted.groupby('Mode')['Dist_Weighted_P'].sum()/sP['Per_Weight'].sum()).reset_index()
# need to fix this, the distances don't make sense
# weight_daily_travel=pd.DataFrame(0.001*weighted.groupby('Mode')['Dist_Weighted_P'].sum()/(weighted['Trip_Weight'].mean()*len(sP['HH_PNR'].drop_duplicates()))).reset_index()
weight_daily_travel=pd.DataFrame(0.001*weighted.groupby('Mode')['Dist_Weighted_P'].sum()/(weighted['Trip_Weight'].mean()*len(sHPW['HH_PNR'].drop_duplicates()))).reset_index()
weight_daily_travel.rename(columns={'Dist_Weighted_P':'Daily_Travel_cap'},inplace=True)
weight_daily_travel['Mode_Share']=weight_daily_travel['Daily_Travel_cap']/weight_daily_travel['Daily_Travel_cap'].sum()

weight_daily_travel_all=pd.DataFrame(data={'Mode':['All'],'Daily_travel_all':0.001*weighted['Dist_Weighted_P'].sum()/weighted['Trip_Weight'].sum()})

carown=sHPW.loc[:,['HHNR','HH_Weight','CarOwnershipHH']].drop_duplicates()
own=pd.DataFrame(data={'Mode':['Car'],'Ownership':sum(carown['CarOwnershipHH']*carown['HH_Weight'])/sum(carown['HH_Weight'])})
weight_daily_travel=weight_daily_travel.merge(own,how='left')
weight_daily_travel=weight_daily_travel.merge(weight_daily_travel_all,how='outer')

weight_daily_travel.to_csv('../outputs/summary_stats/'+city+'_stats.csv',index=False)

# mode_share=sHPW.groupby('Mode')['Trip_Distance'].sum()/sum(sHPW.groupby('Mode')['Trip_Distance'].sum())
# weighted=sHPW.loc[:,('Trip_Weight','Mode','Trip_Distance')]
# weighted['Dist_Weighted_P']=weighted['Trip_Weight']*weighted['Trip_Distance']
# mode_share_weighted=weighted.groupby('Mode')['Dist_Weighted_P'].sum()/sum(weighted.groupby('Mode')['Dist_Weighted_P'].sum())

# print('Weighted mode share in ' + city)
# print(mode_share_weighted)

# print('N Trips')
# print(len(sHPW))

# print('Avg trip distance in km, overall: ', round(0.001*np.average(sHPW['Trip_Distance'],weights=sHPW['Trip_Weight']),1))

# person_trips=pd.DataFrame(sHPW.groupby('HH_PNR')['Trip_Distance'].sum()).reset_index()
# #print('Average travel distance per person per day, all modes, km/cap: ' , str(round(0.001*person_trips['Trip_Distance'].mean(),1)))
# print('Average travel distance per person per day, all modes, km/cap: ' , str(round(sum(0.001*weighted.groupby('Mode')['Dist_Weighted_P'].sum()/(weighted['Trip_Weight'].mean()*len(sHPW['HH_PNR'].drop_duplicates()))),1)))
# 0.001*weighted.groupby('Mode')['Dist_Weighted_P'].sum()/(weighted['Trip_Weight'].mean()*len(sHPW['HH_PNR'].drop_duplicates()))

# person_mode_trips=pd.DataFrame(sHPW.groupby(['HH_PNR','Mode'])['Trip_Distance'].sum()).reset_index()
# print('Average travel distance per person per day, all modes, km/cap: ')
# #print(str(round(0.001*person_mode_trips.groupby('Mode')['Trip_Distance'].sum()/len(person_trips),2)))
# print(str(round(0.001*weighted.groupby('Mode')['Dist_Weighted_P'].sum()/(weighted['Trip_Weight'].mean()*len(sHPW['HH_PNR'].drop_duplicates())),2)))


if len(sHPW.loc[:,('HH_PNR','Day')].drop_duplicates())!=len(sHPW.loc[:,('HH_PNR')].drop_duplicates()):
    print('NB! Some respondents report trips over more than one day')
# add trip speed and save

#sHPW['Trip_Speed']=round(60*0.001*sHPW['Trip_Distance']/(sHPW['Trip_Duration']),2) # in km/hr
sHPW.to_csv('../outputs/Combined/Wien.csv',index=False,encoding='utf-8-sig')

# load in UF stats
# population density
pop_dens=pd.read_csv('../outputs/density_geounits/' + city + '_pop_density.csv',dtype={'geocode':str})
pop_dens.drop(columns=['geometry','NAME','state','Population','area'],inplace=True)
# building density and distance to city center, here plz needs to change to geocode
bld_dens=pd.read_csv('../outputs/CenterSubcenter/' + city + '_dist.csv',dtype={'geocode':str})
bld_dens.drop(columns=['wgt_center'],inplace=True)
# connectivity stats
conn=pd.read_csv('../outputs/Connectivity/connectivity_stats_' + city + '.csv',dtype={'geocode':str})
# decide which connectivity stats we want to keep
conn=conn.loc[:,('geocode','clean_intersection_density_km','street_length_avg')]

# land-use
lu=pd.read_csv('../outputs/LU/UA_' + city + '.csv',dtype={'geocode':str})
# decide which lu varibales to use
lu=lu.loc[:,('geocode','pc_urb_fabric','pc_comm','pc_road','pc_urban')]

# now merge all urban form features with the survey data.

# population density origin
sHPW_UF=sHPW.merge(pop_dens,left_on='Ori_geocode',right_on='geocode').copy()
sHPW_UF.drop(columns='geocode',inplace=True)
sHPW_UF.rename(columns={'Density':'PopDensity_origin'},inplace=True)
# population density destination
sHPW_UF=sHPW_UF.merge(pop_dens,left_on='Res_geocode',right_on='geocode').copy() # allow for nans in destination data, see if/how model deals with them
sHPW_UF.drop(columns='geocode',inplace=True)
sHPW_UF.rename(columns={'Density':'PopDensity_res'},inplace=True)

# buidling density and distance to centers origin
sHPW_UF=sHPW_UF.merge(bld_dens,left_on='Ori_geocode',right_on='geocode').copy() 
sHPW_UF.drop(columns='geocode',inplace=True)
sHPW_UF.rename(columns={'minDist_subcenter':'DistSubcenter_origin','Distance2Center':'DistCenter_origin','build_vol_density':'BuildDensity_origin'},inplace=True)
# buidling density and distance to centers destination
sHPW_UF=sHPW_UF.merge(bld_dens,left_on='Res_geocode',right_on='geocode').copy() # allow for nans in destination data, see if/how model deals with them
sHPW_UF.drop(columns='geocode',inplace=True)
sHPW_UF.rename(columns={'minDist_subcenter':'DistSubcenter_res','Distance2Center':'DistCenter_res','build_vol_density':'BuildDensity_res'},inplace=True)

# connectivity stats, origin
sHPW_UF=sHPW_UF.merge(conn,left_on='Ori_geocode',right_on='geocode').copy() 
sHPW_UF.drop(columns='geocode',inplace=True)
sHPW_UF.rename(columns={'k_avg':'K_avg_origin','clean_intersection_density_km':'IntersecDensity_origin','street_density_km':'StreetDensity_origin',
'streets_per_node_avg':'StreetsPerNode_origin','street_length_avg':'StreetLength_origin'},inplace=True)

# connectivity stats, destination
sHPW_UF=sHPW_UF.merge(conn,left_on='Res_geocode',right_on='geocode').copy() 
sHPW_UF.drop(columns='geocode',inplace=True)
sHPW_UF.rename(columns={'k_avg':'K_avg_res','clean_intersection_density_km':'IntersecDensity_res','street_density_km':'StreetDensity_res',
'streets_per_node_avg':'StreetsPerNode_res','street_length_avg':'StreetLength_res'},inplace=True)

# land-use stats, origin
sHPW_UF=sHPW_UF.merge(lu,left_on='Ori_geocode',right_on='geocode').copy() 
sHPW_UF.drop(columns='geocode',inplace=True)
sHPW_UF.rename(columns={'pc_urb_fabric':'LU_UrbFab_origin','pc_comm':'LU_Comm_origin','pc_road':'LU_Road_origin',
'pc_urban':'LU_Urban_origin'},inplace=True)

# land-use stats, destination
sHPW_UF=sHPW_UF.merge(lu,left_on='Res_geocode',right_on='geocode').copy() 
sHPW_UF.drop(columns='geocode',inplace=True)
sHPW_UF.rename(columns={'pc_urb_fabric':'LU_UrbFab_res','pc_comm':'LU_Comm_res','pc_road':'LU_Road_res',
'pc_urban':'LU_Urban_res'},inplace=True)

# recalculate population densities based on urban fabric denominator (Changed to urban, as some demoninators were too low, even some 0 values), and building volume densities based on urban demoninator
# sHPW_UF['UrbPopDensity_origin']=sHPW_UF['PopDensity_origin']/sHPW_UF['LU_UrbFab_origin']
# sHPW_UF['UrbPopDensity_res']=sHPW_UF['PopDensity_res']/sHPW_UF['LU_UrbFab_res']
sHPW_UF['UrbPopDensity_origin']=sHPW_UF['PopDensity_origin']/sHPW_UF['LU_Urban_origin']
sHPW_UF['UrbPopDensity_res']=sHPW_UF['PopDensity_res']/sHPW_UF['LU_Urban_res']

sHPW_UF['UrbBuildDensity_origin']=sHPW_UF['BuildDensity_origin']/sHPW_UF['LU_Urban_origin']
sHPW_UF['UrbBuildDensity_res']=sHPW_UF['BuildDensity_res']/sHPW_UF['LU_Urban_res']

sHPW_UF.loc[sHPW_UF['Time2Transit']<0,'Time2Transit']=np.nan
mean_time_tran=pd.DataFrame(sHPW_UF[['Res_geocode', 'HHNR','HH_Weight','Time2Transit']].drop_duplicates().groupby(['Res_geocode'])['Time2Transit'].mean()).reset_index()
mean_time_tran.rename(columns={'Time2Transit':'MeanTime2Transit'},inplace=True)

# mean time to transit, origin
sHPW_UF=sHPW_UF.merge(mean_time_tran,left_on='Ori_geocode',right_on='Res_geocode').copy() 
sHPW_UF.drop(columns={'Res_geocode_y',},inplace=True)
sHPW_UF.rename(columns={'MeanTime2Transit':'MeanTime2Transit_origin','Res_geocode_x':'Res_geocode'},inplace=True)

# mean time to transit, destination
# sHPW_UF=sHPW_UF.merge(mean_time_tran,left_on='Des_geocode',right_on='Res_geocode').copy() 
# sHPW_UF.drop(columns={'Res_geocode_y',},inplace=True)
# sHPW_UF.rename(columns={'MeanTime2Transit':'MeanTime2Transit_dest','Res_geocode_x':'Res_geocode'},inplace=True)
sHPW_UF=sHPW_UF.merge(mean_time_tran,left_on='Res_geocode',right_on='Res_geocode').copy() 
sHPW_UF.rename(columns={'MeanTime2Transit':'MeanTime2Transit_res'},inplace=True)

sHPW_UF.to_csv('../outputs/Combined/'+city+'_UF.csv',index=False)

# create the ori_geo_unit based on the municipalities

sHPW['Ori_geo_unit']=sHPW['Ori_geocode']

sHPW['geo_unit']=sHPW['Res_geocode']

# now create and save some summary stats by postcode
sHPW['Trip_Distance_Weighted']=sHPW['Trip_Distance']*sHPW['Trip_Weight']

# mode share of trip distance by originating postcode
ms_dist_all=pd.DataFrame(sHPW.groupby(['Ori_geo_unit'])['Trip_Distance_Weighted'].sum())
ms_dist_all.reset_index(inplace=True)
ms_dist_all.rename(columns={'Trip_Distance_Weighted':'Trip_Distance_All'},inplace=True)

ms_dist=pd.DataFrame(sHPW.groupby(['Ori_geo_unit','Mode'])['Trip_Distance_Weighted'].sum())
ms_dist.reset_index(inplace=True)
ms_dist=ms_dist.merge(ms_dist_all)
ms_dist['Share_Distance']=ms_dist['Trip_Distance_Weighted']/ms_dist['Trip_Distance_All']
ms_dist.drop(columns=['Trip_Distance_Weighted','Trip_Distance_All'],inplace=True)

# convert from long to wide 
msdp=ms_dist.pivot(index='Ori_geo_unit',columns=['Mode'])
msdp.columns = ['_'.join(col) for col in msdp.columns.values]
msdp.reset_index(inplace=True)
msdp.to_csv('../outputs/Summary_geounits/' + city + '_modeshare_origin.csv',index=False)

# mode share of trip distance by residential postcode
ms_dist_all_res=pd.DataFrame(sHPW.groupby(['geo_unit'])['Trip_Distance_Weighted'].sum())
ms_dist_all_res.reset_index(inplace=True)
ms_dist_all_res.rename(columns={'Trip_Distance_Weighted':'Trip_Distance_All'},inplace=True)

ms_dist_res=pd.DataFrame(sHPW.groupby(['geo_unit','Mode'])['Trip_Distance_Weighted'].sum())
ms_dist_res.reset_index(inplace=True)
ms_dist_res=ms_dist_res.merge(ms_dist_all_res)
ms_dist_res['Share_Distance']=ms_dist_res['Trip_Distance_Weighted']/ms_dist_res['Trip_Distance_All']
ms_dist_res.drop(columns=['Trip_Distance_Weighted','Trip_Distance_All'],inplace=True)

# convert long to wide
msdrp=ms_dist_res.pivot(index='geo_unit',columns=['Mode'])
msdrp.columns = ['_'.join(col) for col in msdrp.columns.values]
msdrp.reset_index(inplace=True)

# avg travel km/day/cap by postcode of residence
# first calculate weighted number of surveyed people by postcode
plz_per_weight=pd.DataFrame(sHPW[['geo_unit', 'HH_PNR','Per_Weight']].drop_duplicates().groupby(['geo_unit'])['Per_Weight'].sum()).reset_index() 
# next calculate sum weighted travel distance by residence postcode
plz_dist_weight=pd.DataFrame(sHPW.groupby(['geo_unit'])['Trip_Distance_Weighted'].sum()).reset_index()
plz_dist_weight=plz_dist_weight.merge(plz_per_weight)
# then divide
plz_dist_weight['Daily_Distance_Person']=0.001*round(plz_dist_weight['Trip_Distance_Weighted']/plz_dist_weight['Per_Weight'])
plz_dist_weight.drop(columns=['Trip_Distance_Weighted','Per_Weight'],inplace=True)
# and car travel distance by residence postcode
plz_dist_car=pd.DataFrame(sHPW.loc[sHPW['Mode']=='Car',:].groupby(['geo_unit'])['Trip_Distance_Weighted'].sum()).reset_index()
plz_dist_car=plz_dist_car.merge(plz_per_weight)
plz_dist_car['Daily_Distance_Person_Car']=0.001*round(plz_dist_car['Trip_Distance_Weighted']/plz_dist_car['Per_Weight'])
plz_dist_car.drop(columns=['Trip_Distance_Weighted','Per_Weight'],inplace=True)

# car ownhership rates by household
plz_hh_weight=pd.DataFrame(sHPW[['geo_unit', 'HHNR','HH_Weight']].drop_duplicates().groupby(['geo_unit'])['HH_Weight'].sum()).reset_index() 
plz_hh_car=pd.DataFrame(sHPW.loc[sHPW['CarAvailable']==1,('geo_unit', 'HHNR','HH_Weight')].drop_duplicates().groupby(['geo_unit'])['HH_Weight'].sum()).reset_index() 
plz_hh_car.rename(columns={'HH_Weight':'HH_WithCar'},inplace=True)
plz_hh_car=plz_hh_car.merge(plz_hh_weight)
plz_hh_car['CarOwnership_HH']=round(plz_hh_car['HH_WithCar']/plz_hh_car['HH_Weight'],3)
plz_hh_car.drop(columns=['HH_WithCar','HH_Weight'],inplace=True)

summary=msdrp.merge(plz_dist_weight)
summary=summary.merge(plz_dist_car)
summary=summary.merge(plz_hh_car)
summary.to_csv('../outputs/Summary_geounits/'+city+'.csv',index=False)
# plot histograms of travel distances by each and all modes
per_dist_mode=pd.DataFrame(sHPW.groupby(['HH_PNR','Mode'])['Trip_Distance'].sum()*0.001).reset_index()
fig, ax = plt.subplots()
plt.hist(per_dist_mode.loc[per_dist_mode['Mode']=='Car','Trip_Distance'],bins=range(0, 105, 5),color='red')
plt.title('Avg. Daily Travel by Car, ' + city,fontsize=16)
plt.xlabel('km/cap/day',fontsize=12)
plt.ylabel('Respondents',fontsize=12)
ax.grid()
fig.savefig('../figures/plots/'+ city+'_Daily_Car.png',facecolor='w')

fig, ax = plt.subplots()
plt.hist(per_dist_mode.loc[per_dist_mode['Mode']=='Transit','Trip_Distance'],bins=range(0, 105, 5),color='green')
plt.title('Avg. Daily Travel by Transit, ' + city,fontsize=16)
plt.xlabel('km/cap/day',fontsize=12)
plt.ylabel('Respondents',fontsize=12)
ax.grid()
fig.savefig('../figures/plots/'+ city+'_Daily_Transit.png',facecolor='w')

fig, ax = plt.subplots()
plt.hist(per_dist_mode.loc[per_dist_mode['Mode']=='Foot','Trip_Distance'],bins=range(0, 20, 1),color='blue')
plt.title('Avg. Daily Travel by Foot, ' + city,fontsize=16)
plt.xlabel('km/cap/day',fontsize=12)
plt.ylabel('Respondents',fontsize=12)
ax.grid()
ax.set_xticks(range(0, 20, 2))
fig.savefig('../figures/plots/'+ city+'_Daily_Foot.png',facecolor='w')

fig, ax = plt.subplots()
plt.hist(per_dist_mode.loc[per_dist_mode['Mode']=='Bike','Trip_Distance'],bins=range(0, 105, 5),color='Orange')
plt.title('Avg. Daily Travel by Bike, ' + city,fontsize=16)
plt.xlabel('km/cap/day',fontsize=12)
plt.ylabel('Respondents',fontsize=12)
ax.grid()
fig.savefig('../figures/plots/'+ city+'_Daily_Bike.png',facecolor='w')

person_dist_all=pd.DataFrame(sHPW.groupby(['HH_PNR'])['Trip_Distance'].sum()*0.001).reset_index()
fig, ax = plt.subplots()
plt.hist(person_dist_all['Trip_Distance'],bins=range(0, 105, 5),color='purple')
plt.title('Avg. Daily Travel, All Modes, ' + city,fontsize=16)
plt.xlabel('km/cap/day',fontsize=12)
plt.ylabel('Respondents',fontsize=12)
ax.grid()
fig.savefig('../figures/plots/'+ city+'_Daily_AllModes.png',facecolor='w')

# make dist categories
sHPW['Trip_Distance_Cat']=np.nan
sHPW.loc[sHPW['Trip_Distance']<1000,'Trip_Distance_Cat']='0-1'
sHPW.loc[(sHPW['Trip_Distance']>=1000)&(sHPW['Trip_Distance']<2000),'Trip_Distance_Cat']='1-2'
sHPW.loc[(sHPW['Trip_Distance']>=2000)&(sHPW['Trip_Distance']<4000),'Trip_Distance_Cat']='2-4'
sHPW.loc[(sHPW['Trip_Distance']>=4000)&(sHPW['Trip_Distance']<8000),'Trip_Distance_Cat']='4-8'
sHPW.loc[(sHPW['Trip_Distance']>=8000),'Trip_Distance_Cat']='8+'

# plot trip mode by distance
mode_dist=sHPW.loc[:,('Mode','Trip_Distance_Cat','Trip_Purpose')].groupby(['Mode','Trip_Distance_Cat']).count().unstack('Mode').reset_index()
mode_dist.columns = ['_'.join(col) for col in mode_dist.columns.values]
mode_dist.reset_index(inplace=True,drop=True)
mode_dist.columns=[col.replace('Trip_Purpose_','') for col in mode_dist.columns]

mode_dist_pc=sHPW.loc[:,('Mode','Trip_Distance_Cat','Trip_Weight')].groupby(['Mode','Trip_Distance_Cat']).sum().unstack('Mode').reset_index()
mode_dist_pc.columns = ['_'.join(col) for col in mode_dist_pc.columns.values]
mode_dist_pc.reset_index(inplace=True,drop=True)
mode_dist_pc.columns=[col.replace('Trip_Weight_','') for col in mode_dist_pc.columns]
mode_dist_pc.loc[:,('2_3_Wheel','Bike','Car','Foot','Transit')]=100*mode_dist_pc.loc[:,('2_3_Wheel','Bike','Car','Foot','Transit')]/sHPW.loc[:,('Trip_Weight')].sum()
fig, ax = plt.subplots()
mode_dist.plot(kind='bar',stacked=True,ax=ax)
ax.set_title('Trip Mode by Distance, '+city,color='black',fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1.0, 1.0),fontsize=12)
ax.set_xticklabels(mode_dist['Trip_Distance_Cat_'].values)
plt.xticks(rotation = 0,fontsize=12)
plt.xlabel('Distance Bands (km)',fontsize=12)
plt.ylabel('Num. Trips',fontsize=12)
fig.savefig('../figures/bars/'+ city+'_ModeDistance.png',facecolor='w',bbox_inches='tight')

fig, ax = plt.subplots()
mode_dist_pc.plot(kind='bar',stacked=True,ax=ax)
ax.set_title('Trip Mode by Distance, '+city,color='black',fontsize=20)
ax.set_ylim(0,50)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1.0, 1.0),fontsize=14)
ax.set_xticklabels(mode_dist_pc['Trip_Distance_Cat_'].values)
plt.xticks(rotation = 0,fontsize=14)
plt.xlabel('Distance Bands (km)',fontsize=14)
plt.ylabel('Share of Trips (%)',fontsize=14)
fig.savefig('../figures/bars/'+ city+'_ModeDistance_pc.png',facecolor='w',bbox_inches='tight')

# plot trip mode by purpose and distance 
mode_purp=sHPW.loc[:,('Mode','Trip_Distance_Cat','Trip_Purpose_Agg')].groupby(['Mode','Trip_Purpose_Agg']).count().unstack('Mode').reset_index()
mode_purp.columns = [''.join(col) for col in mode_purp.columns.values]
mode_purp.reset_index(inplace=True,drop=True)
mode_purp.columns=[col.replace('Trip_Distance_Cat','') for col in mode_purp.columns]
mode_purp.fillna(0,inplace=True)

mode_purp_dist=sHPW.loc[:,('Mode','Trip_Distance_Cat','Trip_Purpose_Agg','Trip_Time')].groupby(['Mode','Trip_Purpose_Agg','Trip_Distance_Cat']).count().unstack('Mode').reset_index()
mode_purp_dist.columns = [''.join(col) for col in mode_purp_dist.columns.values]
mode_purp_dist.reset_index(inplace=True,drop=True)
mode_purp_dist.columns=[col.replace('Trip_Time','') for col in mode_purp_dist.columns]
mode_purp_dist.fillna(0,inplace=True)

fig, ax = plt.subplots(3,2,figsize=(14,12))
axp=ax[0][0]
mode_purp_dist.loc[mode_purp_dist['Trip_Distance_Cat']=='0-1',].plot(kind='bar',stacked=True,ax=axp,legend=False)
axp.set_title('Distance = 0-1km',color='black',fontsize=16)
axp.set_xticklabels(mode_purp_dist['Trip_Purpose_Agg'].unique())
axp.get_xaxis().set_visible(False)
plt.sca(axp)
plt.ylabel('Num. Trips',fontsize=12)

axp=ax[0][1]
mode_purp_dist.loc[mode_purp_dist['Trip_Distance_Cat']=='1-2',].plot(kind='bar',stacked=True,ax=axp)
axp.set_title('Distance = 1-2km',color='black',fontsize=16)
handles, labels = axp.get_legend_handles_labels()
axp.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1.0, 1.0),fontsize=12)
axp.set_xticklabels(mode_purp_dist['Trip_Purpose_Agg'].unique())
axp.get_xaxis().set_visible(False)

axp=ax[1][0]
mode_purp_dist.loc[mode_purp_dist['Trip_Distance_Cat']=='2-4',].plot(kind='bar',stacked=True,ax=axp,legend=False)
axp.set_title('Distance = 2-4km',color='black',fontsize=16)
axp.set_xticklabels(mode_purp_dist['Trip_Purpose_Agg'].unique())
axp.get_xaxis().set_visible(False)
plt.sca(axp)
plt.ylabel('Num. Trips',fontsize=12)

axp=ax[1][1]
mode_purp_dist.loc[mode_purp_dist['Trip_Distance_Cat']=='4-8',].plot(kind='bar',stacked=True,ax=axp)
axp.set_title('Distance = 4-8km',color='black',fontsize=16)
handles, labels = axp.get_legend_handles_labels()
axp.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1.0, 1.0),fontsize=12)
axp.set_xticklabels(mode_purp_dist['Trip_Purpose_Agg'].unique())
axp.get_xaxis().set_visible(False)

axp=ax[2][0]
mode_purp_dist.loc[mode_purp_dist['Trip_Distance_Cat']=='8+',].plot(kind='bar',stacked=True,ax=axp,legend=False)
axp.set_title('Distance >8km',color='black',fontsize=16)
axp.set_xticklabels(mode_purp_dist['Trip_Purpose_Agg'].unique())

plt.sca(axp)
plt.xticks(rotation=45,fontsize=12,ha='right',rotation_mode='anchor')
plt.ylabel('Num. Trips',fontsize=12)

axp=ax[2][1]
mode_purp.plot(kind='bar',stacked=True,ax=axp)
axp.set_title('All Distances',color='black',fontsize=16)
handles, labels = axp.get_legend_handles_labels()
axp.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1.0, 1.0),fontsize=12)
axp.set_xticklabels(mode_purp['Trip_Purpose_Agg'].unique())
#axp.text(s='All Dist', x=0.05, y=0.9,transform=axp.transAxes,fontsize=14)
plt.sca(axp)
plt.xticks(rotation=45,fontsize=12,ha='right',rotation_mode='anchor')

fig.suptitle("Trip Mode by Purpose & Distance, " + city, fontsize=22,y=0.95)
fig.savefig('../figures/bars/'+ city+'_ModePurposeDistance.png',facecolor='w',bbox_inches='tight')

# # calcuate trip speeds and summarize by mode and distance
# sHPW.replace([np.inf, -np.inf], np.nan, inplace=True)
# print('Average speed by mode and distance category: ' + city)
# round(sHPW.dropna(axis=0,subset='Trip_Speed').groupby(['Trip_Distance_Cat','Mode'])['Trip_Speed'].mean(),2)