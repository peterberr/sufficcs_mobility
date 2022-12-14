# script to combine and harmonize survey data for Paris, and calculate some summary statistics

def import_csv_w_wkt_to_gdf(path,crs,geometry_col='geometry'):
	'''
	Import a csv file with WKT geometry column into a GeoDataFrame

    Last modified: 12/09/2020. By: Nikola

    '''

	df = pd.read_csv(path)
	gdf = gpd.GeoDataFrame(df, 
						geometry=df[geometry_col].apply(wkt.loads),
						crs=crs)
	return(gdf)

import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import trunc
from shapely import wkt

# load in city survey microdata
city='Paris'

# inputs:
# Paris dictionary pickle files, to create harmonised variables from the raw survey data
# SrV survey data files (houeshold, population, trips) for Paris 
# population density by spatial unit for Paris
# other pre-calculated urban form features (building density, distance to center, land-use data, connectivity metrics) by spatial unit 

if city == 'Paris':
    fn_hh='../../MSCA_data/FranceRQ/lil-0883_IleDeFrance.csv/Csv/menages_semaine.csv'
    fn_p='../../MSCA_data/FranceRQ/lil-0883_IleDeFrance.csv/Csv/personnes_semaine.csv'
    fn_t='../../MSCA_data/FranceRQ/lil-0883_IleDeFrance.csv/Csv/deplacements_semaine.csv'
    # fn_hh0='../../MSCA_data/FranceRQ/lil-0933_Paris.csv/Csv/Fichiers_Original/toulouse_2013_ori_men.csv'

# For Tolouse, first get household income from the original data SH0, as this is excluded from the standardised data.

# sH0=pd.read_csv(fn_hh0,sep=';')
sH=pd.read_csv(fn_hh,sep=';')
sP=pd.read_csv(fn_p,sep=';')
sW=pd.read_csv(fn_t,sep=';')

# load pickled dictionary files
with open('../dictionaries/Paris_var.pkl','rb') as f:
    var_all = pickle.load(f)

with open('../dictionaries/Paris_val.pkl','rb') as f:
    value_all = pickle.load(f)

with open('../dictionaries/Paris_na.pkl','rb') as f:
    na_all = pickle.load(f)

# map old variable names and values to new variable names and values

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

# define, sector, zone, and household id !! city specific !!
sH['geo_unit'] = sH.loc[:,'Commune']
sP['geo_unit'] = sP.loc[:,'Commune']
sP['HH_PNR']=sP['HHNR'].astype('str')+'_'+sP['Person'].astype('str')
sW['geo_unit'] = sW.loc[:,'Commune']
sW['HH_PNR']=sW['HHNR'].astype('str')+'_'+sW['Person'].astype('str')
sW['HH_P_WNR']=sW['HH_PNR'].astype('str')+'_'+sW['Trip'].astype('str')

# bring HHNR and geo_unit to the left side of the HH df
cols=sH.columns.tolist()
cols_new = ['HHNR', 'geo_unit'] + [value for value in cols if value not in {'HHNR', 'geo_unit'}]
sH=sH[cols_new]

# bring HHNR, HH_PNR, to the left side of the Per df
cols=sP.columns.tolist()
cols_new = ['HHNR','HH_PNR'] + [value for value in cols if value not in {'HHNR', 'HH_PNR'}]
sP=sP[cols_new]

# bring HHNR, HH_PNR, HH_P_WNR to the left side of the W df
cols=sW.columns.tolist()
cols_new = ['HHNR','HH_PNR','HH_P_WNR'] + [value for value in cols if value not in {'HHNR', 'HH_PNR','HH_P_WNR'}]
sW=sW[cols_new]

# calculate household size from the sP data !! Not required for Paris !!

# address inconsistencies with the Person 'Education' variable, arising from respondents who have completed a certain level of education/training responding no dimploma yet, even though they have lower diplomas than the one they are currently studying for
# these assumptions are based on French law, in which it is mandatory to go to school until age 16 (end of secondary school), so anyone in an occupation post-16 is very likely to have some education. https://www.expatica.com/fr/education/children-education/french-education-system-101147/
# they may need defined differently for non-French surveys
sP.loc[(sP['Age']>11) & (sP['Age']<17) & (sP['Education']=='No diploma yet'),'Education']="Elementary" # if aged between 12 and 16, assume at least an Elementary education
sP.loc[(sP['Age']>16) & (sP['Age']<20) & (sP['Education']=='No diploma yet'),'Education']="Secondary" # if aged between 17 and 19, assume at least a Secondary education.
sP.loc[(sP['Age']>15) & (sP['Occupation']=='Student_3rdLevel') & (sP['Education']=='No diploma yet'),'Education']="Secondary+BAC" # if a 3rd level student, assume at least Secondary eduction with BAC
sP.loc[(sP['Age']>15) & (sP['Occupation']=='Trainee') & (sP['Education']=='No diploma yet'),'Education']="Secondary" # If a trainee, assume at least a secondary education

# address the NA values for Education and Occupation for children under 5
sP.loc[sP['Age']<5,['Education','Occupation']]='Pre-School'

# # create the origin and destination id's based on what we extracted from the microdata !! this is city specific !!
sW['Ori_geo_unit']=sW.loc[:,'Ori_Comm']
sW['Des_geo_unit']=sW.loc[:,'Des_Comm']

# define the simplified origin and destination reasons. !! This is city specific !! 
sW['Ori_Reason_Agg']='Other'
sW.loc[sW['Ori_Reason1']==1,'Ori_Reason_Agg']='Home'
sW.loc[sW['Ori_Reason1'].isin([2,3]),'Ori_Reason_Agg']='Work'
sW.loc[sW['Ori_Reason1']==6,'Ori_Reason_Agg']='Personal'
sW.loc[sW['Ori_Reason1']==4,'Ori_Reason_Agg']='School'
sW.loc[sW['Ori_Reason1']==5,'Ori_Reason_Agg']='Shopping'
sW.loc[sW['Ori_Reason1']==8,'Ori_Reason_Agg']='Leisure'
sW.loc[sW['Ori_Reason1']==7,'Ori_Reason_Agg']='Accompanying/Kids'

sW['Des_Reason_Agg']='Other'
sW.loc[sW['Des_Reason1']==1,'Des_Reason_Agg']='Home'
sW.loc[sW['Des_Reason1'].isin([2,3]),'Des_Reason_Agg']='Work'
sW.loc[sW['Des_Reason1']==6,'Des_Reason_Agg']='Personal'
sW.loc[sW['Des_Reason1']==4,'Des_Reason_Agg']='School'
sW.loc[sW['Des_Reason1']==5,'Des_Reason_Agg']='Shopping'
sW.loc[sW['Des_Reason1']==8,'Des_Reason_Agg']='Leisure'
sW.loc[sW['Des_Reason1']==7,'Des_Reason_Agg']='Accompanying/Kids'

sW['trip_type_all']=sW['Ori_Reason_Agg']+'-'+sW['Des_Reason_Agg']

# now calculate the detailed o-d trip purposes, this should be harmonized 
sW['Trip_Purpose']='Other'
sW.loc[(sW['Ori_Reason_Agg'].isin(['Home','Personal'])) & (sW['Des_Reason_Agg'].isin(['Home','Personal'])),'Trip_Purpose']='Home???Personal' #
sW.loc[(sW['Ori_Reason_Agg'].isin(['Home','Accompanying/Kids'])) & (sW['Des_Reason_Agg'].isin(['Home','Accompanying/Kids'])),'Trip_Purpose']='Home???Companion' #
sW.loc[(sW['Ori_Reason_Agg'].isin(['Work','Accompanying/Kids'])) & (sW['Des_Reason_Agg'].isin(['Work','Accompanying/Kids'])),'Trip_Purpose']='Work???Companion' #
sW.loc[(sW['Ori_Reason_Agg'].isin(['Home','Other'])) & (sW['Des_Reason_Agg'].isin(['Home','Other'])),'Trip_Purpose']='Other???Home'
sW.loc[sW['trip_type_all']=='Home-Shopping','Trip_Purpose']='Home-Shopping'
sW.loc[sW['trip_type_all']=='Shopping-Home','Trip_Purpose']='Shopping-Home'
sW.loc[sW['trip_type_all']=='Home-School','Trip_Purpose']='Home-School'
sW.loc[sW['trip_type_all']=='School-Home','Trip_Purpose']='School-Home'
sW.loc[sW['trip_type_all']=='Home-Work','Trip_Purpose']='Home-Work'
sW.loc[sW['trip_type_all']=='Work-Home','Trip_Purpose']='Work-Home'
sW.loc[sW['trip_type_all']=='Home-Leisure','Trip_Purpose']='Home-Leisure'
sW.loc[sW['trip_type_all']=='Leisure-Home','Trip_Purpose']='Leisure-Home'
sW.loc[sW['trip_type_all'].isin(['Shopping-Shopping','Personal-Shopping','Leisure-Shopping']),'Trip_Purpose']='Shopping' # this is kind of unique because we include the personal/leisure origins before the shopping destination
sW.loc[sW['trip_type_all']=='Work-Work','Trip_Purpose']='Work'
sW.loc[sW['trip_type_all']=='Work-Work','Trip_Purpose']='Work'
sW.loc[sW['trip_type_all']=='Work-Leisure','Trip_Purpose']='Work-Leisure'
sW.loc[sW['trip_type_all']=='Work-Shopping','Trip_Purpose']='Work-Shopping'
sW.loc[sW['trip_type_all']=='Leisure-Work','Trip_Purpose']='Leisure-Work'
sW.loc[sW['trip_type_all']=='Leisure-Leisure','Trip_Purpose']='Leisure'
sW.loc[sW['trip_type_all']=='Accompanying/Kids-Accompanying/Kids','Trip_Purpose']='Companion'
sW.loc[sW['trip_type_all']=='School-Leisure','Trip_Purpose']='School-Leisure'
sW.loc[sW['trip_type_all']=='Leisure-School','Trip_Purpose']='Leisure-School'

# make the aggregated trip purpose, pick up here
sW['Trip_Purpose_Agg']='Other'
sW.loc[sW['Trip_Purpose'].isin(['Home-Work','Work-Home']),'Trip_Purpose_Agg']='Home???Work'
sW.loc[sW['Trip_Purpose'].isin(['Home-School','School-Home']),'Trip_Purpose_Agg']='Home???School'
sW.loc[sW['Trip_Purpose'].isin(['Home-Shopping','Shopping-Home']),'Trip_Purpose_Agg']='Home???Shopping'
sW.loc[sW['Trip_Purpose'].isin(['Home???Companion']),'Trip_Purpose_Agg']='Home???Companion'
sW.loc[sW['Trip_Purpose'].isin(['Home-Leisure','Leisure-Home','Home???Personal']),'Trip_Purpose_Agg']='Home???Leisure'

sW['Hour']=sW.loc[:,'Time']
sW.loc[sW['Hour']>24,'Hour']=sW.loc[sW['Hour']>24,'Hour']-24

sW['Trip_Time']='Nighttime Off-Peak'
sW.loc[sW['Hour'].isin([7,8,9]),'Trip_Time']='AM_Rush'
sW.loc[sW['Hour'].isin([12,13]),'Trip_Time']='Lunch'
sW.loc[sW['Hour'].isin([16,17,18]),'Trip_Time']='PM Rush'
sW.loc[sW['Hour'].isin([19,20]),'Trip_Time']='Evening'
sW.loc[sW['Hour'].isin([10,11,14,15]),'Trip_Time']='Daytime Off-Peak'

# we can drop the commune and geo_unit variables as these refer to the location of residence, not the origin/destination of the trip, which are coded elsewhere
sW.drop(columns=['Commune','geo_unit','Grid_Cell','HHNR','Person', 'Ori_Reason1','Ori_Reason2', 'Des_Reason1', 'Des_Reason2','Hour','Time'],inplace=True)
sP.drop(columns=['Commune','geo_unit','Grid_Cell'],inplace=True)

# merge together the household, person, and trip files
sHP=sH.merge(sP,on='HHNR')
sHPW=sHP.merge(sW,on='HH_PNR')
# bring HHNR, HH_PNR, HH_P_WNR to the left side of the combined df
cols=sHPW.columns.tolist()
cols_new = ['HHNR','HH_PNR','HH_P_WNR','geo_unit','Ori_Comm','Des_Comm','Trip_Time','Trip_Purpose'] + [value for value in cols if value not in {'HHNR','HH_PNR','HH_P_WNR','geo_unit','Ori_Sec_Zone','Des_Sec_Zone','Trip_Time','Trip_Purpose','Mode', 'Trip_Distance'}] +['Mode', 'Trip_Distance']
sHPW=sHPW[cols_new]

# Dataframe cleaningrestrict data frame to trips starting within the defined city boundary, and to trips within a certain distance range (e.g. 100m - 50km). Also remove unknown mode
# For Paris, we initially use the low-density geo_units as the resolution, for this list, can load in the density file.
fp = '../shapefiles/density_geounits/Paris_pop_density_lowres.csv'
pa = import_csv_w_wkt_to_gdf(fp,crs=3035,geometry_col='geometry')

# use the code dictionary to create the origin and destination geocodes
sHPW['Ori_geocode']=sW.loc[:,'Ori_Comm']
sHPW['Des_geocode']=sW.loc[:,'Des_Comm']

# remove trips that don't start within the region where we collect urban features data
sHPW=sHPW.loc[sHPW['Ori_geocode'].isin(pa['geo_unit']),:]
# further cleaning to retrict the trips to those between 0.1km and 50km
sHPW=sHPW.loc[(sHPW['Trip_Distance']>=0.05) & (sHPW['Trip_Distance']<=50),:]
sHPW=sHPW.loc[sHPW['Mode']!='Other',:]

# calculate and print some summary stats
mode_share=sHPW.groupby('Mode')['Trip_Distance'].sum()/sum(sHPW.groupby('Mode')['Trip_Distance'].sum())
weighted=sHPW.loc[:,('Per_Weight','Mode','Trip_Distance')]
weighted['Dist_Weighted_P']=weighted['Per_Weight']*weighted['Trip_Distance']
mode_share_weighted=weighted.groupby('Mode')['Dist_Weighted_P'].sum()/sum(weighted.groupby('Mode')['Dist_Weighted_P'].sum())

print('Weighted mode share')
print(mode_share_weighted)

print('N Trips')
print(len(sHPW))

print('Avg trip distance in km, overall: ', round(np.average(sHPW['Trip_Distance'],weights=sHPW['Per_Weight']),1))

person_trips=pd.DataFrame(sHPW.groupby('HH_PNR')['Trip_Distance'].sum()).reset_index()
print('Average travel distance per person per day, all modes, km/cap: ' , str(round(person_trips['Trip_Distance'].mean(),1)))

person_mode_trips=pd.DataFrame(sHPW.groupby(['HH_PNR','Mode'])['Trip_Distance'].sum()).reset_index()
print('Average travel distance per person per day, all modes, km/cap: ')
print(str(round(person_mode_trips.groupby('Mode')['Trip_Distance'].sum()/len(person_trips),2)))

if len(sHPW.loc[:,('HH_PNR','Day')].drop_duplicates())!=len(person_trips):
    print('NB! Some respondents report trips over more than one day')

#sHPW['Trip_Speed']=round(60*sHPW['Trip_Distance']/(sHPW['Trip_Duration']),2)

sHPW.to_csv('../outputs/Combined/'+city+'.csv',index=False)

# now create and save some summary stats by postcode
sHPW['Trip_Distance_Weighted']=sHPW['Trip_Distance']*sHPW['Per_Weight']

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
plz_dist_weight['Daily_Distance_Person']=round(plz_dist_weight['Trip_Distance_Weighted']/plz_dist_weight['Per_Weight'],3)
plz_dist_weight.drop(columns=['Trip_Distance_Weighted','Per_Weight'],inplace=True)
# and car travel distance by residence postcode
plz_dist_car=pd.DataFrame(sHPW.loc[sHPW['Mode']=='Car',:].groupby(['geo_unit'])['Trip_Distance_Weighted'].sum()).reset_index()
plz_dist_car=plz_dist_car.merge(plz_per_weight)
plz_dist_car['Daily_Distance_Person_Car']=round(plz_dist_car['Trip_Distance_Weighted']/plz_dist_car['Per_Weight'],3)
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
per_dist_mode=pd.DataFrame(sHPW.groupby(['HH_PNR','Mode'])['Trip_Distance'].sum()).reset_index()
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

person_dist_all=pd.DataFrame(sHPW.groupby(['HH_PNR'])['Trip_Distance'].sum()).reset_index()
fig, ax = plt.subplots()
plt.hist(person_dist_all['Trip_Distance'],bins=range(0, 105, 5),color='purple')
plt.title('Avg. Daily Travel, All Modes, ' + city,fontsize=16)
plt.xlabel('km/cap/day',fontsize=12)
plt.ylabel('Respondents',fontsize=12)
ax.grid()
fig.savefig('../figures/plots/'+ city+'_Daily_AllModes.png',facecolor='w')

# prepare to plot car ownership and distance traveled by income
sHPW['Income']=np.nan
sHPW.loc[sHPW['IncomeDetailed']=='Under800','Income']=500
sHPW.loc[sHPW['IncomeDetailed']=='800-1200','Income']=1000
sHPW.loc[sHPW['IncomeDetailed']=='1200-1600','Income']=1400
sHPW.loc[sHPW['IncomeDetailed']=='1600-2000','Income']=1800
sHPW.loc[sHPW['IncomeDetailed']=='2000-2400','Income']=2200
sHPW.loc[sHPW['IncomeDetailed']=='2400-3000','Income']=2700
sHPW.loc[sHPW['IncomeDetailed']=='3000-3500','Income']=3250
sHPW.loc[sHPW['IncomeDetailed']=='3500-4500','Income']=4000
sHPW.loc[sHPW['IncomeDetailed']=='4500-5500','Income']=5000
sHPW.loc[sHPW['IncomeDetailed']=='Over5500','Income']=6000

# plot car ownership by income 
hh_inc_car=pd.DataFrame(sHPW.groupby(['HHNR'])['Income','CarAvailable'].mean()).reset_index()
inc_car=hh_inc_car.groupby('Income')['CarAvailable'].mean().reset_index()
fig, ax = plt.subplots()
plt.plot(inc_car['Income'],100*inc_car['CarAvailable'])
plt.title('Car Ownership by Income, '+city,fontsize=16)
plt.xlabel('HH Income, EUR/month',fontsize=12)
plt.ylabel('HH Car Ownership Rate (%)',fontsize=12)
ax.grid()
ax.set_ylim(0,95)
fig.savefig('../figures/plots/'+ city+'_CarOwnership.png',facecolor='w')

# plot Car Mode Share by Income
person_dist_car=per_dist_mode.loc[per_dist_mode['Mode']=='Car',:]
person_dist_car.rename(columns={'Trip_Distance':'Trip_Distance_Car'},inplace=True)
person_dist_car.drop(columns='Mode',inplace=True)

person_income=pd.DataFrame(sHPW.groupby(['HH_PNR'])['Income'].mean()).reset_index()
person_income_dist=person_income.merge(person_dist_all)
person_income_dist=person_income_dist.merge(person_dist_car,how='left')
person_income_dist['Trip_Distance_Car'].fillna(0,inplace=True)
person_mode_income=pd.DataFrame(person_income_dist.groupby(['Income'])['Trip_Distance','Trip_Distance_Car'].mean()).reset_index()
person_mode_income['Car_ModeShare']=person_mode_income['Trip_Distance_Car']/person_mode_income['Trip_Distance']

fig, ax = plt.subplots()
plt.plot(person_mode_income['Income'],100*person_mode_income['Car_ModeShare'])
plt.title('Car Mode Share by Income, ' + city,fontsize=16)
plt.xlabel('HH Income, EUR/month',fontsize=12)
plt.ylabel('Share of Travel Dist. by Car (%)',fontsize=12)
ax.grid()
ax.set_ylim(0,75)
fig.savefig('../figures/plots/'+ city+'_CarModeShare.png',facecolor='w')
inc_stats=inc_car.merge(person_mode_income)
inc_stats['City']=city
inc_stats.to_csv('../figures/plots/income_stats_Paris.csv',index=False)


# make dist categories
# make dist categories
sHPW['Trip_Distance_Cat']=np.nan
sHPW.loc[sHPW['Trip_Distance']<1,'Trip_Distance_Cat']='0-1'
sHPW.loc[(sHPW['Trip_Distance']>=1)&(sHPW['Trip_Distance']<2),'Trip_Distance_Cat']='1-2'
sHPW.loc[(sHPW['Trip_Distance']>=2)&(sHPW['Trip_Distance']<4),'Trip_Distance_Cat']='2-4'
sHPW.loc[(sHPW['Trip_Distance']>=4)&(sHPW['Trip_Distance']<8),'Trip_Distance_Cat']='4-8'
sHPW.loc[(sHPW['Trip_Distance']>=8),'Trip_Distance_Cat']='8+'


# plot trip mode by distance
mode_dist=sHPW.loc[:,('Mode','Trip_Distance_Cat','Trip_Purpose')].groupby(['Mode','Trip_Distance_Cat']).count().unstack('Mode').reset_index()
mode_dist.columns = ['_'.join(col) for col in mode_dist.columns.values]
mode_dist.reset_index(inplace=True,drop=True)
mode_dist.columns=[col.replace('Trip_Purpose_','') for col in mode_dist.columns]

# mode_dist_pc=mode_dist.copy()
# mode_dist_pc.loc[:,('2_3_Wheel','Bike','Car','Foot','Transit')]=100*mode_dist_pc.loc[:,('2_3_Wheel','Bike','Car','Foot','Transit')]/len(sHPW)

mode_dist_pc=sHPW.loc[:,('Mode','Trip_Distance_Cat','Per_Weight')].groupby(['Mode','Trip_Distance_Cat']).sum().unstack('Mode').reset_index()
mode_dist_pc.columns = ['_'.join(col) for col in mode_dist_pc.columns.values]
mode_dist_pc.reset_index(inplace=True,drop=True)
mode_dist_pc.columns=[col.replace('Per_Weight_','') for col in mode_dist_pc.columns]
mode_dist_pc.loc[:,('2_3_Wheel','Bike','Car','Foot','Transit')]=100*mode_dist_pc.loc[:,('2_3_Wheel','Bike','Car','Foot','Transit')]/sHPW.loc[:,('Per_Weight')].sum()

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

# sHPW.replace([np.inf, -np.inf], np.nan, inplace=True)
# print('Average speed by mode and distance category: ' + city)
# print(round(sHPW.dropna(axis=0,subset='Trip_Speed').groupby(['Trip_Distance_Cat','Mode'])['Trip_Speed'].mean(),2))