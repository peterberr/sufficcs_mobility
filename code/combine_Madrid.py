# script to combine and harmonize survey data for Madrid, and calculate some summary statistics

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import trunc
city = 'Madrid'

# inputs:
# Madrid dictionary pickle files, to create harmonised variables from the raw survey data
# SrV survey data files (houeshold, population, trips) for Madrid 
# population density by spatial unit for Madrid
# other pre-calculated urban form features (building density, distance to center, land-use data, connectivity metrics) by spatial unit 

# All Madrid data can be accessed at https://datos.comunidad.madrid/catalogo/dataset/resultados-edm2018
fn_hh='../../MSCA_data/madrid/EDM2018/EDM2018HOGARES.csv'
fn_p='../../MSCA_data/madrid/EDM2018/EDM2018INDIVIDUOS.csv'
fn_t='../../MSCA_data/madrid/EDM2018/EDM2018VIAJES.csv'

sH=pd.read_csv(fn_hh,sep=',',encoding='latin-1')
sP=pd.read_csv(fn_p,sep=',',encoding='latin-1')
sW=pd.read_csv(fn_t,sep=',',encoding='latin-1')

# load pickled dictionary files
with open('../dictionaries/Madrid_var.pkl','rb') as f:
    var_all = pickle.load(f)

with open('../dictionaries/Madrid_val.pkl','rb') as f:
    value_all = pickle.load(f)

with open('../dictionaries/Madrid_na.pkl','rb') as f:
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

# make a concordance of high-res zones to low-res sectors
sec_zone=sH.loc[:,('Sector','Zone')].drop_duplicates().reset_index(drop=True)
# check there is no mapping of detailed zones 
max(sec_zone['Zone'].value_counts())<2
sec_zone_dict=sec_zone.to_dict(orient='records')

# define unique person and trip numbers
sP['HH_PNR']=sP['HHNR'].astype('str')+'_'+sP['Person'].astype('str')
sW['HH_PNR']=sW['HHNR'].astype('str')+'_'+sW['Person'].astype('str')
sW['HH_P_WNR']=sW['HH_PNR']+'_'+sW['Trip'].astype('str')

# bring HHNR and geo_unit to the left side of the HH df
cols=sH.columns.tolist()
cols_new = ['HHNR', 'Sector','Zone'] + [value for value in cols if value not in {'HHNR','Sector','Zone'}]
sH=sH[cols_new]

# bring HHNR, HH_PNR, to the left side of the Per df
cols=sP.columns.tolist()
cols_new = ['HHNR','HH_PNR'] + [value for value in cols if value not in {'HHNR', 'HH_PNR'}]
sP=sP[cols_new]

# bring HHNR, HH_PNR, HH_P_WNR to the left side of the W df
cols=sW.columns.tolist()
cols_new = ['HHNR','HH_PNR','HH_P_WNR'] + [value for value in cols if value not in {'HHNR', 'HH_PNR','HH_P_WNR'}]
sW=sW[cols_new]


# define houshold vehicle ownership
sH['CarAvailable']=0
sH.loc[sH['Type_V1'].isin([1,3,4]),'CarAvailable']=1
sH.loc[sH['Type_V2'].isin([1,3,4]),'CarAvailable']=1
sH.loc[sH['Type_V3'].isin([1,3,4]),'CarAvailable']=1
sH.loc[sH['Type_V4'].isin([1,3,4]),'CarAvailable']=1
sH.loc[sH['Type_V5'].isin([1,3,4]),'CarAvailable']=1

sH['2_3WAvailable']=0
sH.loc[sH['Type_V1']==2,'2_3WAvailable']=1
sH.loc[sH['Type_V2']==2,'2_3WAvailable']=1
sH.loc[sH['Type_V3']==2,'2_3WAvailable']=1
sH.loc[sH['Type_V4']==2,'2_3WAvailable']=1
sH.loc[sH['Type_V5']==2,'2_3WAvailable']=1

# define trip time of day
sW['Hour']=round(sW['Time_Departure']/100) # + round((sW['Time'] % 100)/60) now just use the hours
sW.loc[sW['Hour']>24,'Hour']=sW.loc[sW['Hour']>24,'Hour']-24
# break the trip departure times into grouped times of day
sW['Trip_Time']='Nighttime Off-Peak'
sW.loc[sW['Hour'].isin([6,7,8,9]),'Trip_Time']='AM_Rush'
sW.loc[sW['Hour'].isin([12,13]),'Trip_Time']='Lunch'
sW.loc[sW['Hour'].isin([16,17,18]),'Trip_Time']='PM Rush'
sW.loc[sW['Hour'].isin([19,20,21]),'Trip_Time']='Evening'
sW.loc[sW['Hour'].isin([10,11,14,15]),'Trip_Time']='Daytime Off-Peak'

sW['Ori_Reason_Agg']='Other'
sW.loc[sW['Ori_Reason'].isin([1]),'Ori_Reason_Agg']='Home'
sW.loc[sW['Ori_Reason'].isin([2,3]),'Ori_Reason_Agg']='Work'
sW.loc[sW['Ori_Reason'].isin([6,10]),'Ori_Reason_Agg']='Personal'
sW.loc[sW['Ori_Reason'].isin([4]),'Ori_Reason_Agg']='School'
sW.loc[sW['Ori_Reason'].isin([5]),'Ori_Reason_Agg']='Shopping'
sW.loc[sW['Ori_Reason'].isin([8,9]),'Ori_Reason_Agg']='Leisure'
sW.loc[sW['Ori_Reason'].isin([7]),'Ori_Reason_Agg']='Accompanying/Kids'

sW['Des_Reason_Agg']='Other'
sW.loc[sW['Des_Reason'].isin([1]),'Des_Reason_Agg']='Home'
sW.loc[sW['Des_Reason'].isin([2,3]),'Des_Reason_Agg']='Work'
sW.loc[sW['Des_Reason'].isin([6,10]),'Des_Reason_Agg']='Personal'
sW.loc[sW['Des_Reason'].isin([4]),'Des_Reason_Agg']='School'
sW.loc[sW['Des_Reason'].isin([5]),'Des_Reason_Agg']='Shopping'
sW.loc[sW['Des_Reason'].isin([8,9]),'Des_Reason_Agg']='Leisure'
sW.loc[sW['Des_Reason'].isin([7]),'Des_Reason_Agg']='Accompanying/Kids'

sW['trip_type_all']=sW['Ori_Reason_Agg']+'-'+sW['Des_Reason_Agg']

# now calculate the detailed o-d trip purposes, this should be harmonized 
sW['Trip_Purpose']='Other'
sW.loc[(sW['Ori_Reason_Agg'].isin(['Home','Personal'])) & (sW['Des_Reason_Agg'].isin(['Home','Personal'])),'Trip_Purpose']='Home↔Personal' #
sW.loc[(sW['Ori_Reason_Agg'].isin(['Home','Accompanying/Kids'])) & (sW['Des_Reason_Agg'].isin(['Home','Accompanying/Kids'])),'Trip_Purpose']='Home↔Companion' #
sW.loc[(sW['Ori_Reason_Agg'].isin(['Work','Accompanying/Kids'])) & (sW['Des_Reason_Agg'].isin(['Work','Accompanying/Kids'])),'Trip_Purpose']='Work↔Companion' #
sW.loc[(sW['Ori_Reason_Agg'].isin(['Home','Other'])) & (sW['Des_Reason_Agg'].isin(['Home','Other'])),'Trip_Purpose']='Other↔Home'
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
sW.loc[sW['trip_type_all']=='Accompanying/Kids-Accompanying/Kids','Trip_Purpose']='Companion'

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
# these assumptions are based on Spanish law, in which it is mandatory to go to school until age 16 (end of secondary school), so anyone in an occupation post-16 is very likely to have some education. https://www.expatica.com/es/education/children-education/education-in-spain-103110/
sP.loc[(sP['Age']>11) & (sP['Age']<17) & (sP['Education']=='No diploma yet'),'Education']="Elementary" # if aged between 12 and 16, assume at least an Elementary education
sP.loc[(sP['Age']>16) & (sP['Age']<20) & (sP['Education']=='No diploma yet'),'Education']="Secondary" # if aged between 17 and 19, assume at least a Secondary education.
sP.loc[(sP['Age']>15) & (sP['Occupation']=='Trainee') & (sP['Education']=='No diploma yet'),'Education']="Secondary" # If a trainee, assume at least a secondary education

sW.drop(columns=['Ori_Reason','Des_Reason','HHNR','Person','Hour'],inplace=True)

# calculate trip duration
hr_diff=(sW['Time_Arrival']/100).apply(np.floor)-(sW['Time_Departure']/100).apply(np.floor).astype(int)
sW['Trip_Duration']=hr_diff*60+ (sW.loc[:,('Time_Arrival')] % 100) - (sW.loc[:,('Time_Departure')] % 100)

# merge together the household, person, and trip files
sHP=sH.merge(sP,on='HHNR')

sHPW=sHP.merge(sW,on='HH_PNR')
# bring HHNR, HH_PNR, HH_P_WNR to the left side of the combined df
cols=sHPW.columns.tolist()
cols_new = ['HHNR','HH_PNR','HH_P_WNR','Sector','Zone','Ori_Zone','Des_Zone','Trip_Time','Trip_Purpose'] + [value for value in cols if value not in {'HHNR','HH_PNR','HH_P_WNR','Sector','Zone','Ori_Zone','Des_Zone','Trip_Time','Trip_Purpose','Mode', 'Trip_Distance'}] +['Mode', 'Trip_Distance']
sHPW=sHPW[cols_new]

# load in dictionary of mixed geocodes, which we will use for merging with the urban form features
with open('../dictionaries/' + city + '_mixed_geocode.pkl','rb') as f:
    code_dict = pickle.load(f)

# use the code dictionary to create the origin and destination geocodes
sHPW['Ori_geocode']=sHPW['Ori_Zone'].astype('str').str.zfill(6).map(code_dict)
sHPW['Des_geocode']=sHPW['Des_Zone'].astype('str').str.zfill(6).map(code_dict)
# remove trips that don't start within the region where we collect urban features data
sHPW=sHPW.loc[sHPW['Ori_geocode'].isna()==False,:]

# further cleaning to retrict the trips to those between 0.1km and 50km and remove trips w/o mode information
sHPW['Trip_Distance']=sHPW.loc[:,'Trip_Distance']*1000
sHPW=sHPW.loc[(sHPW['Trip_Distance']>=50) & (sHPW['Trip_Distance']<=50000),:] 
sHPW=sHPW.loc[sHPW['Mode']!='Other',:]

mode_share=sHPW.groupby('Mode')['Trip_Distance'].sum()/sum(sHPW.groupby('Mode')['Trip_Distance'].sum())
weighted=sHPW.loc[:,('Trip_Weight','Mode','Trip_Distance')]
weighted['Dist_Weighted_P']=weighted['Trip_Weight']*weighted['Trip_Distance']
mode_share_weighted=weighted.groupby('Mode')['Dist_Weighted_P'].sum()/sum(weighted.groupby('Mode')['Dist_Weighted_P'].sum())

print('Weighted mode share in ' + city)
print(mode_share_weighted)

print('N Trips')
print(len(sHPW))

print('Avg trip distance in km, overall: ', round(0.001*np.average(sHPW['Trip_Distance'],weights=sHPW['Trip_Weight']),1))

person_trips=pd.DataFrame(sHPW.groupby('HH_PNR')['Trip_Distance'].sum()).reset_index()
print('Average travel distance per person per day, all modes, km/cap: ' , str(round(0.001*person_trips['Trip_Distance'].mean(),1)))

person_mode_trips=pd.DataFrame(sHPW.groupby(['HH_PNR','Mode'])['Trip_Distance'].sum()).reset_index()
print('Average travel distance per person per day, all modes, km/cap: ')
print(str(round(0.001*person_mode_trips.groupby('Mode')['Trip_Distance'].sum()/len(person_trips),2)))

if len(sHPW.loc[:,('HH_PNR','Day')].drop_duplicates())!=len(person_trips): 
    print('NB! Some respondents report trips over more than one day')

# save

#sHPW['Trip_Speed']=round(60*0.001*sHPW['Trip_Distance']/(sHPW['Trip_Duration']),2)
sHPW.to_csv('../outputs/Combined/Madrid.csv',index=False)


# create the ori_geo_unit based on the origin sectors
with open('../dictionaries/Madrid_zone_sec.pkl','rb') as f:
    zone_sec = pickle.load(f)

sHPW['Ori_geo_unit']=sHPW['Ori_Zone'].map(zone_sec)

sHPW['geo_unit']=sHPW['Sector']

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

# calcuate trip speeds and summarize by mode and distance
#sHPW['Trip_Speed']=round(60*0.001*sHPW['Trip_Distance']/(sHPW['Trip_Duration']),2)
# sHPW.replace([np.inf, -np.inf], np.nan, inplace=True)
# print('Average speed by mode and distance category: ' + city)
# round(sHPW.dropna(axis=0,subset='Trip_Speed').groupby(['Trip_Distance_Cat','Mode'])['Trip_Speed'].mean(),2)