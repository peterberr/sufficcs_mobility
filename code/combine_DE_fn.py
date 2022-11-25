# script to combine and harmonize survey data for German cities, and calculate some summary statistics

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# inputs:
# German city dictionary pickle files, to create harmonised variables from the raw survey data
# SrV survey data files (houeshold, population, trips) for German cities
# population density by postcode for German cities
# other pre-calculated urban form features (building density, distance to center, land-use data, connectivity metrics) by postcode for each city


# load pickled dictionary files
with open('../dictionaries/Germany_var.pkl','rb') as f:
    var_all = pickle.load(f)

with open('../dictionaries/Germany_val.pkl','rb') as f:
    value_all = pickle.load(f)

with open('../dictionaries/Germany_na.pkl','rb') as f:
    na_all = pickle.load(f)

with open('../dictionaries/city_postcode_DE.pkl','rb') as f:
    city_plz = pickle.load(f)


inc_stats_all=pd.DataFrame(columns = ['Income', 'CarAvailable', 'Trip_Distance', 'Trip_Distance_Car','Car_ModeShare', 'City'])

def combine_survey_data(city):
    global inc_stats_all
    print(city)
    fn_hh='../../MSCA_data/SrV/' + city + '/SrV2018_Einzeldaten_' + city + '_SciUse_H2018.csv'
    fn_p='../../MSCA_data/SrV/' + city + '/SrV2018_Einzeldaten_' + city + '_SciUse_P2018.csv'
    fn_t='../../MSCA_data/SrV/' + city + '/SrV2018_Einzeldaten_' + city + '_SciUse_W2018.csv'

    sH=pd.read_csv(fn_hh,encoding='latin_1',sep=';',dtype={'PLZ':str,'GEWICHT_HH':str})
    sH.dropna(subset=['HHNR'],inplace=True)
    sP=pd.read_csv(fn_p,encoding='latin_1',sep=';')
    sW=pd.read_csv(fn_t,encoding='latin_1',sep=';',dtype={'V_START_PLZ':str,'V_ZIEL_PLZ':str})

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

    # change decimal from , to . and convert those variables from string to float
    sH.loc[:,'HH_Weight']=sH.loc[:,'HH_Weight'].map(lambda x: x.replace(',','.')).astype('float')
    sP.loc[:,'Per_Weight']=sP.loc[:,'Per_Weight'].map(lambda x: x.replace(',','.')).astype('float')
    sW.loc[:,'Trip_Weight']=sW.loc[:,'Trip_Weight'].map(lambda x: x.replace(',','.')).astype('float')
    sW.loc[:,'Trip_Distance']=sW.loc[:,'Trip_Distance'].map(lambda x: x.replace(',','.')).astype('float')
    sW.loc[:,'Trip_Distance_GIS']=sW.loc[:,'Trip_Distance_GIS'].map(lambda x: x.replace(',','.')).astype('float')

    # define unique person and trip numbers
    sP['HH_PNR']=sP['HHNR'].astype('str')+'_'+sP['Person'].astype('str')
    sW['HH_PNR']=sW['HHNR'].astype('str')+'_'+sW['Person'].astype('str')
    sW['HH_P_WNR']=sW['HH_PNR']+'_'+sW['Trip'].astype('str')

    # bring HHNR and geo_unit to the left side of the HH df
    cols=sH.columns.tolist()
    cols_new = ['HHNR', 'Postcode'] + [value for value in cols if value not in {'HHNR', 'Postcode'}]
    sH=sH[cols_new]

    # bring HHNR, HH_PNR, to the left side of the Per df
    cols=sP.columns.tolist()
    cols_new = ['HHNR','HH_PNR'] + [value for value in cols if value not in {'HHNR', 'HH_PNR'}]
    sP=sP[cols_new]

    # bring HHNR, HH_PNR, HH_P_WNR to the left side of the W df
    cols=sW.columns.tolist()
    cols_new = ['HHNR','HH_PNR','HH_P_WNR'] + [value for value in cols if value not in {'HHNR', 'HH_PNR','HH_P_WNR'}]
    sW=sW[cols_new]

    # Define time to transit in the household file
    sH['Time2Transit']=47
    # in case answer is NA (-8 or -10 or 0) because e.g. trams don't exist in a city, convert to 99.
    sH.loc[sH['Time2Bus']<1,'Time2Bus']=99
    sH.loc[sH['Time2Tram']<1,'Time2Tram']=99
    sH.loc[sH['Time2SBahn']<1,'Time2SBahn']=99
    sH.loc[sH['Time2UBahn']<1,'Time2UBahn']=99
    sH.loc[sH['Time2Train']<1,'Time2Train']=99

    for idx, row in sH.iterrows():
        sH.loc[idx,'Time2Transit']=sH.loc[idx,['Time2Bus','Time2Tram','Time2SBahn','Time2UBahn','Time2Train']].min()
    # then replace any 99s with nan
    sH.loc[:,'Time2Transit'].replace(99,np.nan,inplace=True)

    # define Month and Season in the trip file
    sW['Month']=sW.loc[:,'Date'].str[3:5]
    sW['Season']='Winter'
    sW.loc[sW['Month'].isin(['03','04','05']),'Season']='Spring'
    sW.loc[sW['Month'].isin(['06','07','08']),'Season']='Summer'
    sW.loc[sW['Month'].isin(['09','10','11']),'Season']='Autumn'
    sW['Season'].value_counts()

    # Define hour of trip starting time
    # sW.loc[:,'Hour+1']=0
    # sW.loc[sW['Minute']>30,'Hour+1']=1
    #sW['Hour']=sW.loc[:,'Time']+sW.loc[:,'Hour+1']
    sW['Hour']=sW.loc[:,'Time']

    # Define trip time categories
    sW['Trip_Time']='Nighttime Off-Peak'
    sW.loc[sW['Hour'].isin([6,7,8,9]),'Trip_Time']='AM_Rush'
    sW.loc[sW['Hour'].isin([12,13]),'Trip_Time']='Lunch'
    sW.loc[sW['Hour'].isin([16,17,18]),'Trip_Time']='PM Rush'
    sW.loc[sW['Hour'].isin([19,20,21]),'Trip_Time']='Evening'
    sW.loc[sW['Hour'].isin([10,11,14,15]),'Trip_Time']='Daytime Off-Peak'
    sW['trip_type_all']=sW['Ori_Reason']+'-'+sW['Des_Reason']

    # remove negatives in the n_accomp and n_others variables
    sW.loc[sW['N_accomp_HH']<0,'N_accomp_HH']=0
    sW.loc[sW['N_others_Car']<0,'N_others_Car']=0


    # address inconsistencies with the Person 'Education' variable, arising from respondents who have completed a certain level of education/training responding no dimploma yet, even though they have lower diplomas than the one they are currently studying for
    # these assumptions are based on German law, in which it is mandatory to go to school until age 16 (end of secondary school), so anyone in an occupation post-16 is very likely to have some education. https://www.studying-in-germany.org/german-education-system/

    sP.loc[(sP['Age']>9) & (sP['Age']<19) & (sP['Education']=='No diploma yet'),'Education']="Elementary" # if aged between 10 and 18, assume at least an Elementary education
    sP.loc[(sP['Age']>18) & (sP['Age']<22) & (sP['Education']=='No diploma yet'),'Education']="Secondary" # if aged 19 and 21, assume at least a Secondary education.

    # address the NA values for Education and Occupation for children under 6
    sP.loc[sP['Age']<7,['Education','Occupation']]='Pre-School'

    #sP.drop(columns=['HHNR'],inplace=True)
    sW.drop(columns=['HHNR','Person', 'Ori_Reason', 'Des_Reason','Time'],inplace=True)
    sW=sW.loc[sW['Trip_Valid']==1,:]
    # merge together the household, person, and trip files
    sHP=sH.merge(sP,on='HHNR')
    sHPW=sHP.merge(sW,on='HH_PNR')


    cols=sHPW.columns.tolist()
    cols_new = ['HHNR','HH_PNR','HH_P_WNR','Postcode','Ori_Plz','Des_Plz','Trip_Time','Trip_Purpose'] + [value for value in cols if value not in {'HHNR','HH_PNR','HH_P_WNR','Postcode','Ori_Plz','Des_Plz','Trip_Time','Trip_Purpose','Mode', 'Trip_Distance'}] +['Mode', 'Trip_Distance']
    sHPW=sHPW[cols_new]

    sHPW=sHPW.loc[sHPW['Ori_Plz'].isin(city_plz[city]),:]
    sHPW.sort_values(by='HH_P_WNR',inplace=True)
    sHPW.reset_index(drop=True,inplace=True)

    sHPW['Trip_Distance_Calc']=sHPW.loc[:,'Trip_Distance_GIS']
    sHPW.loc[sHPW['Trip_Distance_GIS_valid']==0,'Trip_Distance_Calc']=sHPW.loc[sHPW['Trip_Distance_GIS_valid']==0,'Trip_Distance']

    # sHPW['Trip_Distance']=sHPW.loc[:,'Trip_Distance']*1000
    sHPW['Trip_Distance']=sHPW.loc[:,'Trip_Distance_Calc']*1000
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
    print(str(round(0.001*person_mode_trips.groupby('Mode')['Trip_Distance'].sum()/len(person_trips),1)))

    if len(sHPW.loc[:,('HH_PNR','Day')].drop_duplicates())!=len(person_trips):
        print('NB! Some respondents report trips over more than one day')

    sHPW['Trip_Speed']=round(60*0.001*sHPW['Trip_Distance']/(sHPW['Trip_Duration']),2)

    sHPW.to_csv('../outputs/Combined/'+city+'.csv',index=False)

    # load in UF stats
    # population density
    pop_dens=pd.read_csv('../outputs/density_geounits/' + city + '_pop_density.csv',dtype={'geo_unit':str})
    pop_dens.drop(columns=['geometry','note','population','area'],inplace=True)
    # building density and distance to city center
    bld_dens=pd.read_csv('../outputs/CenterSubcenter/' + city + '_dist.csv',dtype={'plz':str})
    # connectivity stats
    conn=pd.read_csv('../outputs/Connectivity/connectivity_stats_' + city + '.csv',dtype={'plz':str})
    # decide which connectivity stats we want to keep
    conn=conn.loc[:,('plz','k_avg','clean_intersection_density_km','street_density_km','streets_per_node_avg','street_length_avg')]
    conn_vars=conn.drop(columns='plz').columns
    # land-use
    lu=pd.read_csv('../outputs/LU/UA_' + city + '.csv',dtype={'plz':str})
    # decide which lu varibales to use
    lu=lu.loc[:,('plz','pc_urb_fabric','pc_comm','pc_road','pc_urban')]

    # now merge all urban form features with the survey data.

    # population density origin
    sHPW_UF=sHPW.merge(pop_dens,left_on='Ori_Plz',right_on='geo_unit').copy()
    sHPW_UF.drop(columns='geo_unit',inplace=True)
    sHPW_UF.rename(columns={'Density':'PopDensity_origin'},inplace=True)
    # population density destination
    #sHPW_UF=sHPW_UF.merge(pop_dens,left_on='Des_Plz',right_on='geo_unit',how='left').copy() # allow for nans in destination data, see if/how model deals with them
    sHPW_UF=sHPW_UF.merge(pop_dens,left_on='Des_Plz',right_on='geo_unit').copy() # allow for nans in destination data, see if/how model deals with them
    sHPW_UF.drop(columns='geo_unit',inplace=True)
    sHPW_UF.rename(columns={'Density':'PopDensity_dest'},inplace=True)

    # buidling density and distance to centers origin
    sHPW_UF=sHPW_UF.merge(bld_dens,left_on='Ori_Plz',right_on='plz').copy() 
    sHPW_UF.drop(columns='plz',inplace=True)
    sHPW_UF.rename(columns={'minDist_subcenter':'DistSubcenter_origin','Distance2Center':'DistCenter_origin','build_vol_density':'BuildDensity_origin'},inplace=True)
    # buidling density and distance to centers destination
    sHPW_UF=sHPW_UF.merge(bld_dens,left_on='Des_Plz',right_on='plz').copy() # allow for nans in destination data, see if/how model deals with them
    sHPW_UF.drop(columns='plz',inplace=True)
    sHPW_UF.rename(columns={'minDist_subcenter':'DistSubcenter_dest','Distance2Center':'DistCenter_dest','build_vol_density':'BuildDensity_dest'},inplace=True)

    # connectivity stats, origin
    sHPW_UF=sHPW_UF.merge(conn,left_on='Ori_Plz',right_on='plz').copy() 
    sHPW_UF.drop(columns='plz',inplace=True)
    sHPW_UF.rename(columns={'k_avg':'K_avg_origin','clean_intersection_density_km':'IntersecDensity_origin','street_density_km':'StreetDensity_origin',
    'streets_per_node_avg':'StreetsPerNode_origin','street_length_avg':'StreetLength_origin'},inplace=True)

    # connectivity stats, destination
    sHPW_UF=sHPW_UF.merge(conn,left_on='Des_Plz',right_on='plz').copy() 
    sHPW_UF.drop(columns='plz',inplace=True)
    sHPW_UF.rename(columns={'k_avg':'K_avg_dest','clean_intersection_density_km':'IntersecDensity_dest','street_density_km':'StreetDensity_dest',
    'streets_per_node_avg':'StreetsPerNode_dest','street_length_avg':'StreetLength_dest'},inplace=True)

    # land-use stats, origin
    sHPW_UF=sHPW_UF.merge(lu,left_on='Ori_Plz',right_on='plz').copy() 
    sHPW_UF.drop(columns='plz',inplace=True)
    sHPW_UF.rename(columns={'pc_urb_fabric':'LU_UrbFab_origin','pc_comm':'LU_Comm_origin','pc_road':'LU_Road_origin',
    'pc_urban':'LU_Urban_origin'},inplace=True)

    # land-use stats, destination
    sHPW_UF=sHPW_UF.merge(lu,left_on='Des_Plz',right_on='plz').copy() 
    sHPW_UF.drop(columns='plz',inplace=True)
    sHPW_UF.rename(columns={'pc_urb_fabric':'LU_UrbFab_dest','pc_comm':'LU_Comm_dest','pc_road':'LU_Road_dest',
    'pc_urban':'LU_Urban_dest'},inplace=True)

    # mean time to transit
    mean_time_tran=pd.DataFrame(sHPW[['Postcode', 'HHNR','HH_Weight','Time2Transit']].drop_duplicates().groupby(['Postcode'])['Time2Transit'].sum()/sHPW[['Postcode', 'HHNR','HH_Weight','Time2Transit']].drop_duplicates().groupby(['Postcode'])['HH_Weight'].sum()).reset_index()
    mean_time_tran.rename(columns={0:'MeanTime2Transit'},inplace=True)

    # mean time to transit, origin
    sHPW_UF=sHPW_UF.merge(mean_time_tran,left_on='Ori_Plz',right_on='Postcode').copy() 
    sHPW_UF.drop(columns={'Postcode_y',},inplace=True)
    sHPW_UF.rename(columns={'MeanTime2Transit':'MeanTime2Transit_origin','Postcode_x':'Postcode'},inplace=True)

    # mean time to transit, destination
    sHPW_UF=sHPW_UF.merge(mean_time_tran,left_on='Des_Plz',right_on='Postcode').copy() 
    sHPW_UF.drop(columns={'Postcode_y',},inplace=True)
    sHPW_UF.rename(columns={'MeanTime2Transit':'MeanTime2Transit_dest','Postcode_x':'Postcode'},inplace=True)

    sHPW_UF.to_csv('../outputs/Combined/'+city+'_UF.csv',index=False)


    # now create and save some summary stats by postcode ######
    sHPW['Trip_Distance_Weighted']=sHPW['Trip_Distance']*sHPW['Trip_Weight']

    # mode share of trip distance by originating postcode
    ms_dist_all=pd.DataFrame(sHPW.groupby(['Ori_Plz'])['Trip_Distance_Weighted'].sum())
    ms_dist_all.reset_index(inplace=True)
    ms_dist_all.rename(columns={'Trip_Distance_Weighted':'Trip_Distance_All'},inplace=True)

    ms_dist=pd.DataFrame(sHPW.groupby(['Ori_Plz','Mode'])['Trip_Distance_Weighted'].sum())
    ms_dist.reset_index(inplace=True)
    ms_dist=ms_dist.merge(ms_dist_all)
    ms_dist['Share_Distance']=ms_dist['Trip_Distance_Weighted']/ms_dist['Trip_Distance_All']
    ms_dist.drop(columns=['Trip_Distance_Weighted','Trip_Distance_All'],inplace=True)

    # convert from long to wide 
    msdp=ms_dist.pivot(index='Ori_Plz',columns=['Mode'])
    msdp.columns = ['_'.join(col) for col in msdp.columns.values]
    msdp.reset_index(inplace=True)
    msdp.to_csv('../outputs/Summary_geounits/' + city + '_modeshare_origin.csv',index=False)

    # mode share of trip distance by residential postcode
    ms_dist_all_res=pd.DataFrame(sHPW.groupby(['Postcode'])['Trip_Distance_Weighted'].sum())
    ms_dist_all_res.reset_index(inplace=True)
    ms_dist_all_res.rename(columns={'Trip_Distance_Weighted':'Trip_Distance_All'},inplace=True)

    ms_dist_res=pd.DataFrame(sHPW.groupby(['Postcode','Mode'])['Trip_Distance_Weighted'].sum())
    ms_dist_res.reset_index(inplace=True)
    ms_dist_res=ms_dist_res.merge(ms_dist_all_res)
    ms_dist_res['Share_Distance']=ms_dist_res['Trip_Distance_Weighted']/ms_dist_res['Trip_Distance_All']
    ms_dist_res.drop(columns=['Trip_Distance_Weighted','Trip_Distance_All'],inplace=True)

    # convert long to wide
    msdrp=ms_dist_res.pivot(index='Postcode',columns=['Mode'])
    msdrp.columns = ['_'.join(col) for col in msdrp.columns.values]
    msdrp.reset_index(inplace=True)

    # avg travel km/day/cap by postcode of residence
    # first calculate weighted number of surveyed people by postcode
    plz_per_weight=pd.DataFrame(sHPW[['Postcode', 'HH_PNR','Per_Weight']].drop_duplicates().groupby(['Postcode'])['Per_Weight'].sum()).reset_index() 
    # next calculate sum weighted travel distance by residence postcode
    plz_dist_weight=pd.DataFrame(sHPW.groupby(['Postcode'])['Trip_Distance_Weighted'].sum()).reset_index()
    plz_dist_weight=plz_dist_weight.merge(plz_per_weight)
    # then divide
    plz_dist_weight['Daily_Distance_Person']=0.001*round(plz_dist_weight['Trip_Distance_Weighted']/plz_dist_weight['Per_Weight'])
    plz_dist_weight.drop(columns=['Trip_Distance_Weighted','Per_Weight'],inplace=True)
    # and car travel distance by residence postcode
    plz_dist_car=pd.DataFrame(sHPW.loc[sHPW['Mode']=='Car',:].groupby(['Postcode'])['Trip_Distance_Weighted'].sum()).reset_index()
    plz_dist_car=plz_dist_car.merge(plz_per_weight)
    plz_dist_car['Daily_Distance_Person_Car']=0.001*round(plz_dist_car['Trip_Distance_Weighted']/plz_dist_car['Per_Weight'])
    plz_dist_car.drop(columns=['Trip_Distance_Weighted','Per_Weight'],inplace=True)

    # car ownhership rates by household
    plz_hh_weight=pd.DataFrame(sHPW[['Postcode', 'HHNR','HH_Weight']].drop_duplicates().groupby(['Postcode'])['HH_Weight'].sum()).reset_index() 
    plz_hh_car=pd.DataFrame(sHPW.loc[sHPW['CarAvailable']==1,('Postcode', 'HHNR','HH_Weight')].drop_duplicates().groupby(['Postcode'])['HH_Weight'].sum()).reset_index() 
    plz_hh_car.rename(columns={'HH_Weight':'HH_WithCar'},inplace=True)
    plz_hh_car=plz_hh_car.merge(plz_hh_weight)
    plz_hh_car['CarOwnership_HH']=round(plz_hh_car['HH_WithCar']/plz_hh_car['HH_Weight'],3)
    plz_hh_car.drop(columns=['HH_WithCar','HH_Weight'],inplace=True)

    # and mean time to transite by postcode


    summary=msdrp.merge(plz_dist_weight,how='left')
    summary=summary.merge(plz_dist_car,how='left')
    summary=summary.merge(plz_hh_car,how='left')
    summary=summary.merge(mean_time_tran,how='left')
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

    # prepare to plot car ownership and distance traveled by income
    sHPW['Income']=np.nan
    sHPW.loc[sHPW['IncomeDetailed']=='Under500','Income']=350
    sHPW.loc[sHPW['IncomeDetailed']=='500-900','Income']=700
    sHPW.loc[sHPW['IncomeDetailed']=='900-1500','Income']=1200
    sHPW.loc[sHPW['IncomeDetailed']=='1500-2000','Income']=1750
    sHPW.loc[sHPW['IncomeDetailed']=='2000-2600','Income']=2300
    sHPW.loc[sHPW['IncomeDetailed']=='2600-3000','Income']=2800
    sHPW.loc[sHPW['IncomeDetailed']=='3000-3600','Income']=3300
    sHPW.loc[sHPW['IncomeDetailed']=='3600-4600','Income']=4000
    sHPW.loc[sHPW['IncomeDetailed']=='4600-5600','Income']=5000
    sHPW.loc[sHPW['IncomeDetailed']=='Over5600','Income']=6000

    # plot car ownership by income 
    hh_inc_car=pd.DataFrame(sHPW.groupby(['HHNR'])['Income','CarAvailable'].mean()).reset_index()
    inc_car=hh_inc_car.groupby('Income')['CarAvailable'].mean().reset_index()
    fig, ax = plt.subplots()
    plt.plot(inc_car['Income'],100*inc_car['CarAvailable'])
    plt.title('Car Ownership by Income, '+city,fontsize=16)
    plt.xlabel('HH Income, EUR/month',fontsize=12)
    plt.ylabel('HH Car Ownership Rate (%)',fontsize=12)
    ax.grid()
    ax.set_ylim(0,100)
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
    ax.set_ylim(10,100) 
    fig.savefig('../figures/plots/'+ city+'_CarModeShare.png',facecolor='w')

    inc_stats=inc_car.merge(person_mode_income)
    inc_stats['City']=city
    inc_stats_all=pd.concat([inc_stats_all,inc_stats])   

    # make dist categories
    sHPW['Trip_Distance_Cat']=np.nan
    sHPW.loc[sHPW['Trip_Distance_Calc']<1,'Trip_Distance_Cat']='0-1'
    sHPW.loc[(sHPW['Trip_Distance_Calc']>=1)&(sHPW['Trip_Distance_Calc']<2),'Trip_Distance_Cat']='1-2'
    sHPW.loc[(sHPW['Trip_Distance_Calc']>=2)&(sHPW['Trip_Distance_Calc']<4),'Trip_Distance_Cat']='2-4'
    sHPW.loc[(sHPW['Trip_Distance_Calc']>=4)&(sHPW['Trip_Distance_Calc']<8),'Trip_Distance_Cat']='4-8'
    sHPW.loc[(sHPW['Trip_Distance_Calc']>=8),'Trip_Distance_Cat']='8+'

    # plot trip mode by distance
    mode_dist=sHPW.loc[:,('Mode','Trip_Distance_Cat','Trip_Purpose')].groupby(['Mode','Trip_Distance_Cat']).count().unstack('Mode').reset_index()
    mode_dist.columns = ['_'.join(col) for col in mode_dist.columns.values]
    mode_dist.reset_index(inplace=True,drop=True)
    mode_dist.columns=[col.replace('Trip_Purpose_','') for col in mode_dist.columns]

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

    # calcuate trip speeds and summarize by mode and distance
    # sHPW['Trip_Speed']=round(60*0.001*sHPW['Trip_Distance']/(sHPW['Trip_Duration']),2)
    # sHPW.replace([np.inf, -np.inf], np.nan, inplace=True)
    # print('Average speed by mode and distance category: ' + city)
    # print(round(sHPW.dropna(axis=0,subset='Trip_Speed').groupby(['Trip_Distance_Cat','Mode'])['Trip_Speed'].mean(),2))

#cities=pd.Series(['Berlin','Dresden','Düsseldorf','Frankfurt am Main','Kassel','Leipzig','Magdeburg','Potsdam'])
cities=pd.Series(['Berlin','Dresden','Leipzig','Magdeburg','Potsdam']) # currently only cities with complete UF data
cities.apply(combine_survey_data) # args refers to the size threshold above which to divide large units into their smaller sub-components, e.g. 10km2
inc_stats_all.to_csv('../figures/plots/income_stats_DE.csv',index=False)