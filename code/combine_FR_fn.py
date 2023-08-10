# # script to combine and harmonize survey data for French cities, and calculate some summary statistics
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import trunc

# inputs:
# French city dictionary pickle files, to create harmonised variables from the raw survey data
# French city survey data files (houeshold, population, trips), excluding Paris
# population density by spatial unit for French cities
# other pre-calculated urban form features (building density, distance to center, land-use data, connectivity metrics) by spatial unit for each city

#
inc_stats_all=pd.DataFrame(columns = ['Income', 'CarAvailable', 'Trip_Distance', 'Trip_Distance_Car','Car_ModeShare', 'City'])

def combine_survey_data(city):
    global inc_stats_all
    print(city)
    if city == 'Clermont':
        fn_hh='../../MSCA_data/FranceRQ/lil-0924_Clermont.csv/Csv/Fichiers_Standard_Face_a_face/clermontfer_2012_std_faf_men.csv'
        fn_p='../../MSCA_data/FranceRQ/lil-0924_Clermont.csv/Csv/Fichiers_Standard_Face_a_face/clermontfer_2012_std_faf_pers.csv'
        fn_t='../../MSCA_data/FranceRQ/lil-0924_Clermont.csv/Csv/Fichiers_Standard_Face_a_face/clermontfer_2012_std_faf_depl.csv'
        fn_hh0='../../MSCA_data/FranceRQ/lil-0924_Clermont.csv/Csv/Fichiers_Original_Face_a_face/clermontfer_2012_ori_faf_men.csv'

    if city == 'Toulouse':
        fn_hh='../../MSCA_data/FranceRQ/lil-0933_Toulouse.csv/Csv/Fichiers_Standard/toulouse_2013_std_men.csv'
        fn_p='../../MSCA_data/FranceRQ/lil-0933_Toulouse.csv/Csv/Fichiers_Standard/toulouse_2013_std_pers.csv'
        fn_t='../../MSCA_data/FranceRQ/lil-0933_Toulouse.csv/Csv/Fichiers_Standard/toulouse_2013_std_depl.csv'
        fn_hh0='../../MSCA_data/FranceRQ/lil-0933_Toulouse.csv/Csv/Fichiers_Original/toulouse_2013_ori_men.csv'

    if city == 'Montpellier':
        fn_hh='../../MSCA_data/FranceRQ/lil-0937_Montpellier.csv/Csv/Fichiers_Standard_Face_a_face/montpellier_2014_std_faf_men.csv'
        fn_p='../../MSCA_data/FranceRQ/lil-0937_Montpellier.csv/Csv/Fichiers_Standard_Face_a_face/montpellier_2014_std_faf_pers.csv'
        fn_t='../../MSCA_data/FranceRQ/lil-0937_Montpellier.csv/Csv/Fichiers_Standard_Face_a_face/montpellier_2014_std_faf_depl.csv'

    if city == 'Lyon':
        fn_hh='../../MSCA_data/FranceRQ/lil-1023_Lyon.csv/Csv/Fichiers_Standard_Face_a_face/lyon_2015_std_faf_men.csv'
        fn_p='../../MSCA_data/FranceRQ/lil-1023_Lyon.csv/Csv/Fichiers_Standard_Face_a_face/lyon_2015_std_faf_pers.csv'
        fn_t='../../MSCA_data/FranceRQ/lil-1023_Lyon.csv/Csv/Fichiers_Standard_Face_a_face/lyon_2015_std_faf_depl.csv'

    if city == 'Nantes':
        fn_hh='../../MSCA_data/FranceRQ/lil-1024_Nantes.csv/Csv/Fichiers_Standard_Face_a_face/nantes_2015_std_faf_men.csv'
        fn_p='../../MSCA_data/FranceRQ/lil-1024_Nantes.csv/Csv/Fichiers_Standard_Face_a_face/nantes_2015_std_faf_pers.csv'
        fn_t='../../MSCA_data/FranceRQ/lil-1024_Nantes.csv/Csv/Fichiers_Standard_Face_a_face/nantes_2015_std_faf_depl.csv'

    if city == 'Nimes':
        fn_hh='../../MSCA_data/FranceRQ/lil-1135_Nimes.csv/Csv/Fichiers_Standard/nimes_2015_std_men.csv'
        fn_p='../../MSCA_data/FranceRQ/lil-1135_Nimes.csv/Csv/Fichiers_Standard/nimes_2015_std_pers.csv'
        fn_t='../../MSCA_data/FranceRQ/lil-1135_Nimes.csv/Csv/Fichiers_Standard/nimes_2015_std_depl.csv'

    if city == 'Lille':
        fn_hh='../../MSCA_data/FranceRQ/lil-1152_Lille.csv/Csv/Fichiers_Standard/lille_2016_std_men.csv'
        fn_p='../../MSCA_data/FranceRQ/lil-1152_Lille.csv/Csv/Fichiers_Standard/lille_2016_std_pers.csv'
        fn_t='../../MSCA_data/FranceRQ/lil-1152_Lille.csv/Csv/Fichiers_Standard/lille_2016_std_depl.csv'

    if city == 'Dijon':
        fn_hh='../../MSCA_data/FranceRQ/lil-1214_Dijon.csv/Csv/Fichiers_Standard_Face_a_face/dijon_2016_std_faf_men.csv'
        fn_p='../../MSCA_data/FranceRQ/lil-1214_Dijon.csv/Csv/Fichiers_Standard_Face_a_face/dijon_2016_std_faf_pers.csv'
        fn_t='../../MSCA_data/FranceRQ/lil-1214_Dijon.csv/Csv/Fichiers_Standard_Face_a_face/dijon_2016_std_faf_depl.csv'

    sH=pd.read_csv(fn_hh,sep=';')
    sP=pd.read_csv(fn_p,sep=';')
    sW=pd.read_csv(fn_t,sep=';')

    # load pickled dictionary files
    with open('../dictionaries/' + city + '_var.pkl','rb') as f:
        var_all = pickle.load(f)

    with open('../dictionaries/' + city + '_val.pkl','rb') as f:
        value_all = pickle.load(f)

    with open('../dictionaries/' + city + '_na.pkl','rb') as f:
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

    # define household id !! this is city specific !!
    # sH0['HHNR']=sH0['geo_unit'].astype('str')+'_'+sH0['Sample'].astype('str')

    # define, sector, zone, and household id !! this is city specific !!

    sH['geo_unit'] = sH['Sector_Zone'].apply(lambda y: trunc(0.001*y))
    sH['Zone'] = sH['Sector_Zone'].apply(lambda y: y % 1000)
    sH['HHNR']=sH['geo_unit'].astype('str')+'_'+sH['Sample'].astype('str')

    # define, sector, zone, household, and person id !! this is city specific !!
    sP['geo_unit'] = sP['Sector_Zone'].apply(lambda y: trunc(0.001*y))
    sP['Zone'] = sP['Sector_Zone'].apply(lambda y: y % 1000)
    sP['HHNR']=sP['geo_unit'].astype('str')+'_'+sP['Sample'].astype('str')
    sP['HH_PNR']=sP['HHNR']+'_'+sP['Person'].astype('str')

    # define, sector, zone, household, and person, and trip id !! this is city specific !!
    sW['geo_unit'] = sW['Sector_Zone'].apply(lambda y: trunc(0.001*y))
    sW['Zone'] = sW['Sector_Zone'].apply(lambda y: y % 1000)
    sW['HHNR']=sW['geo_unit'].astype('str')+'_'+sW['Sample'].astype('str')
    sW['HH_PNR']=sW['HHNR']+'_'+sW['Person'].astype('str')
    sW['HH_P_WNR']=sW['HH_PNR']+'_'+sW['Trip'].astype('str')

    if len(sH['HHNR'].unique())!=len(sH):
        print('Unique HHNR not found with sector + sample. Defining HHNR instead with sector_zone + sample.')
        sH['HHNR']=sH['Sector_Zone'].astype('str').str.zfill(6)+'_'+sH['Sample'].astype('str')
        
        sP['HHNR']=sP['Sector_Zone'].astype('str').str.zfill(6)+'_'+sP['Sample'].astype('str')
        sP['HH_PNR']=sP['HHNR']+'_'+sP['Person'].astype('str')

        sW['HHNR']=sW['Sector_Zone'].astype('str').str.zfill(6)+'_'+sW['Sample'].astype('str')
        sW['HH_PNR']=sW['HHNR']+'_'+sW['Person'].astype('str')
        sW['HH_P_WNR']=sW['HH_PNR']+'_'+sW['Trip'].astype('str')

        if len(sH['HHNR'].unique())!=len(sH):
            print('Unique HHNR still not found with sector_zone + sample.')

    # merge the household income variable into the household file !! this is only for some cities !!
    if 'fn_hh0' in locals():
        sH0=pd.read_csv(fn_hh0,sep=';')

        for k in var_all['HH0'].keys():
            if len(value_all['HH0'][k])>0:
                sH0[k]=sH0[var_all['HH0'][k]].map(value_all['HH0'][k])
            elif len(value_all['HH0'][k])==0:
                sH0[k]=sH0[var_all['HH0'][k]]
        sH0=sH0[list(value_all['HH0'].keys())]
        sH0['HHNR']=sH0['geo_unit'].astype('str')+'_'+sH0['Sample'].astype('str')
        sH0.drop(columns=['Sample','geo_unit'],inplace=True)
        sH=sH.merge(sH0,on='HHNR')

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

    # calculate household size from the sP data !! This is city specific !! at least in other surevys it is included as it's own variable
    hhs=sP['HHNR'].value_counts().reset_index()
    hhs.rename(columns={'index':'HHNR','HHNR':'HHSize'},inplace=True)
    sH=sH.merge(hhs,on='HHNR')

    # address inconsistencies with the Person 'Education' variable, arising from respondents who have completed a certain level of education/training responding no dimploma yet, even though they have lower diplomas than the one they are currently studying for
    # these assumptions are based on French law, in which it is mandatory to go to school until age 16 (end of secondary school), so anyone in an occupation post-16 is very likely to have some education. https://www.expatica.com/fr/education/children-education/french-education-system-101147/
    # they may need defined differently for non-French surveys
    sP.loc[(sP['Age']>11) & (sP['Age']<17) & (sP['Education']=='No diploma yet'),'Education']="Elementary" # if aged between 12 and 16, assume at least an Elementary education
    sP.loc[(sP['Age']>16) & (sP['Age']<20) & (sP['Education']=='No diploma yet'),'Education']="Secondary" # if aged between 17 and 19, assume at least a Secondary education.
    sP.loc[(sP['Age']>15) & (sP['Occupation']=='Student_3rdLevel') & (sP['Education']=='No diploma yet'),'Education']="Secondary+BAC" # if a 3rd level student, assume at least Secondary eduction with BAC
    sP.loc[(sP['Age']>15) & (sP['Occupation']=='Trainee') & (sP['Education']=='No diploma yet'),'Education']="Secondary" # If a trainee, assume at least a secondary education

    # address the NA values for Education and Occupation for children under 5
    sP.loc[sP['Age']<5,['Education','Occupation']]='Pre-School'

    # combine the two 'car parking available' variables 
    sP['Work/Study_CarParkAvailable']=0
    sP.loc[(sP['Work/Study_CarParkAvailable1']==1) | (sP['Work/Study_CarParkAvailable2']==1),'Work/Study_CarParkAvailable']=1
    sP.drop(columns=['Sector_Zone','Zone','Sample','geo_unit','Work/Study_CarParkAvailable1','Work/Study_CarParkAvailable2'],inplace=True)

    # create the origin and destination id's based on what we extracted from the microdata !! this is city specific !!
    sW['Ori_geo_unit']=sW['Ori_Sec_Zone'].apply(lambda y: trunc(0.001*y))
    sW['Des_geo_unit']=sW['Des_Sec_Zone'].apply(lambda y: trunc(0.001*y))

    # combine the trip reasons for independent and accompanied travellers !! this is city specific !! 
    # This may not be achievable, as not all cities specify the origin and desination of the accompanied person, and it may be better then to simply classify the trip reason as accompanying/kids
    # sW.loc[sW['Ori_Reason1'].isin([61,62,63,64,71,72,73,74]),'Ori_Reason1']=sW['Ori_Reason2']
    # sW.loc[sW['Des_Reason1'].isin([61,62,63,64,71,72,73,74]),'Des_Reason1']=sW['Des_Reason2']

    # define the simplified origin and destination reasons. !! This is city specific !! 
    sW['Ori_Reason_Agg']='Other'
    sW.loc[sW['Ori_Reason1'].isin([1,2]),'Ori_Reason_Agg']='Home'
    sW.loc[sW['Ori_Reason1'].isin([11,12,13,14,81]),'Ori_Reason_Agg']='Work'
    # In Toulouse, 'Personal' refers to two categories: "Being looked after (childminder, crèche...)" and "Receiving care (health)"
    sW.loc[sW['Ori_Reason1'].isin([21,41]),'Ori_Reason_Agg']='Personal' # this was prev. classified as 'Care' but was converted to 'Personal' for ease of harmonization with other cities. 
    sW.loc[sW['Ori_Reason1'].isin([22,23,24,25,26,27,28,29,96,97]),'Ori_Reason_Agg']='School'
    sW.loc[sW['Ori_Reason1'].isin([30,31,32,33,34,35,82,98]),'Ori_Reason_Agg']='Shopping'
    sW.loc[sW['Ori_Reason1'].isin([51,52,53,54]),'Ori_Reason_Agg']='Leisure'
    sW.loc[sW['Ori_Reason1'].isin([61,62,63,64,71,72,73,74]),'Ori_Reason_Agg']='Accompanying/Kids'

    sW['Des_Reason_Agg']='Other'
    sW.loc[sW['Des_Reason1'].isin([1,2]),'Des_Reason_Agg']='Home'
    sW.loc[sW['Des_Reason1'].isin([11,12,13,14,81]),'Des_Reason_Agg']='Work'
    sW.loc[sW['Des_Reason1'].isin([21,41]),'Des_Reason_Agg']='Personal' # this was prev. classified as 'Care' but was converted to 'Personal' for ease of harmonization with other cities.
    sW.loc[sW['Des_Reason1'].isin([22,23,24,25,26,27,28,29,96,97]),'Des_Reason_Agg']='School'
    sW.loc[sW['Des_Reason1'].isin([30,31,32,33,34,35,82,98]),'Des_Reason_Agg']='Shopping'
    sW.loc[sW['Des_Reason1'].isin([51,52,53,54]),'Des_Reason_Agg']='Leisure'
    sW.loc[sW['Des_Reason1'].isin([61,62,63,64,71,72,73,74]),'Des_Reason_Agg']='Accompanying/Kids'

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

    # calculate the trip hour and time of day using the departure time or hour (whichever is given)
    sW['Hour']=round(sW['Time']/100) # + round((sW['Time'] % 100)/60) now just use the hours
    sW.loc[sW['Hour']>24,'Hour']=sW.loc[sW['Hour']>24,'Hour']-24
    # break the trip departure times into grouped times of day
    sW['Trip_Time']='Nighttime Off-Peak'
    sW.loc[sW['Hour'].isin([6,7,8,9]),'Trip_Time']='AM_Rush'
    sW.loc[sW['Hour'].isin([12,13]),'Trip_Time']='Lunch'
    sW.loc[sW['Hour'].isin([16,17,18]),'Trip_Time']='PM Rush'
    sW.loc[sW['Hour'].isin([19,20,21]),'Trip_Time']='Evening'
    sW.loc[sW['Hour'].isin([10,11,14,15]),'Trip_Time']='Daytime Off-Peak'

    sW.drop(columns=['Sector_Zone','Sample','Zone','geo_unit','HHNR','Person', 'Ori_Reason1','Ori_Reason2', 'Des_Reason1', 'Des_Reason2','Hour','Time'],inplace=True,errors='ignore')

    # merge together the household, person, and trip files
    sHP=sH.merge(sP,on='HHNR')
    sHPW=sHP.merge(sW,on='HH_PNR')

    # make a generic variable for parking available at work/study destinations
    sHPW['ParkingAvailable_Dest']=0
    sHPW.loc[(sHPW['Des_Reason_Agg'].isin(['Work','School'])) & (sHPW['Work/Study_CarParkAvailable']==1) ,'ParkingAvailable_Dest']=1
    sHPW.drop(columns=['Work/Study_CarParkAvailable'],inplace=True)

    # bring HHNR, HH_PNR, HH_P_WNR to the left side of the combined df
    cols=sHPW.columns.tolist()
    cols_new = ['HHNR','HH_PNR','HH_P_WNR','geo_unit','Ori_Sec_Zone','Des_Sec_Zone','Trip_Time','Trip_Purpose'] + [value for value in cols if value not in {'HHNR','HH_PNR','HH_P_WNR','geo_unit','Ori_Sec_Zone','Des_Sec_Zone','Trip_Time','Trip_Purpose','Mode', 'Trip_Distance'}] +['Mode', 'Trip_Distance']
    sHPW=sHPW[cols_new]

    # load in dictionary of mixed geocodes, which we will use for merging with the urban form features
    with open('../dictionaries/' + city + '_mixed_geocode.pkl','rb') as f:
        code_dict = pickle.load(f)

    # use the code dictionary to create the origin and destination geocodes
    sHPW['Ori_geocode']=sHPW['Ori_Sec_Zone'].astype("string").str.zfill(6).map(code_dict).astype("string") # need this format https://stackoverflow.com/questions/22231592/pandas-change-data-type-of-series-to-string
    sHPW['Des_geocode']=sHPW['Des_Sec_Zone'].astype("string").str.zfill(6).map(code_dict).astype("string")
    sHPW['Res_geocode']=sHPW['Sector_Zone'].astype("string").str.zfill(6).map(code_dict).astype("string") 

    # these are the highest resolution geo ids, and are used to merge with hi-res connectivity and distance urban form features
    sHPW['Ori_Sec_Zone']=sHPW['Ori_Sec_Zone'].astype("string").str.zfill(6)
    sHPW['Des_Sec_Zone']=sHPW['Des_Sec_Zone'].astype("string").str.zfill(6)
    sHPW['Res_Sec_Zone']=sHPW['Sector_Zone'].astype("string").str.zfill(6)
    sHPW.drop(columns='Sector_Zone',inplace=True)
    # remove trips that don't start within the region where we collect urban features data
    sHPW=sHPW.loc[sHPW['Ori_geocode'].isna()==False,:]

    # further cleaning to retrict the trips to those between 0.1km and 50km
    sHPW=sHPW.loc[(sHPW['Trip_Distance']>=50) & (sHPW['Trip_Distance']<=50000),:] 
    sHPW=sHPW.loc[sHPW['Mode']!='Other',:]

    weighted=sHPW.loc[:,('Per_Weight','Mode','Trip_Distance','Trip_Purpose_Agg')]
    weighted['Dist_Weighted_P']=weighted['Per_Weight']*weighted['Trip_Distance']

    unique_persons=sHPW.loc[:,['HH_PNR','Per_Weight']].drop_duplicates()

    weight_daily_travel=pd.DataFrame(0.001*weighted.groupby('Mode')['Dist_Weighted_P'].sum()/unique_persons['Per_Weight'].sum()).reset_index()
    commute_avg=0.001*weighted.loc[weighted['Trip_Purpose_Agg']=='Home↔Work','Dist_Weighted_P'].sum()/weighted.loc[weighted['Trip_Purpose_Agg']=='Home↔Work','Per_Weight'].sum()
    trip_avg=0.001*weighted['Dist_Weighted_P'].sum()/weighted['Per_Weight'].sum()
    weight_trip_avg=pd.DataFrame(data={'Mode':['All','Commute'],'Avg_trip_dist':[trip_avg,commute_avg]})

    weight_daily_travel.rename(columns={'Dist_Weighted_P':'Daily_Travel_cap'},inplace=True)
    weight_daily_travel['Mode_Share']=weight_daily_travel['Daily_Travel_cap']/weight_daily_travel['Daily_Travel_cap'].sum()

    carown=sHPW.loc[:,['HHNR','HH_Weight','CarOwnershipHH']].drop_duplicates()
    own=pd.DataFrame(data={'Mode':['Car'],'Ownership':sum(carown['CarOwnershipHH']*carown['HH_Weight'])/sum(carown['HH_Weight'])})
    weight_daily_travel=weight_daily_travel.merge(own,how='left')
    weight_daily_travel=weight_daily_travel.merge(weight_trip_avg,how='outer')
    weight_daily_travel.loc[weight_daily_travel['Mode']=='All','Daily_Travel_cap']=weight_daily_travel['Daily_Travel_cap'].sum()

    weight_daily_travel.to_csv('../outputs/summary_stats/'+city+'_stats.csv',index=False)

    if len(sHPW.loc[:,('HH_PNR','Day')].drop_duplicates())!=len(sHPW.loc[:,('HH_PNR')].drop_duplicates()):
        print('NB! Some respondents report trips over more than one day')

    # rename geo-unit to res_geo_unit
    sHPW['Res_geo_unit']=sHPW['geo_unit'].astype("string")
    sHPW.drop(columns='geo_unit',inplace=True)

    sHPW.to_csv('../outputs/Combined/'+city+'.csv',index=False)

    # load in UF stats
    # population density
    if city=='Paris':
        pop_dens=pd.read_csv('../outputs/density_geounits/' + city + '_pop_density_lowres.csv',dtype={'geo_unit':str})
        pop_dens.rename(columns={'geo_unit':'geocode'},inplace=True)
    else:
        pop_dens=pd.read_csv('../outputs/density_geounits/' + city + '_pop_density_mixres.csv',dtype={'geocode':str})
    pop_dens.drop(columns=['geometry','area','source'],inplace=True)
    # building density and distance to city center
    bld_dens=pd.read_csv('../outputs/CenterSubcenter/' + city + '_dist.csv',dtype={'geocode':str})
    bld_dens.drop(columns=['wgt_center','minDist_subcenter','Distance2Center'],inplace=True)
    d2=pd.read_csv('../outputs/CenterSubcenter/' + city + '_dist_hires.csv',dtype={'geocode':str})
    d2.drop(columns=['wgt_center','build_vol_density'],inplace=True)
    # connectivity stats, defined at high resolution
    conn=pd.read_csv('../outputs/Connectivity/connectivity_stats_' + city + '.csv',dtype={'geocode':str})
    # decide which connectivity stats we want to keep
    conn=conn.loc[:,('geocode','clean_intersection_density_km','street_length_avg','streets_per_node_avg','bike_lane_share')]
    conn['bike_lane_share']=conn['bike_lane_share'].fillna(0)

    # land-use
    lu=pd.read_csv('../outputs/LU/UA_' + city + '.csv',dtype={'geocode':str})
    # decide which lu varibales to use
    lu=lu.loc[:,('geocode','pc_urb_fabric','pc_comm','pc_road','pc_urban')]

    # now merge all urban form features with the survey data.
    # population density origin
    sHPW_UF=sHPW.merge(pop_dens,left_on='Ori_geocode',right_on='geocode').copy()
    sHPW_UF.drop(columns='geocode',inplace=True)
    sHPW_UF.rename(columns={'Density':'PopDensity_origin'},inplace=True)
    # population density residential
    sHPW_UF=sHPW_UF.merge(pop_dens,left_on='Res_geocode',right_on='geocode').copy()
    sHPW_UF.drop(columns='geocode',inplace=True)
    sHPW_UF.rename(columns={'Density':'PopDensity_res'},inplace=True)

    # buidling density origin
    sHPW_UF=sHPW_UF.merge(bld_dens,left_on='Ori_geocode',right_on='geocode').copy() 
    sHPW_UF.drop(columns='geocode',inplace=True)
    sHPW_UF.rename(columns={'build_vol_density':'BuildDensity_origin'},inplace=True)
    # buidling density residential
    sHPW_UF=sHPW_UF.merge(bld_dens,left_on='Res_geocode',right_on='geocode').copy() # allow for nans in destination data, see if/how model deals with them
    sHPW_UF.drop(columns='geocode',inplace=True)
    sHPW_UF.rename(columns={'build_vol_density':'BuildDensity_res'},inplace=True)

    # d2 origin
    sHPW_UF=sHPW_UF.merge(d2,left_on='Ori_Sec_Zone',right_on='geocode').copy() 
    sHPW_UF.drop(columns='geocode',inplace=True)
    sHPW_UF.rename(columns={'minDist_subcenter':'DistSubcenter_origin','Distance2Center':'DistCenter_origin'},inplace=True)
    # d2residential
    sHPW_UF=sHPW_UF.merge(d2,left_on='Res_Sec_Zone',right_on='geocode').copy() # allow for nans in destination data, see if/how model deals with them
    sHPW_UF.drop(columns='geocode',inplace=True)
    sHPW_UF.rename(columns={'minDist_subcenter':'DistSubcenter_res','Distance2Center':'DistCenter_res'},inplace=True)

    # # connectivity stats, origin
    # sHPW_UF=sHPW_UF.merge(conn,left_on='Ori_Sec_Zone',right_on='geocode').copy() 
    # sHPW_UF.drop(columns='geocode',inplace=True)
    # sHPW_UF.rename(columns={'k_avg':'K_avg_origin','clean_intersection_density_km':'IntersecDensity_origin','street_density_km':'StreetDensity_origin',
    # 'streets_per_node_avg':'StreetsPerNode_origin','street_length_avg':'StreetLength_origin'},inplace=True)

    # # connectivity stats, res
    # sHPW_UF=sHPW_UF.merge(conn,left_on='Res_Sec_Zone',right_on='geocode').copy() 
    # sHPW_UF.drop(columns='geocode',inplace=True)
    # sHPW_UF.rename(columns={'k_avg':'K_avg_dest','clean_intersection_density_km':'IntersecDensity_res','street_density_km':'StreetDensity_res',
    # 'streets_per_node_avg':'StreetsPerNode_res','street_length_avg':'StreetLength_res'},inplace=True)

    # connectivity stats, origin
    sHPW_UF=sHPW_UF.merge(conn,left_on='Ori_Sec_Zone',right_on='geocode').copy() 
    sHPW_UF.drop(columns='geocode',inplace=True)
    sHPW_UF.rename(columns={'clean_intersection_density_km':'IntersecDensity_origin',
                            'street_length_avg':'street_length_origin',
                            'streets_per_node_avg':'streets_per_node_origin',
                            'bike_lane_share':'bike_lane_share_origin'},inplace=True)

    # connectivity stats, residential
    sHPW_UF=sHPW_UF.merge(conn,left_on='Res_Sec_Zone',right_on='geocode').copy() 
    sHPW_UF.drop(columns='geocode',inplace=True)
    sHPW_UF.rename(columns={'clean_intersection_density_km':'IntersecDensity_res',
                            'street_length_avg':'street_length_res',
                            'streets_per_node_avg':'streets_per_node_res',
                            'bike_lane_share':'bike_lane_share_res'},inplace=True)
    
    # land-use stats, origin
    sHPW_UF=sHPW_UF.merge(lu,left_on='Ori_geocode',right_on='geocode').copy() 
    sHPW_UF.drop(columns='geocode',inplace=True)
    sHPW_UF.rename(columns={'pc_urb_fabric':'LU_UrbFab_origin','pc_comm':'LU_Comm_origin','pc_road':'LU_Road_origin',
    'pc_urban':'LU_Urban_origin'},inplace=True)
    # land-use stats, destination
    sHPW_UF=sHPW_UF.merge(lu,left_on='Des_geocode',right_on='geocode').copy() 
    sHPW_UF.drop(columns='geocode',inplace=True)
    sHPW_UF.rename(columns={'pc_urb_fabric':'LU_UrbFab_dest','pc_comm':'LU_Comm_dest','pc_road':'LU_Road_dest',
    'pc_urban':'LU_Urban_dest'},inplace=True)
    # land-use stats, res
    sHPW_UF=sHPW_UF.merge(lu,left_on='Res_geocode',right_on='geocode').copy() 
    sHPW_UF.drop(columns='geocode',inplace=True)
    sHPW_UF.rename(columns={'pc_urb_fabric':'LU_UrbFab_res','pc_comm':'LU_Comm_res','pc_road':'LU_Road_res',
    'pc_urban':'LU_Urban_res'},inplace=True)

    sHPW_UF['UrbPopDensity_origin']=sHPW_UF['PopDensity_origin']/sHPW_UF['LU_Urban_origin']
    sHPW_UF['UrbPopDensity_res']=sHPW_UF['PopDensity_res']/sHPW_UF['LU_Urban_res']

    sHPW_UF['UrbBuildDensity_origin']=sHPW_UF['BuildDensity_origin']/sHPW_UF['LU_Urban_origin']
    sHPW_UF['UrbBuildDensity_res']=sHPW_UF['BuildDensity_res']/sHPW_UF['LU_Urban_res']

    sHPW_UF.to_csv('../outputs/Combined/'+city+'_UF.csv',index=False)

    # now create and save some summary stats by postcode
    sHPW['Trip_Distance_Weighted']=sHPW['Trip_Distance']*sHPW['Per_Weight']

    # mode share of trip distance by geocode. later repeat to do the same by geounit
    ms_dist_all_res=pd.DataFrame(sHPW.groupby(['Res_geocode'])['Trip_Distance_Weighted'].sum())
    ms_dist_all_res.reset_index(inplace=True)
    ms_dist_all_res.rename(columns={'Trip_Distance_Weighted':'Trip_Distance_All'},inplace=True)

    ms_dist_res=pd.DataFrame(sHPW.groupby(['Res_geocode','Mode'])['Trip_Distance_Weighted'].sum())
    ms_dist_res.reset_index(inplace=True)
    ms_dist_res=ms_dist_res.merge(ms_dist_all_res)
    ms_dist_res['Share_Distance']=ms_dist_res['Trip_Distance_Weighted']/ms_dist_res['Trip_Distance_All']
    ms_dist_res.drop(columns=['Trip_Distance_Weighted','Trip_Distance_All'],inplace=True)

    # convert long to wide
    msdrp=ms_dist_res.pivot(index='Res_geocode',columns=['Mode'])
    msdrp.columns = ['_'.join(col) for col in msdrp.columns.values]
    msdrp.reset_index(inplace=True)

    # avg travel km/day/cap by postcode of residence
    # first calculate weighted number of surveyed people by postcode
    plz_per_weight=pd.DataFrame(sHPW[['Res_geocode', 'HH_PNR','Per_Weight']].drop_duplicates().groupby(['Res_geocode'])['Per_Weight'].sum()).reset_index() 
    # next calculate sum weighted travel distance by residence postcode
    plz_dist_weight=pd.DataFrame(sHPW.groupby(['Res_geocode'])['Trip_Distance_Weighted'].sum()).reset_index()
    plz_dist_weight=plz_dist_weight.merge(plz_per_weight)
    # then divide
    plz_dist_weight['Daily_Distance_Person']=0.001*round(plz_dist_weight['Trip_Distance_Weighted']/plz_dist_weight['Per_Weight'])
    plz_dist_weight.drop(columns=['Trip_Distance_Weighted','Per_Weight'],inplace=True)
    # and car travel distance by residence postcode
    plz_dist_car=pd.DataFrame(sHPW.loc[sHPW['Mode']=='Car',:].groupby(['Res_geocode'])['Trip_Distance_Weighted'].sum()).reset_index()
    plz_dist_car=plz_dist_car.merge(plz_per_weight)
    plz_dist_car['Daily_Distance_Person_Car']=0.001*round(plz_dist_car['Trip_Distance_Weighted']/plz_dist_car['Per_Weight'])
    plz_dist_car.drop(columns=['Trip_Distance_Weighted','Per_Weight'],inplace=True)

    # car ownership rates by household
    plz_hh_weight=pd.DataFrame(sHPW[['Res_geocode', 'HHNR','HH_Weight']].drop_duplicates().groupby(['Res_geocode'])['HH_Weight'].sum()).reset_index() 
    plz_hh_car=pd.DataFrame(sHPW.loc[sHPW['CarAvailable']==1,('Res_geocode', 'HHNR','HH_Weight')].drop_duplicates().groupby(['Res_geocode'])['HH_Weight'].sum()).reset_index() 
    plz_hh_car.rename(columns={'HH_Weight':'HH_WithCar'},inplace=True)
    plz_hh_car=plz_hh_car.merge(plz_hh_weight)
    plz_hh_car['CarOwnership_HH']=round(plz_hh_car['HH_WithCar']/plz_hh_car['HH_Weight'],3)
    plz_hh_car.drop(columns=['HH_WithCar','HH_Weight'],inplace=True)

    summary=msdrp.merge(plz_dist_weight)
    summary=summary.merge(plz_dist_car)
    summary=summary.merge(plz_hh_car)
    summary.to_csv('../outputs/summary_geounits/'+city+'.csv',index=False)

    # mode share of trip distance by GEOUNIT, i.e. low resolution
    ms_dist_all_res=pd.DataFrame(sHPW.groupby(['Res_geo_unit'])['Trip_Distance_Weighted'].sum())
    ms_dist_all_res.reset_index(inplace=True)
    ms_dist_all_res.rename(columns={'Trip_Distance_Weighted':'Trip_Distance_All'},inplace=True)

    ms_dist_res=pd.DataFrame(sHPW.groupby(['Res_geo_unit','Mode'])['Trip_Distance_Weighted'].sum())
    ms_dist_res.reset_index(inplace=True)
    ms_dist_res=ms_dist_res.merge(ms_dist_all_res)
    ms_dist_res['Share_Distance']=ms_dist_res['Trip_Distance_Weighted']/ms_dist_res['Trip_Distance_All']
    ms_dist_res.drop(columns=['Trip_Distance_Weighted','Trip_Distance_All'],inplace=True)

    # convert long to wide
    msdrp=ms_dist_res.pivot(index='Res_geo_unit',columns=['Mode'])
    msdrp.columns = ['_'.join(col) for col in msdrp.columns.values]
    msdrp.reset_index(inplace=True)

    # avg travel km/day/cap by postcode of residence
    # first calculate weighted number of surveyed people by postcode
    plz_per_weight=pd.DataFrame(sHPW[['Res_geo_unit', 'HH_PNR','Per_Weight']].drop_duplicates().groupby(['Res_geo_unit'])['Per_Weight'].sum()).reset_index() 
    # next calculate sum weighted travel distance by residence postcode
    plz_dist_weight=pd.DataFrame(sHPW.groupby(['Res_geo_unit'])['Trip_Distance_Weighted'].sum()).reset_index()
    plz_dist_weight=plz_dist_weight.merge(plz_per_weight)
    # then divide
    plz_dist_weight['Daily_Distance_Person']=0.001*round(plz_dist_weight['Trip_Distance_Weighted']/plz_dist_weight['Per_Weight'])
    plz_dist_weight.drop(columns=['Trip_Distance_Weighted','Per_Weight'],inplace=True)
    # and car travel distance by residence postcode
    plz_dist_car=pd.DataFrame(sHPW.loc[sHPW['Mode']=='Car',:].groupby(['Res_geo_unit'])['Trip_Distance_Weighted'].sum()).reset_index()
    plz_dist_car=plz_dist_car.merge(plz_per_weight)
    plz_dist_car['Daily_Distance_Person_Car']=0.001*round(plz_dist_car['Trip_Distance_Weighted']/plz_dist_car['Per_Weight'])
    plz_dist_car.drop(columns=['Trip_Distance_Weighted','Per_Weight'],inplace=True)

    # car ownership rates by household
    plz_hh_weight=pd.DataFrame(sHPW[['Res_geo_unit', 'HHNR','HH_Weight']].drop_duplicates().groupby(['Res_geo_unit'])['HH_Weight'].sum()).reset_index() 
    plz_hh_car=pd.DataFrame(sHPW.loc[sHPW['CarAvailable']==1,('Res_geo_unit', 'HHNR','HH_Weight')].drop_duplicates().groupby(['Res_geo_unit'])['HH_Weight'].sum()).reset_index() 
    plz_hh_car.rename(columns={'HH_Weight':'HH_WithCar'},inplace=True)
    plz_hh_car=plz_hh_car.merge(plz_hh_weight)
    plz_hh_car['CarOwnership_HH']=round(plz_hh_car['HH_WithCar']/plz_hh_car['HH_Weight'],3)
    plz_hh_car.drop(columns=['HH_WithCar','HH_Weight'],inplace=True)

    summary=msdrp.merge(plz_dist_weight)
    summary=summary.merge(plz_dist_car)
    summary=summary.merge(plz_hh_car)
    summary.to_csv('../outputs/summary_geounits/'+city+'_agg.csv',index=False)

        # prepare to plot car ownership and distance traveled by income
    if city in ['Clermont','Toulouse']:

        sHPW['Income']=np.nan
        sHPW.loc[sHPW['IncomeDetailed']=='Under1000','Income']=500
        sHPW.loc[sHPW['IncomeDetailed']=='1000-2000','Income']=1500
        sHPW.loc[sHPW['IncomeDetailed']=='2000-3000','Income']=2500
        sHPW.loc[sHPW['IncomeDetailed']=='3000-4000','Income']=3500
        sHPW.loc[sHPW['IncomeDetailed']=='4000-6000','Income']=5000
        sHPW.loc[sHPW['IncomeDetailed']=='Over6000','Income']=6500
        sHPW.loc[sHPW['IncomeDetailed']=='4000-5000','Income']=4500
        sHPW.loc[sHPW['IncomeDetailed']=='5000-7000','Income']=6000
        sHPW.loc[sHPW['IncomeDetailed']=='Over7000','Income']=7500

        # plot car ownership by income 
        hh_inc_car=pd.DataFrame(sHPW.groupby(['HHNR'])['Income','CarOwnershipHH'].mean()).reset_index()
        inc_car=hh_inc_car.groupby('Income')['CarOwnershipHH'].mean().reset_index()
        inc_N=hh_inc_car.groupby('Income')['CarOwnershipHH'].count().reset_index()
        inc_N.rename(columns={'CarOwnershipHH':'N_HH'},inplace=True)
        inc_car=inc_car.merge(inc_N,on='Income')
        fig, ax = plt.subplots()
        plt.plot(inc_car['Income'],100*inc_car['CarOwnershipHH'])
        plt.title('Car Ownership by Income, '+city,fontsize=16)
        plt.xlabel('HH Income, EUR/month',fontsize=12)
        plt.ylabel('HH Car Ownership Rate (%)',fontsize=12)
        ax.grid()
        ax.set_ylim(0,100)
        fig.savefig('../figures/plots/'+ city+'_CarOwnership.png',facecolor='w')

        # plot Car Mode Share by Income
        # these next two vars need recalculated.
        per_dist_mode=pd.DataFrame(sHPW.groupby(['HH_PNR','Mode'])['Trip_Distance'].sum()*0.001).reset_index()
        person_dist_all=pd.DataFrame(sHPW.groupby(['HH_PNR'])['Trip_Distance'].sum()*0.001).reset_index()

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

        inc_stats=inc_car.merge(person_mode_income)
        inc_stats['City']=city
        inc_stats_all=pd.concat([inc_stats_all,inc_stats])   

        # create df for car ownership analysis
        sHP['IncomeDetailed_Numeric']=np.nan
        sHP.loc[sHP['IncomeDetailed']=='Under1000','IncomeDetailed_Numeric']=500
        sHP.loc[sHP['IncomeDetailed']=='1000-2000','IncomeDetailed_Numeric']=1500
        sHP.loc[sHP['IncomeDetailed']=='2000-3000','IncomeDetailed_Numeric']=2500
        sHP.loc[sHP['IncomeDetailed']=='3000-4000','IncomeDetailed_Numeric']=3500
        sHP.loc[sHP['IncomeDetailed']=='4000-6000','IncomeDetailed_Numeric']=5000
        sHP.loc[sHP['IncomeDetailed']=='Over6000','IncomeDetailed_Numeric']=6500
        sHP.loc[sHP['IncomeDetailed']=='4000-5000','IncomeDetailed_Numeric']=4500
        sHP.loc[sHP['IncomeDetailed']=='5000-7000','IncomeDetailed_Numeric']=6000
        sHP.loc[sHP['IncomeDetailed']=='Over7000','IncomeDetailed_Numeric']=7500

        sHP['Adult']=0
        sHP['Child']=0
        sHP.loc[sHP['Age']>17,'Adult']=1
        sHP.loc[sHP['Age']<18,'Child']=1

        Nadult=sHP.loc[:,['HHNR','Adult']].groupby('HHNR')['Adult'].sum().reset_index()
        Nadult.rename(columns={'Adult':'N_Adult_HH'},inplace=True)
        Nchild=sHP.loc[:,['HHNR','Child']].groupby('HHNR')['Child'].sum().reset_index()
        Nchild.rename(columns={'Child':'N_Child_HH'},inplace=True)
        sHP=sHP.merge(Nadult)
        sHP=sHP.merge(Nchild)
        ages=sHP.loc[:,['HHNR','Age']].groupby('HHNR')['Age'].describe().reset_index() 
        ages=ages.loc[:,['HHNR','min','max']]
        ages.rename(columns={'min':'minAgeHH','max':'maxAgeHH'},inplace=True)
        sHP=sHP.merge(ages)
        sHP['HHType_simp']='Blank'
        sHP.loc[(sHP['HHSize']==1) &  (sHP['Sex']==1), 'HHType_simp']='Single_Male'
        sHP.loc[(sHP['HHSize']==1) & (sHP['Sex']==2), 'HHType_simp']='Single_Female'

        sHP.loc[(sHP['N_Adult_HH']==1) & (sHP['N_Child_HH']>0) & (sHP['Age']>17) & (sHP['Sex']==1), 'HHType_simp']='Single_Male_Parent' 
        sHP.loc[(sHP['N_Adult_HH']==1) & (sHP['N_Child_HH']>0) & (sHP['Age']>17) & (sHP['Sex']==2), 'HHType_simp']='Single_Female_Parent'

        sHP.loc[(sHP['N_Adult_HH']>1) & (sHP['N_Child_HH']>0), 'HHType_simp']='MultiAdult_Kids'

        sHP.loc[(sHP['N_Adult_HH']>1) & (sHP['N_Child_HH']==0) , 'HHType_simp']='MultiAdult'

        # fill in the type for kids in single parent households
        for hh in sHP.loc[sHP['HHType_simp']=='Blank','HHNR']:
            if 'Single_Male_Parent' in sHP.loc[sHP['HHNR']==hh,'HHType_simp'].values:
                sHP.loc[sHP['HHNR']==hh,'HHType_simp']='Single_Male_Parent'
            if 'Single_Female_Parent' in sHP.loc[sHP['HHNR']==hh,'HHType_simp'].values:
                sHP.loc[sHP['HHNR']==hh,'HHType_simp']='Single_Female_Parent'

        sHP['HHType_det']='Blank'
        sHP.loc[(sHP['N_Adult_HH']==1) & (sHP['N_Child_HH']==0) & (sHP['Sex']==1) & (sHP['Age']<40), 'HHType_det']='Single_Male_Under40'
        sHP.loc[(sHP['N_Adult_HH']==1) & (sHP['N_Child_HH']==0) & (sHP['Sex']==2) & (sHP['Age']<40), 'HHType_det']='Single_Female_Under40'

        sHP.loc[(sHP['N_Adult_HH']==1) & (sHP['N_Child_HH']==0) & (sHP['Sex']==1) & (sHP['Age']>39) & (sHP['Age']<65), 'HHType_det']='Single_Male_40-64'
        sHP.loc[(sHP['N_Adult_HH']==1) & (sHP['N_Child_HH']==0) & (sHP['Sex']==2) & (sHP['Age']>39) & (sHP['Age']<65), 'HHType_det']='Single_Female_40-64'

        sHP.loc[(sHP['N_Adult_HH']==1) & (sHP['N_Child_HH']==0) & (sHP['Sex']==1) & (sHP['Age']>64), 'HHType_det']='Single_Male_>64'
        sHP.loc[(sHP['N_Adult_HH']==1) & (sHP['N_Child_HH']==0) & (sHP['Sex']==2) & (sHP['Age']>64), 'HHType_det']='Single_Female_>64'

        sHP.loc[(sHP['N_Adult_HH']==1) & (sHP['N_Child_HH']>0) & (sHP['Age']>17) & (sHP['Sex']==1), 'HHType_det']='Single_Male_Parent' 
        sHP.loc[(sHP['N_Adult_HH']==1) & (sHP['N_Child_HH']>0) & (sHP['Age']>17) & (sHP['Sex']==2), 'HHType_det']='Single_Female_Parent'

        sHP.loc[(sHP['N_Adult_HH']>1) & (sHP['N_Child_HH']>0), 'HHType_det']='MultiAdult_Kids'

        sHP.loc[(sHP['N_Adult_HH']>1) & (sHP['N_Child_HH']==0) & (sHP['minAgeHH']>64), 'HHType_det']='MultiAdult_>65'
        sHP.loc[(sHP['N_Adult_HH']>1) & (sHP['N_Child_HH']==0) & (sHP['maxAgeHH']<65), 'HHType_det']='MultiAdult_Under65'
        sHP.loc[(sHP['N_Adult_HH']>1) & (sHP['N_Child_HH']==0) & (sHP['minAgeHH']<65) & (sHP['maxAgeHH']>64), 'HHType_det']='MultiAdult_MixGen'

        # fill in the type for kids in single parent households
        for hh in sHP.loc[sHP['HHType_det']=='Blank','HHNR']:
            if 'Single_Male_Parent' in sHP.loc[sHP['HHNR']==hh,'HHType_det'].values:
                sHP.loc[sHP['HHNR']==hh,'HHType_det']='Single_Male_Parent'
            if 'Single_Female_Parent' in sHP.loc[sHP['HHNR']==hh,'HHType_det'].values:
                sHP.loc[sHP['HHNR']==hh,'HHType_det']='Single_Female_Parent'

        sHP['UniversityEducation']=0
        sHP['InEmployment']=0
        sHP['AllRetired']=0

        for h in sHP['HHNR']:
            if 'University' in sHP.loc[sHP['HHNR']==h, 'Education'].values:
                # print(h)
                # print('Uni')
                sHP.loc[sHP['HHNR']==h, 'UniversityEducation']=1
            if any([any('Employed_PartTime' == sHP.loc[sHP['HHNR']==h, 'Occupation'].values), any('Employed_FullTime' == sHP.loc[sHP['HHNR']==h, 'Occupation'].values)]):
                sHP.loc[sHP['HHNR']==h, 'InEmployment']=1
            if all('Retired' == sHP.loc[sHP['HHNR']==h, 'Occupation'].values):
                sHP.loc[sHP['HHNR']==h, 'AllRetired']=1        

        sH2=sHP.drop(columns=['HH_PNR', 'Person', 'Age', 'Sex','Occupation','Education','Adult','Child','Per_Weight','DrivingLicense', 'HH_Weight', 'minAgeHH', 'Month', 'Season', 'Day',
                        'Work/Study_AtHome', 'Work/Study_BikeParkAvailable','Work/Study_CarParkAvailable',
                        'Relationship','BikeAvailable','TransitSubscription','CarAvailable','2_3WAvailable'])
        sH2.drop_duplicates(inplace=True)
        if len(sH2)!=len(sH):
            print('Mismatch with total N households')
            print('sH2',len(sH2))
            print('sH',len(sH))

        sH2['Res_geocode']=sH2['Sector_Zone'].astype("string").str.zfill(6).map(code_dict).astype("string") 
        sH2['Res_Sec_Zone']=sH2['Sector_Zone'].astype("string").str.zfill(6)

        # now merge all urban form features with the survey data.
        # population density 
        sH2_UF=sH2.merge(pop_dens,left_on='Res_geocode',right_on='geocode').copy()
        sH2_UF.drop(columns='geocode',inplace=True)
        sH2_UF.rename(columns={'Density':'PopDensity'},inplace=True)

        # buidling density 
        sH2_UF=sH2_UF.merge(bld_dens,left_on='Res_geocode',right_on='geocode').copy() 
        sH2_UF.drop(columns='geocode',inplace=True)
        sH2_UF.rename(columns={'build_vol_density':'BuildDensity'},inplace=True)

        # d2c
        sH2_UF=sH2_UF.merge(d2,left_on='Res_Sec_Zone',right_on='geocode').copy() 
        sH2_UF.drop(columns='geocode',inplace=True)
        sH2_UF.rename(columns={'minDist_subcenter':'DistSubcenter','Distance2Center':'DistCenter'},inplace=True)

        # connectivity stats, 
        sH2_UF=sH2_UF.merge(conn,left_on='Res_Sec_Zone',right_on='geocode').copy() 
        sH2_UF.drop(columns='geocode',inplace=True)
        sH2_UF.rename(columns={'k_avg':'K_avg','clean_intersection_density_km':'IntersecDensity','street_density_km':'StreetDensity',
        'streets_per_node_avg':'StreetsPerNode','street_length_avg':'StreetLength'},inplace=True)

        # land-use stats, 
        sH2_UF=sH2_UF.merge(lu,left_on='Res_geocode',right_on='geocode').copy() 
        sH2_UF.drop(columns='geocode',inplace=True)
        sH2_UF.rename(columns={'pc_urb_fabric':'LU_UrbFab','pc_comm':'LU_Comm','pc_road':'LU_Road',
        'pc_urban':'LU_Urban'},inplace=True)

        # recalculate population densities based on urban fabric denominator (Changed to urban, as some demoninators were too low, even some 0 values), and building volume densities based on urban demoninator
        sH2_UF['UrbPopDensity']=sH2_UF['PopDensity']/sH2_UF['LU_Urban']
        sH2_UF['UrbBuildDensity']=sH2_UF['BuildDensity']/sH2_UF['LU_Urban']
        # save
        sH2_UF.to_csv('../outputs/Combined/' + city + '_co_UF.csv',index=False)

        # calculate summary stats by geocode

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


cities=pd.Series(['Clermont','Toulouse','Montpellier','Lyon','Nantes','Nimes','Lille','Dijon'])
cities.apply(combine_survey_data) # args refers to the size threshold above which to divide large units into their smaller sub-components, e.g. 10km2
inc_stats_all.to_csv('../figures/plots/income_stats_FR.csv',index=False)