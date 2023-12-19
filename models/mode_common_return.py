# load in required packages
import numpy as np
import pandas as pd
import geopandas as gpd
import shap
import re
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, cross_validate, GroupKFold, StratifiedGroupKFold, RepeatedKFold, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, linear_model
from xgboost import XGBClassifier, XGBRegressor
import statsmodels.formula.api as smf
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pickle

cities_all=['Berlin','Dresden','Düsseldorf','Frankfurt am Main','Kassel','Leipzig','Magdeburg','Potsdam','Clermont','Dijon','Lille','Lyon','Montpellier','Nantes','Nimes','Paris','Toulouse','Madrid','Wien','France_other','Germany_other']
countries=['Germany','Germany','Germany','Germany','Germany','Germany','Germany','Germany','France','France','France','France','France','France','France','France','France','Spain','Austria','France','Germany']
cities_small=['Dresden','Düsseldorf','Frankfurt am Main','Kassel','Leipzig','Magdeburg','Potsdam','Clermont','Dijon','Lille','Lyon','Montpellier','Nantes','Nimes','Toulouse','France_other','Germany_other']
cities_main=['Berlin','Paris','Madrid','Wien','France_other','Germany_other']
#form_str="Mode_num ~ FeatureM_Trip_Time + FeatureM_Season + FeatureM_Trip_Purpose_Agg + FeatureM_Sex + FeatureM_Age + FeatureM_Trip_Distance + FeatureM_CarOwnershipHH + FeatureM_HHSize + FeatureM_Occupation +  FeatureM_Education  + FeatureM_UrbPopDensity_res +  FeatureM_DistSubcenter_res + FeatureM_DistCenter_res +  FeatureM_UrbBuildDensity_res +  FeatureM_IntersecDensity_res +  FeatureM_street_length_res +FeatureM_bike_lane_share_res +  FeatureM_LU_UrbFab_res + FeatureM_LU_Comm_res + FeatureM_transit_accessibility_res" # FeatureM_CarAvailable replaced by FeatureM_CarOwnershipHH

def mode_model(city):
    country=countries[cities_all.index(city)]
    print(city, country)
    if city=='Germany_other':
            city0='Dresden'
            df0=pd.read_csv('../outputs/Combined/' + city0 + '_UF.csv')
            print(len(df0.columns), 'columns in the data for ', city0)
            df0['City']=city0
            df_all=df0.copy()

            cities=['Leipzig','Magdeburg','Potsdam','Frankfurt am Main','Düsseldorf','Kassel']
            for city1 in cities:
                print(city)
                df1=pd.read_csv('../outputs/Combined/' + city1 + '_UF.csv')
                print(len(df1.columns), 'columns in the data for ', city1)
                df1['City']=city1
                if len(df1.columns==df_all.columns):
                    df_all=pd.concat([df_all,df1])
                    print(city1, 'added.')
                    print(len(df_all), 'rows in the combined dataframe')
            df_all['HHNR']=df_all['City']+'_'+df_all['HHNR'].astype(int).astype(str)
            df_all['HH_PNR']=df_all['City']+'_'+df_all['HH_PNR'].astype(int).astype(str)
            df_all['HH_P_WNR']=df_all['City']+'_'+df_all['HH_P_WNR'].astype(int).astype(str)
            df=df_all.copy()
    elif city=='France_other':
            city0='Clermont'
            df0=pd.read_csv('../outputs/Combined/' + city0 + '_UF.csv')
            df0.drop(columns=['IncomeDetailed', 'IncomeHarmonised','Des_Sec_Zone', 'Month', 'Ori_Sec_Zone','Res_Sec_Zone', 'Sample','Sector_Zone', 'Zone','geo_unit','N_Stops', 'N_Legs','Mode_disagg'],errors='ignore',inplace=True)
            print(len(df0.columns), 'columns in the data for ', city0)
            df0['City']=city0
            df_all=df0.copy()

            cities=['Toulouse','Montpellier','Lyon','Nantes','Nimes','Lille','Dijon']
            for city1 in cities:
                print(city1)
                df1=pd.read_csv('../outputs/Combined/' + city1 + '_UF.csv')
                df1.drop(columns=['IncomeDetailed', 'IncomeHarmonised', 'Des_Sec_Zone', 'Month', 'Ori_Sec_Zone','Res_Sec_Zone',  'Sample','Sector_Zone', 'Zone','geo_unit',
                                'Commune', 'Des_Cell', 'Grid_Cell', 'NoMobilityConstraints','Ori_Cell','N_Stops', 'N_Legs','Mode_disagg'],errors='ignore',inplace=True) # plus spme non-shared Paris variables
                print(len(df1.columns), 'columns in the data for ', city1)
                df1['City']=city1
                if len(df1.columns==df_all.columns):
                    df1=df1[df_all.columns] # this is required for Paris, where the same columns exist after the dropping above, but the order is different
                    df_all=pd.concat([df_all,df1])
                    print(city1, 'added.')
                    print(len(df_all), 'rows in the combined dataframe')
            df_all['HHNR']=df_all['City']+'_'+df_all['HHNR'].astype(str)
            df_all['HH_PNR']=df_all['City']+'_'+df_all['HH_PNR'].astype(str)
            df_all['HH_P_WNR']=df_all['City']+'_'+df_all['HH_P_WNR'].astype(str)
            df=df_all.copy()
        
    else:
        df=pd.read_csv('../outputs/Combined/' + city + '_UF.csv')
    # here we limit to trips starting at home. therefore residential urban form features are same as origin features
    df['Start']=df['trip_type_all'].str[:4]
    df['End']=df['trip_type_all'].str[-4:]
    #df=df.loc[df['Start']=='Home',]
    df=df.loc[(df['End']=='Home') | (df['Start']=='Home'),]
    if country=='Germany':
        df['TravelAlone']=1
        df.loc[(df['N_accomp_HH']>0),'TravelAlone']=0
        df=df.loc[df['Age']>=0,].copy()
        # extract the columns of interest for a trip distance  model. remove variables related to motorisation/equipment ownership
        df=df.loc[:,('HH_P_WNR','HH_PNR', 'HHNR','Ori_geocode', 'Des_geocode','Res_geocode',#'Dist_group', # IDs, trip geocodes, home-Res_geocode
            'Trip_Time', 'Season','Trip_Purpose_Agg','CarOwnershipHH', 'Trip_Distance', # trip details, keep number of accompanying householders now as 'TravelAlone', but remove n_others_car as it gives away the mode.     
            #'TravelAlone',
            'HHSize', #'IncomeDetailed_Numeric', #'IncomeDetailed', 'HHType', # household details
            'Sex',  'Occupation', 'Education','Age', # 'MobilityConstraints',
            'UrbPopDensity_res','DistSubcenter_res', 'DistCenter_res','UrbBuildDensity_res',# 'MeanTime2Transit_res',
            'IntersecDensity_res', 'street_length_res','bike_lane_share_res','transit_accessibility_res', # 'diff', 'K_avg_res', 'StreetDensity_res', 'StreetsPerNode_res', 'K_avg_dest','StreetDensity_dest', 'StreetsPerNode_dest', 
            'LU_UrbFab_res','LU_Comm_res',    # urban form features, land-use features are now all from UA. removed 'LU_Road_res', 'LU_Road_dest',
            # target: mode
            'Mode')
            ]
    elif country=='France':
                # extract the columns of interest for a trip distance  model. remove variables related to motorisation/equipment ownership
        df=df.loc[:,('HH_P_WNR','HH_PNR', 'HHNR','Ori_geocode', 'Des_geocode','Res_geocode',#'Dist_group', # IDs, trip geocodes, home-Res_geocode
            'Trip_Time', 'Season','Trip_Purpose_Agg','CarOwnershipHH', 'Trip_Distance', # trip details, keep number of accompanying householders, but remove n_others_car as it gives away the mode.     
            'HHSize', #'IncomeDetailed', # household details
            'Sex', 'Occupation', 'Education','Age', #'ParkingAvailable_Dest', # personal details, only use age for now, not age group, check later what works beter. missing 'MobilityConstraints' in FR
            'UrbPopDensity_res', 'DistSubcenter_res', 'DistCenter_res','UrbBuildDensity_res', # 'DistSubcenter_dest', 'DistCenter_dest', missing time2trans in FR
            'IntersecDensity_res', 'street_length_res', 'bike_lane_share_res','transit_accessibility_res',
            'LU_UrbFab_res','LU_Comm_res',   # urban form features, land-use features are now all from UA. removed 'LU_Road_res', 'LU_Road_dest',
            # target: mode
            'Mode')
            ]
        
    if city=='Wien':
        df=df.loc[:,('HH_P_WNR','HH_PNR', 'HHNR','Ori_geocode', 'Des_geocode','Res_geocode',#'Dist_group', # IDs, trip geocodes, home-Res_geocode
        'Trip_Time', 'Season','Trip_Purpose_Agg','CarOwnershipHH', 'Trip_Distance', # trip details, keep number of accompanying householders now as 'TravelAlone', but remove n_others_car as it gives away the mode.     
        'HHSize', # 'IncomeDescriptiveNumeric', #'IncomeDetailed', 'HHType', # household details
        'Sex', 'Occupation', 'Education','Age', # 'ParkingAvailable_Dest', # personal details, only use age for now, not age group, check later what works beter
        'UrbPopDensity_res', 'DistSubcenter_res', 'DistCenter_res','UrbBuildDensity_res', # 'DistSubcenter_dest', 'DistCenter_dest', 'MeanTime2Transit_res',
        'IntersecDensity_res', 'street_length_res', 'bike_lane_share_res', 'transit_accessibility_res',# 'K_avg_res', 'StreetDensity_res', 'StreetsPerNode_res', 'K_avg_dest','StreetDensity_dest', 'StreetsPerNode_dest', 
        'LU_UrbFab_res','LU_Comm_res',     # urban form features, land-use features are now all from UA. removed 'LU_Road_res', 'LU_Road_dest',
        # target: mode
        'Mode')
        ]

    if city == 'Madrid':
        df=df.loc[:,('HH_P_WNR','HH_PNR', 'HHNR','Ori_geocode', 'Des_geocode','Res_geocode',#'Dist_group', # IDs, trip geocodes, home-Res_geocode
        'Trip_Time', 'Season','Trip_Purpose_Agg','CarOwnershipHH', 'Trip_Distance', # trip details, keep number of accompanying householders now as 'TravelAlone', but remove n_others_car as it gives away the mode.     
        'HHSize', #'IncomeDetailed', 'HHType', # household details
        'Sex', 'Occupation', 'Education','Age', # personal details, only use age for now, not age group, check later what works beter
        'UrbPopDensity_res', 'DistSubcenter_res', 'DistCenter_res','UrbBuildDensity_res',
        'IntersecDensity_res', 'street_length_res', 'bike_lane_share_res','transit_accessibility_res',
        'LU_UrbFab_res','LU_Comm_res',    # urban form features, land-use features are now all from UA. removed 'LU_Road_res', 'LU_Road_dest', 'LU_Urban_res','LU_Urban_dest', 
        # target: Mode
        'Mode')
        ]

    df['Mode_num']=0
    # car, bike, foot, transit
    df.loc[df['Mode']=='Car','Mode_num']=0
    df.loc[df['Mode']=='Bike','Mode_num']=1
    df.loc[df['Mode']=='Foot','Mode_num']=2
    df.loc[df['Mode']=='Transit','Mode_num']=3

    df['UrbPopDensity_res']=0.01*df['UrbPopDensity_res'] # convert from per/km2 to per/ha
    df.loc[:,['bike_lane_share_res','LU_UrbFab_res','LU_Comm_res']]=100*df.loc[:,['bike_lane_share_res','LU_UrbFab_res','LU_Comm_res']] # convert to percentages
    # remove high building density outliers (For Leipzig)
    df=df.loc[df['UrbBuildDensity_res']<1e8,]  
    df['UrbBuildDensity_res']=1e-6*df['UrbBuildDensity_res'] # convert from m3/km2 to m3/m2 

    df.drop(columns='Mode',inplace=True)

    df.loc[df['Education'].isin(['Secondary+BAC','Secondary+Matura']),'Education']='Secondary'

    # here is a new section on variable selection       
    if city=='Wien':
           df.drop(columns=['IntersecDensity_res','LU_UrbFab_res','UrbBuildDensity_res','transit_accessibility_res'],inplace=True)
           form_str="Mode_num ~ FeatureM_Trip_Time + FeatureM_Season + FeatureM_Trip_Purpose_Agg + FeatureM_Sex + FeatureM_Age + FeatureM_Trip_Distance + FeatureM_CarOwnershipHH + FeatureM_HHSize + FeatureM_Occupation + FeatureM_Education + FeatureM_UrbPopDensity_res + FeatureM_DistSubcenter_res + FeatureM_DistCenter_res + FeatureM_street_length_res + FeatureM_bike_lane_share_res + FeatureM_LU_Comm_res" 
    elif city in ['Berlin','Paris']:
           df.drop(columns=['UrbBuildDensity_res'],inplace=True)
           form_str="Mode_num ~ FeatureM_Trip_Time + FeatureM_Season + FeatureM_Trip_Purpose_Agg + FeatureM_Sex + FeatureM_Age + FeatureM_Trip_Distance + FeatureM_CarOwnershipHH + FeatureM_HHSize + FeatureM_Occupation + FeatureM_Education + FeatureM_UrbPopDensity_res + FeatureM_DistSubcenter_res + FeatureM_DistCenter_res +  FeatureM_IntersecDensity_res + FeatureM_street_length_res +FeatureM_bike_lane_share_res + FeatureM_LU_UrbFab_res + FeatureM_LU_Comm_res + FeatureM_transit_accessibility_res"       
    # elif city in cities_small:
    #        df.drop(columns=['IntersecDensity_res'],inplace=True)
    #        form_str="Mode_num ~ FeatureM_Trip_Time + FeatureM_Season + FeatureM_Trip_Purpose_Agg + FeatureM_Sex + FeatureM_Age + FeatureM_Trip_Distance + FeatureM_CarOwnershipHH + FeatureM_HHSize + FeatureM_Occupation +  FeatureM_Education + FeatureM_UrbPopDensity_res +  FeatureM_DistSubcenter_res + FeatureM_DistCenter_res +  FeatureM_UrbBuildDensity_res +  FeatureM_street_length_res +FeatureM_bike_lane_share_res +  FeatureM_LU_UrbFab_res + FeatureM_LU_Comm_res + FeatureM_transit_accessibility_res"     
    else:
         form_str="Mode_num ~ FeatureM_Trip_Time + FeatureM_Season + FeatureM_Trip_Purpose_Agg + FeatureM_Sex + FeatureM_Age + FeatureM_Trip_Distance + FeatureM_CarOwnershipHH + FeatureM_HHSize + FeatureM_Occupation +  FeatureM_Education + FeatureM_UrbPopDensity_res +  FeatureM_DistSubcenter_res + FeatureM_DistCenter_res +  FeatureM_UrbBuildDensity_res +  FeatureM_IntersecDensity_res +  FeatureM_street_length_res +FeatureM_bike_lane_share_res +  FeatureM_LU_UrbFab_res + FeatureM_LU_Comm_res + FeatureM_transit_accessibility_res" 

    # identify the feature columns
    N_non_feature=6 # how many non-features are at the start of the df
    cols=df.columns
    newcols=(df.columns[:N_non_feature].tolist()) + ('FeatureM' +'_'+ cols[N_non_feature:-1]).tolist() + (df.columns[-1:].tolist())
    # change column names
    df.set_axis(newcols,axis=1,inplace=True)
    df = df.reset_index(drop=True)
    df.dropna(inplace=True)
    df0=df.copy()

    # convert  all categorical variables to dummies
    df_Cat=df.select_dtypes('object')[[col for col in df.select_dtypes('object').columns if "FeatureM" in col]]
    for col in df_Cat:
        dum=pd.get_dummies(df[[col]])
        df = pd.concat([df, dum], axis = 1)
        # remove the original categorical columns
    df.drop(df_Cat.columns.tolist(),axis=1,inplace=True)
    # HPO with full dataset, grouping by individual person

    target = 'Mode_num'
    N=len(df)

    # Define the parameter space to be considered
    PS = {"learning_rate": [0.1 ,0.15,0.2,0.3], 
                    "n_estimators": [100, 200, 300, 400],
                    "max_depth":[3, 4, 5]}

    X=df[[col for col in df.columns if "FeatureM" in col]]
    y = df[target]

    tf=(X < 0).all(0)
    print(len(tf[tf]),' columns with value below zero')
    if len(tf[tf])>0:
        print(tf[tf].index.values)
        raise Exception("Some columns have values below zero")

    gr=df['HH_PNR']
    groups=gr
    gkf = list(GroupKFold(n_splits=9).split(X,y,groups)) # i found somewhere online that i had to define the cv splitter as a list, can't find the source at the moment.

    fp='../outputs/ML_Results/'+city+'_HPO_mode_common_new_summary.csv'
    if os.path.isfile(fp):
        print('HPs already identified')
        HPO_summary=pd.read_csv(fp)
        n_parameter_all = HPO_summary['N_est'][0]
        lr_parameter_all = HPO_summary['LR'][0]
        md_parameter_all = HPO_summary['MD'][0]
    else:
        # define grid search cross validator
        tuning_all = GridSearchCV(estimator=XGBClassifier(verbosity=0,use_label_encoder=False), param_grid=PS, cv=gkf, scoring="f1_weighted",return_train_score=True)
        tuning_all.fit(X,y)

        print('best hyper-parameters identified by HPO')
        print(tuning_all.best_params_)
        print('model score with best hyper-paramteres')
        print(tuning_all.best_score_)
        cv_res_all=tuning_all.cv_results_

        n_parameter_all = tuning_all.best_params_['n_estimators']
        lr_parameter_all = tuning_all.best_params_['learning_rate']
        md_parameter_all = tuning_all.best_params_['max_depth']

        # save results of HPO
        r8=['gkf_gridSearch','full','9splits_hhperGroups',tuning_all.best_params_['learning_rate'],tuning_all.best_params_['max_depth'],tuning_all.best_params_['n_estimators'],round(tuning_all.best_score_,3),round(cv_res_all['std_test_score'][tuning_all.best_index_],3),N] #
        # also include other results lists here if HPO is done for more than one cv type or sample
        HPO_summary=pd.DataFrame([r8],columns=['CV_Type','Sample','CV_params','LR','MD','N_est','F1_best','SD_best','N_obs']) # the last element in this case is the sd of f1 scores in the fold which produced best results

    # now redo the CV and calculate the SHAP values with the best HPs
    cv = GroupKFold(n_splits=9)

    y_predict = pd.DataFrame()
    y_predict2 = pd.DataFrame()
    y_test = pd.DataFrame()
    y_test2 = pd.DataFrame()

    shap_values0= pd.DataFrame()
    shap_values1= pd.DataFrame()
    shap_values2= pd.DataFrame()
    shap_values3= pd.DataFrame()

    summ_table_list=[]

    model = XGBClassifier(
        max_depth=md_parameter_all, 
        n_estimators=n_parameter_all, 
        learning_rate=lr_parameter_all)
    
    # model2 = LogisticRegression()
    writer = pd.ExcelWriter('../outputs/ML_Results/mode_MNLR_return/'  + city + '.xlsx', engine='openpyxl')
    for train_idx, test_idx in cv.split(X,groups=gr): # select here 
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        df_train, df_test = df0.iloc[train_idx], df0.iloc[test_idx]
        #id=test_idx[0].astype(str)
        id=datetime.now().strftime("%S%f")
        print('id',id)

        # train & predict
        model.fit(X_train, y_train, verbose=False, eval_set=[(X_train, y_train), (X_test, y_test_fold)])
        y_predict_fold = pd.Series(model.predict(X_test), index=X_test.index)

        # explain
        explainer = shap.TreeExplainer(model)
        shap_values_fold = explainer.shap_values(X_test)

        shap_values_fold0=shap_values_fold[0]
        shap_values_fold1=shap_values_fold[1]
        shap_values_fold2=shap_values_fold[2]
        shap_values_fold3=shap_values_fold[3]
        
        shap_values_fold0 = pd.DataFrame(shap_values_fold0, index=X_test.index, columns=X.columns)
        shap_values_fold1 = pd.DataFrame(shap_values_fold1, index=X_test.index, columns=X.columns)
        shap_values_fold2 = pd.DataFrame(shap_values_fold2, index=X_test.index, columns=X.columns)
        shap_values_fold3 = pd.DataFrame(shap_values_fold3, index=X_test.index, columns=X.columns)    

        y_predict = pd.concat([y_predict, y_predict_fold], axis=0)
        y_test = pd.concat([y_test, y_test_fold], axis=0)

        shap_values0 = pd.concat([shap_values0, shap_values_fold0], axis=0)
        shap_values1 = pd.concat([shap_values1, shap_values_fold1], axis=0)
        shap_values2 = pd.concat([shap_values2, shap_values_fold2], axis=0)
        shap_values3 = pd.concat([shap_values3, shap_values_fold3], axis=0)
        
        #model2.fit(X_train, y_train)
        # y_predict_fold2 = pd.Series(model2.predict(X_test), index=X_test.index)
        # y_predict2 = pd.concat([y_predict2, y_predict_fold2], axis=0)
        
        # X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        # y_train, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        # df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
        y_test_fold2=df_test['Mode_num']

        try:
            log_reg = smf.mnlogit(form_str, data=df_train).fit()
            yhat=np.asarray(log_reg.predict(df_test.drop(columns='Mode_num'))).argmax(1)
            y_predict_fold2 = pd.Series(yhat, index=df_test.index)
            y_predict2 = pd.concat([y_predict2, y_predict_fold2], axis=0)
            y_test2 = pd.concat([y_test2, y_test_fold2], axis=0)

            coeff=log_reg.params.reset_index()
            coeff.rename(columns={'index':'param',0:'bike',1:'walk',2:'transit'},inplace=True)
            coeff.head()
            pval=log_reg.pvalues.reset_index()
            pval.rename(columns={'index':'param',0:'bike_p',1:'walk_p',2:'transit_p'},inplace=True)
            pval.head()
            summ_table=pd.concat([coeff,pval.loc[:,['bike_p','walk_p','transit_p']]],axis=1)
            summ_table['param']=summ_table['param'].str.replace('FeatureM_','')

            st_list_fold=[summ_table.drop(columns='param').to_numpy()]
            summ_table_list.append(st_list_fold)

            summ_table.to_excel(writer, sheet_name='summ' + id,index=False)
        except: 
            print('Logit Model Error')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()

    mdarray=np.array(summ_table_list).squeeze()
    means=np.nanmean(mdarray,axis=0)
    means_df=pd.DataFrame(data=np.hstack((np.reshape(summ_table['param'].to_numpy(),(len(summ_table),1)),means)),columns=summ_table.columns.values)
    means_df.to_csv('../outputs/ML_Results/mode_MNLR_return/'  + city + '_mean.csv',index=False)

    y_test = y_test.squeeze(axis=1)
    y_test2 = y_test2.squeeze(axis=1)
    y_predict = y_predict.squeeze(axis=1)
    y_predict2 = y_predict2.squeeze(axis=1)
    f1_model_ML = metrics.f1_score(y_test, y_predict, average='weighted')
    f1_model_LR = metrics.f1_score(y_test2, y_predict2, average='weighted')
    print('Model f1: ' + city)
    print(f1_model_ML)

    HPO_summary['F1_full_ML']=f1_model_ML
    HPO_summary['F1_full_LR']=f1_model_LR
    HPO_summary['City']=city
    HPO_summary.to_csv('../outputs/ML_Results/' + city + '_HPO_mode_common_return_summary.csv',index=False)

    # optionally here, check which variables are more important than random noise, then downselect X to those variables, and go back to HPO (or CV) and run once more from there.

    shap_valueslist=[shap_values0.sort_index().to_numpy(),shap_values1.sort_index().to_numpy(),shap_values2.sort_index().to_numpy(),shap_values3.sort_index().to_numpy()]
    X_disp=[re.sub('FeatureM_','', x) for x in X.sort_index().columns]
    shap_values_abs=abs(shap_valueslist[0])+abs(shap_valueslist[1])+abs(shap_valueslist[2])+abs(shap_valueslist[3])
    shap_sum = shap_values_abs.mean(axis=0)
    importance_df = pd.DataFrame([X_disp, shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)

    with open('../outputs/ML_Results/shap/mode_common_return/' + city + '_importance.pkl', 'wb') as g:
        pickle.dump(importance_df, g)

    with open('../outputs/ML_Results/shap/mode_common_return/' + city + '_df.pkl', 'wb') as g:
        pickle.dump(df, g)

    # save shap_values, to enable later re-creation and editing of shap plots
    with open('../outputs/ML_Results/shap/mode_common_return/' + city + '.pkl', 'wb') as f:
        pickle.dump(shap_valueslist, f)

    shap.summary_plot(shap_valueslist, X.sort_index(),feature_names=X_disp,max_display=16,  class_names=['car','bike','foot','trans'],class_inds='original', show=False)
    plt.title('Overall Feature Influence Mode Choice ' + city.replace('_',', '))
    plt.savefig('../outputs/ML_Results/result_figures/mode_common_return/' + city + '_mode_FI.png',facecolor='w',dpi=65,bbox_inches='tight')
    plt.close() 


    cm = metrics.confusion_matrix(y_test, y_predict,normalize='true')
    # note the high confusion between bike/foot trips, and the high number of transit trips labelled as bike or car, and the high number of bike trips labelled as foot or car
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.array(['car','bike','foot','trans']))
    disp.plot()
    plt.title('Confusion matrix, mode choice, ' + city + '. F1: ' + str(round(f1_model_ML,3)))
    plt.savefig('../outputs/ML_Results/result_figures/mode_common_return/' + city + '_mode_CM.png',facecolor='w',dpi=65,bbox_inches='tight')
    plt.close() 

    col_dict= {'DistCenter_res':'Dist. to city center','DistSubcenter_res':'Dist. to subenter', 'UrbPopDensity_res':'Population density','UrbBuildDensity_res':'Built-up density','ParkingAvailable_Dest':'Parking available',
        'IntersecDensity_res':'Intersection density','LU_Comm_res':'Commercial area','LU_UrbFab_res':'Urban Fabric area','street_length_res':'Avg. street length','bike_lane_share_res':'Cycle lane share',
        'Trip_Purpose_Agg_Home↔Work':'Commute trip', 'Trip_Purpose_Agg_Home↔Companion':'Companion trip', 'TravelAlone':'Solo trip','Trip_Purpose_Agg_Home↔Leisure':'Leisure trip','Trip_Purpose_Agg_Home↔Shopping':'Shopping trip','Trip_Purpose_Agg_Home↔School':'School trip',
        'Trip_Time_Evening':'Evening trip','Trip_Time_AM_Rush':'Morning rush-hour','Trip_Time_PM Rush':'Evening rush-hour','transit_accessibility_res':'Transit Accessibility', 'Season_Winter':'Winter season',
        'Trip_Distance':'Trip distance','CarOwnershipHH':'Car ownership','Occupation_Student_School':'School Student',
        'Age':'Age','Sex':'Sex','HHSize':'Household size','IncomeDescriptiveNumeric':'Income','IncomeDetailed_Numeric':'Income',
        'Education_University':'University education','Education_Apprenticeship':'Apprenticeship education', 'Occupation_Employed_FullTime':'Employed'}

    X_lab=[*map(col_dict.get, X_disp)]

    shap_values=shap_valueslist

    if city not in ['Berlin','Paris','Madrid','Wien','France_other','Germany_other']:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11,8))
        ax1 = plt.subplot(221)
        shap.summary_plot(shap_values[2], X, feature_names=X_lab, max_display=8, plot_size=None, show=False)
        plt.title('Main Feature Influences for Foot Trips ' + city)
        plt.xlabel("SHAP value", size=11)
        ax2 = plt.subplot(222)
        shap.summary_plot(shap_values[1], X, feature_names=X_lab, max_display=8, plot_size=None, show=False)
        plt.title('Main Feature Influences for Bike Trips ' + city)
        plt.xlabel("SHAP value", size=11)
        ax3 = plt.subplot(223)
        shap.summary_plot(shap_values[0], X, feature_names=X_lab, max_display=8, plot_size=None, show=False)
        plt.title('Main Feature Influences for Car Trips ' + city)
        plt.xlabel("SHAP value", size=11)
        ax3 = plt.subplot(224)
        shap.summary_plot(shap_values[3], X, feature_names=X_lab, max_display=8, plot_size=None, show=False)
        plt.title('Main Feature Influences for Transit Trips ' + city)
        plt.xlabel("SHAP value", size=11)
        plt.savefig('../outputs/ML_Results/result_figures/mode_common_return/' + city + '_FI_all.png',facecolor='w',dpi=65,bbox_inches='tight')
        plt.close() 


    X_lab=[*map(col_dict.get, X_disp)]
    if city == 'Berlin': let='a'
    if city == 'Paris': let='b'
    if city == 'Madrid': let='c'
    if city == 'Wien': let='d'
    if city == 'Germany_other': let='e'
    if city == 'France_other': let='f'

    if city in ['Berlin','Paris','Madrid','Wien','France_other','Germany_other']:
        fig, axes = plt.subplots(figsize=(5.5,4))
        shap.summary_plot(shap_values[0], X, feature_names=X_lab, max_display=8, plot_size=None, show=False)
        if city =='Wien':
            citylab='Vienna'
        else:
            citylab=city
        #     plt.title(let + ') Vienna', size=14)
        # else:
        plt.title(let + ') ' + citylab.replace('_',', '),size=14)
        plt.xlabel("SHAP (probability of car mode choice)", size=11)
        plt.savefig('../outputs/ML_Results/result_figures/mode_common_return/' + city + '_FI_car.png',facecolor='w',dpi=65,bbox_inches='tight')
        plt.close() 
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11,8))
        ax1 = plt.subplot(221)
        shap.summary_plot(shap_values[2], X, feature_names=X_lab, max_display=8, plot_size=None, show=False)
        plt.title(let + ') ' + 'Main Feature Influences for Foot Trips ' + citylab.replace('_',', '))
        plt.xlabel("SHAP (probability of foot mode)", size=11)
        ax2 = plt.subplot(222)
        shap.summary_plot(shap_values[1], X, feature_names=X_lab, max_display=8, plot_size=None, show=False)
        plt.title('Main Feature Influences for Bike Trips ' + citylab.replace('_',', '))
        
        plt.xlabel("SHAP (probability of bike mode)", size=11)
        ax3 = plt.subplot(223)
        shap.summary_plot(shap_values[0], X, feature_names=X_lab, max_display=8, plot_size=None, show=False)
        plt.title('Main Feature Influences for Car Trips ' + citylab.replace('_',', '))
        
        plt.xlabel("SHAP (probability of car mode)", size=11)
        ax3 = plt.subplot(224)
        shap.summary_plot(shap_values[3], X, feature_names=X_lab, max_display=8, plot_size=None, show=False)
        plt.title('Main Feature Influences for Transit Trips ' + citylab.replace('_',', '))
        
        plt.xlabel("SHAP (probability of transit mode)", size=11)

        plt.savefig('../outputs/ML_Results/result_figures/mode_common_return/' + city + '_FI_all.png',facecolor='w',dpi=65,bbox_inches='tight')
        plt.close() 
#cities_list=pd.Series(['France_other','Germany_other']) 
cities_list=pd.Series(cities_main)
#cities_list=pd.Series(['Clermont','Dijon','Lille','Lyon','Montpellier','Nantes','Nimes','Toulouse','France_other'])

cities_list.apply(mode_model) # args refers to the size threshold above which to divide large units into their smaller sub-components, e.g. 10km2