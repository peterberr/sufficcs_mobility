# script to model avergage trip distances for commute trips in all cities
# last update Peter Berrill Aug 1 2023

# load in required packages
import numpy as np
import pandas as pd
import shap
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, cross_validate, GroupKFold, StratifiedGroupKFold, RepeatedKFold, StratifiedKFold, GridSearchCV, KFold
from sklearn import metrics, linear_model
from xgboost import XGBClassifier, XGBRegressor
import os
import sys
import matplotlib.pyplot as plt
import pickle
import statsmodels.formula.api as smf
from datetime import datetime

cities_all=['Berlin','Dresden','Düsseldorf','Frankfurt am Main','Kassel','Leipzig','Magdeburg','Potsdam','Clermont','Dijon','Lille','Lyon','Montpellier','Nantes','Nimes','Paris','Toulouse','Madrid','Wien','France_other','Germany_other']
countries=['Germany','Germany','Germany','Germany','Germany','Germany','Germany','Germany','France','France','France','France','France','France','France','France','France','Spain','Austria','France','Germany']

def dist_commute(city):
    country=countries[cities_all.index(city)]
    print(city, country)
    if city=='Germany_other':
        city0='Dresden'
        df0=pd.read_csv('../outputs/Combined/' + city0 + '_UF.csv')
        df0.loc[(df0['Training'].isin(['Apprenticeship/Business','Craftsman/Technical'])) & (df0['Education']!='University'),'Education']='Apprenticeship'
        df0=df0.loc[:,['HH_P_WNR','HH_PNR', 'HHNR','Ori_geocode', 'Des_geocode','Res_geocode', 
                    'Trip_Time', 'Season','Trip_Purpose_Agg','HHSize',
                    'Sex', 'Occupation', 'Education','Age',
                    #'PopDensity_res','BuildDensity_res',
                    'UrbPopDensity_res', 'UrbBuildDensity_res','DistSubcenter_res', 'DistCenter_res',
                    'IntersecDensity_res', 'street_length_res', 'LU_UrbFab_res',#'bike_lane_share_res',
                    'LU_Comm_res' ,'Trip_Distance']]
        df0['City']=city0
        df_all=df0.copy()

        cities0=['Leipzig','Magdeburg','Potsdam','Frankfurt am Main','Düsseldorf','Kassel']
        for city1 in cities0:
                print(city1)
                df1=pd.read_csv('../outputs/Combined/' + city1 + '_UF.csv')
                df1.loc[(df1['Training'].isin(['Apprenticeship/Business','Craftsman/Technical'])) & (df1['Education']!='University'),'Education']='Apprenticeship'
                df1=df1.loc[:,['HH_P_WNR','HH_PNR', 'HHNR','Ori_geocode', 'Des_geocode','Res_geocode', 
                            'Trip_Time', 'Season','Trip_Purpose_Agg','HHSize',
                            'Sex', 'Occupation', 'Education','Age',
                            #'PopDensity_res','BuildDensity_res',
                            'UrbPopDensity_res', 'UrbBuildDensity_res','DistSubcenter_res', 'DistCenter_res',
                            'IntersecDensity_res', 'street_length_res', 'LU_UrbFab_res',#'bike_lane_share_res',
                            'LU_Comm_res','Trip_Distance']]
                df1['City']=city1
                if len(df1.columns==df_all.columns):
                       df_all=pd.concat([df_all,df1])
                       print(city1, 'added.')
                       #print(len(df_all), 'rows in the combined dataframe')
        df_all['HHNR']=df_all['City']+'_'+df_all['HHNR'].astype(int).astype(str)
        df_all['HH_PNR']=df_all['City']+'_'+df_all['HH_PNR'].astype(int).astype(str)
        df_all['HH_P_WNR']=df_all['City']+'_'+df_all['HH_P_WNR'].astype(str)
        df_all.drop(columns='City',inplace=True)
        df_UF=df_all.copy()
    elif city=='France_other':
        city0='Clermont'
        df0=pd.read_csv('../outputs/Combined/' + city0 + '_UF.csv')
        df0=df0.loc[:,['HH_P_WNR','HH_PNR', 'HHNR','Ori_geocode', 'Des_geocode','Res_geocode', 
                    'Trip_Time', 'Season','Trip_Purpose_Agg','HHSize',
                    'Sex', 'Occupation', 'Education','Age',
                    #'PopDensity_res','BuildDensity_res',
                    'UrbPopDensity_res', 'UrbBuildDensity_res','DistSubcenter_res', 'DistCenter_res',
                    'IntersecDensity_res', 'street_length_res', 'LU_UrbFab_res',#'bike_lane_share_res',
                    'LU_Comm_res', 'Trip_Distance']]
        df0['City']=city0
        df_all=df0.copy()

        cities0=['Toulouse','Montpellier','Lyon','Nantes','Nimes','Lille','Dijon']
        for city1 in cities0:
                print(city1)
                df1=pd.read_csv('../outputs/Combined/' + city1 + '_UF.csv')
                df1=df1.loc[:,['HH_P_WNR','HH_PNR', 'HHNR','Ori_geocode', 'Des_geocode','Res_geocode', 
                            'Trip_Time', 'Season','Trip_Purpose_Agg','HHSize',
                            'Sex', 'Occupation', 'Education','Age',
                            #'PopDensity_res','BuildDensity_res',
                            'UrbPopDensity_res', 'UrbBuildDensity_res','DistSubcenter_res', 'DistCenter_res',
                            'IntersecDensity_res', 'street_length_res', 'LU_UrbFab_res',#'bike_lane_share_res',
                            'LU_Comm_res', 'Trip_Distance']]
                df1['City']=city1
                if len(df1.columns==df_all.columns):
                       df_all=pd.concat([df_all,df1])
                       print(city1, 'added.')
                       #print(len(df_all), 'rows in the combined dataframe')
        df_all['HHNR']=df_all['City']+'_'+df_all['HHNR'].astype(int).astype(str)
        df_all['HH_PNR']=df_all['City']+'_'+df_all['HH_PNR'].astype(int).astype(str)
        df_all['HH_P_WNR']=df_all['City']+'_'+df_all['HH_P_WNR'].astype(str)
        df_all.drop(columns='City',inplace=True)
        df_UF=df_all.copy()
    else:
            df=pd.read_csv('../outputs/Combined/' + city + '_UF.csv',dtype={'Ori_geocode': str, 'Des_geocode': str,'Res_geocode': str })
            if country=='Germany':
                df.loc[(df['Training'].isin(['Apprenticeship/Business','Craftsman/Technical'])) & (df['Education']!='University'),'Education']='Apprenticeship'
            df_UF=df.loc[:,['HH_P_WNR','HH_PNR', 'HHNR','Ori_geocode', 'Des_geocode','Res_geocode', 
                            'Trip_Time', 'Season','Trip_Purpose_Agg','HHSize',
                            'Sex', 'Occupation', 'Education','Age',
                            #'PopDensity_res','BuildDensity_res',
                            'UrbPopDensity_res', 'UrbBuildDensity_res','DistSubcenter_res', 'DistCenter_res',
                            'IntersecDensity_res', 'street_length_res', 'LU_UrbFab_res',#'bike_lane_share_res',
                            'LU_Comm_res', 'Trip_Distance']]
    # restrict to trips between home and work (commuting trips)        
    df_UF=df_UF.loc[df_UF['Trip_Purpose_Agg']=='Home↔Work',]
    df_UF.drop(columns='Trip_Purpose_Agg',inplace=True)
    # restrict to those in employment
    df_UF=df_UF.loc[df_UF['Occupation'].isin(['Trainee','Employed_FullTime','Employed_PartTime','Employed']),]
#     df_UF.loc[df_UF['Education'].isin(['No diploma yet','Other','Apprenticeship','Unknown']),'Education']='Unknown/Other'
#     df_UF.loc[df_UF['Education'].isin(['Secondary+BAC','Secondary+Matura']),'Education']='Secondary'
    Edu_dict={'University':'University','Secondary':'Secondary','Secondary+BAC':'Secondary','Secondary+Matura':'Secondary',
          'Apprenticeship':'Apprenticeship',
          'Elementary':'Primary/None','Pre-School':'Primary/None','No diploma yet':'Primary/None','Unknown':'Primary/None','Other':'Primary/None'}

    df_UF['Education']=df_UF['Education'].map(Edu_dict)
#     if city=='Clermont':
#           df_UF=df_UF.loc[df_UF['Education']!='Unknown/Other',]
    if city in ['Clermont','Nimes']:
          df_UF.loc[df_UF['Education']=='Apprenticeship','Education']='Secondary'

    df=df_UF.dropna()
    df['Sex']=df['Sex']-1 # change from [1,2] to [0,1], for plotting purposes
    df=df.loc[df['UrbBuildDensity_res']<1e8,]   # remove high building density outliers (For Leipzig)

    # identify the feature columns
    N_non_feature=6 # how many non-features are at the start of the df
    cols=df.columns
    newcols=(df.columns[:N_non_feature].tolist()) + ('FeatureD' +'_'+ cols[N_non_feature:-1]).tolist() + (df.columns[-1:].tolist())
    # change column names
    df.set_axis(newcols,axis=1,inplace=True)
    df = df.reset_index(drop=True)
    df0=df.copy()

    # convert  all categorical variables to dummies
    df_Cat=df.select_dtypes('object')[[col for col in df.select_dtypes('object').columns if "FeatureD" in col]]
    for col in df_Cat:
        dum=pd.get_dummies(df[[col]])
        df = pd.concat([df, dum], axis = 1)
        # remove the original categorical columns
    df.drop(df_Cat.columns.tolist(),axis=1,inplace=True)
    # HPO with full dataset, grouping by individual person
    target = 'Trip_Distance'
    N=len(df)
    # Define the parameter space to be considered
    PS = {"learning_rate": [0.1 ,0.15,0.2,0.3], 
                    "n_estimators": [100, 200, 300],
                    "max_depth":[3, 4, 5]}

    X=df[[col for col in df.columns if "FeatureD" in col]]
    y = df[target]

    tf=(X < 0).all(0)
    print(len(tf[tf]),' columns with value below zero')
    if len(tf[tf])>0:
        print(tf[tf].index.values)
        raise Exception("Some columns have values below zero")
    
    gr=df['HH_PNR']
    groups=gr
    gkf = list(GroupKFold(n_splits=9).split(X,y,groups)) # i found somewhere online that i had to define the cv splitter as a list, can't find the source at the moment.

    fp='../outputs/ML_Results/'+city+'_HPO_dist_commute_summary.csv'
    if os.path.isfile(fp):
        print('HPs already identified')
        HPO_summary=pd.read_csv(fp)
        n_parameter_all = HPO_summary['N_est'][0]
        lr_parameter_all = HPO_summary['LR'][0]
        md_parameter_all = HPO_summary['MD'][0]
    else:
        # define grid search cross validator
        tuning_all = GridSearchCV(estimator=XGBRegressor(verbosity=0,use_label_encoder=False), param_grid=PS, cv=gkf, scoring="r2",return_train_score=True)
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

    shap_values = pd.DataFrame()

    r2ml=[]
    r2lr=[]
    summ_table_list=[]

    model = XGBRegressor(
        max_depth=md_parameter_all, 
        n_estimators=n_parameter_all, 
        learning_rate=lr_parameter_all)
    
    writer = pd.ExcelWriter('../outputs/ML_Results/dist_commute/'  + city + '.xlsx', engine='openpyxl')
    form_str="Trip_Distance ~  FeatureD_HHSize + FeatureD_Sex + FeatureD_Education + FeatureD_Age + FeatureD_Season +  FeatureD_DistSubcenter_res + FeatureD_DistCenter_res + FeatureD_UrbPopDensity_res + FeatureD_UrbBuildDensity_res  + FeatureD_IntersecDensity_res + FeatureD_street_length_res + FeatureD_LU_Comm_res +  FeatureD_LU_UrbFab_res" # + FeatureD_bike_lane_share_res"

    for train_idx, test_idx in cv.split(X,groups=gr): # select here 
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        df_train, df_test = df0.iloc[train_idx], df0.iloc[test_idx]
        y_test_fold2=df_test['Trip_Distance']
        id=datetime.now().strftime("%S%f")
        print('id',id)

        # train & predict
        model.fit(X_train, y_train, verbose=False, eval_set=[(X_train, y_train), (X_test, y_test_fold)])
        y_predict_fold = pd.Series(model.predict(X_test), index=X_test.index)
        r2ml.extend([metrics.r2_score(y_test_fold.array, y_predict_fold.array)])
        # explain
        explainer = shap.TreeExplainer(model)
        
        shap_values_fold = explainer.shap_values(X_test,check_additivity=False)
        
        shap_values_fold = pd.DataFrame(shap_values_fold, index=X_test.index, columns=X.columns) 

        y_predict = pd.concat([y_predict, y_predict_fold], axis=0)
        y_test = pd.concat([y_test, y_test_fold], axis=0)

        shap_values = pd.concat([shap_values, shap_values_fold], axis=0)
        
        lin_reg = smf.ols(form_str, data=df_train).fit()
        yhat=np.asarray(lin_reg.predict(df_test.drop(columns='Trip_Distance')))
        y_predict_fold2 = pd.Series(yhat, index=df_test.index)
        y_predict2 = pd.concat([y_predict2, y_predict_fold2], axis=0)
        y_test2 = pd.concat([y_test2, y_test_fold2], axis=0)
        
        r2lr.extend([metrics.r2_score(y_test_fold.array, y_predict_fold2.array)])

        coeff=lin_reg.params.reset_index()
        coeff.rename(columns={'index':'param',0:'coefficient'},inplace=True)

        pval=lin_reg.pvalues.reset_index()
        pval.rename(columns={'index':'param',0:'p'},inplace=True)

        summ_table=pd.concat([coeff,pval['p']],axis=1)
        summ_table['param']=summ_table['param'].str.replace('FeatureD_','')

        st_list_fold=[summ_table.drop(columns='param').to_numpy()]
        summ_table_list.append(st_list_fold)

        summ_table.to_excel(writer, sheet_name='summ' + id,index=False)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()
    
    mdarray=np.array(summ_table_list).squeeze()
    means=np.nanmean(mdarray,axis=0)
    means_df=pd.DataFrame(data=np.hstack((np.reshape(summ_table['param'].to_numpy(),(len(summ_table),1)),means)),columns=summ_table.columns.values)
    means_df.to_csv('../outputs/ML_Results/dist_commute/'  + city + '_mean.csv',index=False)

    y_test = y_test.squeeze(axis=1)
    y_test2 = y_test2.squeeze(axis=1)
    y_predict = y_predict.squeeze(axis=1)
    y_predict2 = y_predict2.squeeze(axis=1)
    r2_model=metrics.r2_score(y_test, y_predict)
    r2_model_reg=metrics.r2_score(y_test2, y_predict2)
    print('GBDT Model r2: ' + city)
    print(r2_model)
    print('LR Model r2: ' + city)
    print(r2_model_reg)
    HPO_summary['R2_full_ML']=r2_model
    HPO_summary['R2_full_LR']=r2_model_reg
    HPO_summary['City']=city
    HPO_summary.to_csv('../outputs/ML_Results/' + city + '_HPO_dist_commute_summary.csv',index=False)

    X_disp=[re.sub('FeatureD_','', x) for x in X.columns]

    shap_values=shap_values.sort_index()
    shap_values.reset_index(inplace=True)
    shap_values=shap_values.groupby('index').mean().reset_index()
    shap_values.drop(columns=['index'],inplace=True)
    if city=='Wien':
          citylab='Vienna'
    else:
          citylab=city


    shap.summary_plot(shap_values.sort_index().to_numpy(), X.sort_index(),feature_names=X_disp,max_display=14,show=False)
    plt.title('Feature Influence for Trip Distance, ' + citylab + ', R2: ' + round(r2_model,3).astype(str))
    plt.xlabel("SHAP value (impact on distance, in m)")
    plt.savefig('../outputs/ML_Results/result_figures/dist_commute/' + city + '_FI_distance.png',facecolor='w',dpi=65,bbox_inches='tight')
    plt.close()
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([X_disp, shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)

    n=importance_df[:10].index

    X.sort_index(inplace=True)
    data=X.sort_index().iloc[:,n]
    data.columns=data.columns.str.replace("FeatureD_", "")
    values=shap_values.sort_index().iloc[:,n]
    #X_disp=[re.sub('FeatureD_','', x) for x in X.sort_index().columns]
    col_dict= {'DistCenter_res':'Dist. to city center','DistSubcenter_res':'Dist. to subenter', 'UrbPopDensity_res':'Population density','UrbBuildDensity_res':'Built-up density',
        'IntersecDensity_res':'Intersection density','LU_Comm_res':'Commercial area','LU_UrbFab_res':'Urban Fabric area','street_length_res':'Avg. street length','bike_lane_share_res':'Cycle lanes',
        'Trip_Time_Evening':'Evening trip','Trip_Time_AM_Rush':'Morning trip','Trip_Time_Nighttime Off-Peak':'Nighttime trip','Trip_Time_Lunch':'Lunchtime trip',
        'Season_Winter':'Winter season',
        'Age':'Age','Sex':'Sex','HHSize':'Household size',
        'Education_University':'University education', 'Occupation_Employed_FullTime':'Employed'}
    data.rename(columns=col_dict,inplace=True)
    #X_lab=[*map(col_dict.get, X_disp)]


    xl=[]
    yl=[]
    y0=[]

    for i in range(len(n)):
            dftemp=pd.DataFrame({'d':data.iloc[:,i],'v':values.iloc[:,i]})
            dftemp=dftemp.groupby('d')['v'].mean().reset_index()
            dftemp['v0']=0
            xl.append(dftemp['d'].values)
            yl.append(dftemp['v'].values)
            y0.append(dftemp['v0'].values)

    fig = plt.figure(figsize=(11,12))
    if city == 'Berlin': let='a'
    if city == 'Paris': let='b'
    if city == 'Madrid': let='c'
    if city == 'Wien': let='d'
    if city == 'Germany_other': let='e'
    if city == 'France_other': let='f'
    else: let = ''
    for i in range(0,6):
            ax1 = fig.add_subplot(321+i)
            xs=data.iloc[:,i]
            ys=values.iloc[:,i]
            x=xl[i]
            y1=y0[i]
            y2=yl[i]
            xlab=data.columns[i]
            #xlab=X_lab[i]

            ax1.scatter(xs,ys,alpha=0.9,s=8)
            plt.plot(x,y1,'k:')
            #plt.plot(x,y2,'k',label='mean')
            #plt.legend(loc="upper left",prop={'size':12})
            if i%2==0:
                    ax1.set_ylabel('SHAP value (m)',size=13)
            else:
                    ax1.set_ylabel('')
            ax1.set_xlabel(xlab,size=13)

            ax2 = ax1.twinx() 
            if len(xs.unique())==2:
                    ax2.hist(xs,bins=[-0.5,0.5,1.5], align='mid',color='gray',alpha=0.25)
                    ax2.set_xticks([-.5,0,0.5,1,1.5])
            else:
                    ax2.hist(xs,bins=30,color='gray',alpha=0.15)
                    ax2.set_ylim(0,len(data))
            ax2.set_yticks([])
    plt.suptitle(let + ") SHAP values for most important UF features, " + citylab,y=0.92,size=16)
    plt.savefig('../outputs/ML_Results/result_figures/dist_commute/' + city + '_main.png',facecolor='w',dpi=65,bbox_inches='tight')
    plt.close()

    # save shap_values, to enable later re-creation and editing of shap plots
    with open('../outputs/ML_Results/shap/dist_commute/' + city + '.pkl', 'wb') as f:
            pickle.dump(shap_values, f)

    with open('../outputs/ML_Results/shap/dist_commute/' + city + '_importance.pkl', 'wb') as g:
            pickle.dump(importance_df, g)

    with open('../outputs/ML_Results/shap/dist_agg/' + city + '_df.pkl', 'wb') as h:
        pickle.dump(df, h)

#cities=pd.Series(cities_all)
cities=pd.Series(['Berlin'])
cities.apply(dist_commute)