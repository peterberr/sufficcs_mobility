# script to model avergage total daily travel in all cities
# last update Peter Berrill June 21 2023

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

def dist_daily(city):
    country=countries[cities_all.index(city)]
    print(city, country)
    if city=='Germany_other':
        city0='Dresden'
        df0=pd.read_csv('../outputs/Combined/' + city0 + '_UF.csv')
        df0['Commute_Trip']=0
        df0.loc[df0['Trip_Purpose_Agg']=='Home↔Work','Commute_Trip']=1
        df0.loc[(df0['Training'].isin(['Apprenticeship/Business','Craftsman/Technical'])) & (df0['Education']!='University'),'Education']='Apprenticeship'

        commute_share=pd.DataFrame(df0.groupby(['HH_PNR'])['Commute_Trip'].mean()).reset_index()
        categ=df0[['HH_PNR','Season','Age','Sex','Occupation','Education','HHSize']].drop_duplicates()
        df_trip_daily=pd.DataFrame(df0.groupby(['Res_geocode','HH_PNR'])['Trip_Distance'].sum())
        df_trip_daily.reset_index(inplace=True,drop=False)
        df_UF_Features=df0.loc[:,['Res_geocode', 
                        'DistSubcenter_res', 'DistCenter_res',
                        'UrbPopDensity_res', 'UrbBuildDensity_res',
                        'IntersecDensity_res', 'street_length_res', 'LU_UrbFab_res','bike_lane_share_res',
                        'LU_Comm_res',]].groupby('Res_geocode').mean().drop_duplicates() 
        df_UF_Features.reset_index(inplace=True)
        df_UF=df_UF_Features.merge(df_trip_daily)
        df_UF=df_UF.merge(categ)
        df_UF=df_UF.merge(commute_share)
        df_UF=pd.concat([df_UF.drop(columns='Trip_Distance'),df_UF['Trip_Distance']],axis=1)

        df_UF['City']=city0
        df_all=df_UF.copy()
        del df_UF

        cities0=['Leipzig','Magdeburg','Potsdam','Frankfurt am Main','Düsseldorf','Kassel']
        for city1 in cities0:
                print(city1)
                df1=pd.read_csv('../outputs/Combined/' + city1 + '_UF.csv')
                df1['Commute_Trip']=0
                df1.loc[df1['Trip_Purpose_Agg']=='Home↔Work','Commute_Trip']=1
                df1.loc[(df1['Training'].isin(['Apprenticeship/Business','Craftsman/Technical'])) & (df1['Education']!='University'),'Education']='Apprenticeship'

                commute_share=pd.DataFrame(df1.groupby(['HH_PNR'])['Commute_Trip'].mean()).reset_index()
                categ=df1[['HH_PNR','Season','Age','Sex','Occupation','Education','HHSize']].drop_duplicates()
                df_trip_daily=pd.DataFrame(df1.groupby(['Res_geocode','HH_PNR'])['Trip_Distance'].sum())
                df_trip_daily.reset_index(inplace=True,drop=False)
                df_UF_Features=df1.loc[:,['Res_geocode', 
                                'DistSubcenter_res', 'DistCenter_res',
                                'UrbPopDensity_res', 'UrbBuildDensity_res',
                                'IntersecDensity_res', 'street_length_res', 'LU_UrbFab_res','bike_lane_share_res',
                                'LU_Comm_res',]].groupby('Res_geocode').mean().drop_duplicates() 
                df_UF_Features.reset_index(inplace=True)
                df_UF=df_UF_Features.merge(df_trip_daily)
                df_UF=df_UF.merge(categ)
                df_UF=df_UF.merge(commute_share)
                df_UF=pd.concat([df_UF.drop(columns='Trip_Distance'),df_UF['Trip_Distance']],axis=1)

                df_UF['City']=city1
                if len(df_UF.columns==df_all.columns):
                       df_all=pd.concat([df_all,df_UF])
                       print(city1, 'added.')
                       print(len(df_all), 'rows in the combined dataframe')
        df_UF=df_all.copy()

    elif city=='France_other':
        city0='Clermont'
        df0=pd.read_csv('../outputs/Combined/' + city0 + '_UF.csv')
        df0['Commute_Trip']=0
        df0.loc[df0['Trip_Purpose_Agg']=='Home↔Work','Commute_Trip']=1

        commute_share=pd.DataFrame(df0.groupby(['HH_PNR'])['Commute_Trip'].mean()).reset_index()
        categ=df0[['HH_PNR','Season','Age','Sex','Occupation','Education','HHSize']].drop_duplicates()
        df_trip_daily=pd.DataFrame(df0.groupby(['Res_geocode','HH_PNR'])['Trip_Distance'].sum())
        df_trip_daily.reset_index(inplace=True,drop=False)
        df_UF_Features=df0.loc[:,['Res_geocode', 
                        'DistSubcenter_res', 'DistCenter_res',
                        'UrbPopDensity_res', 'UrbBuildDensity_res',
                        'IntersecDensity_res', 'street_length_res', 'LU_UrbFab_res','bike_lane_share_res',
                        'LU_Comm_res',]].groupby('Res_geocode').mean().drop_duplicates() 
        df_UF_Features.reset_index(inplace=True)
        df_UF=df_UF_Features.merge(df_trip_daily)
        df_UF=df_UF.merge(categ)
        df_UF=df_UF.merge(commute_share)
        df_UF=pd.concat([df_UF.drop(columns='Trip_Distance'),df_UF['Trip_Distance']],axis=1)

        df_UF['City']=city0
        df_all=df_UF.copy()
        del df_UF

        cities0=['Toulouse','Montpellier','Lyon','Nantes','Nimes','Lille','Dijon']
        for city1 in cities0:
                print(city1)
                df1=pd.read_csv('../outputs/Combined/' + city1 + '_UF.csv')
                df1['Commute_Trip']=0
                df1.loc[df1['Trip_Purpose_Agg']=='Home↔Work','Commute_Trip']=1
                commute_share=pd.DataFrame(df1.groupby(['HH_PNR'])['Commute_Trip'].mean()).reset_index()
                categ=df1[['HH_PNR','Season','Age','Sex','Occupation','Education','HHSize']].drop_duplicates()
                df_trip_daily=pd.DataFrame(df1.groupby(['Res_geocode','HH_PNR'])['Trip_Distance'].sum())
                df_trip_daily.reset_index(inplace=True,drop=False)
                df_UF_Features=df1.loc[:,['Res_geocode', 
                                'DistSubcenter_res', 'DistCenter_res',
                                'UrbPopDensity_res', 'UrbBuildDensity_res',
                                'IntersecDensity_res', 'street_length_res', 'LU_UrbFab_res','bike_lane_share_res',
                                'LU_Comm_res',]].groupby('Res_geocode').mean().drop_duplicates() 
                df_UF_Features.reset_index(inplace=True)
                df_UF=df_UF_Features.merge(df_trip_daily)
                df_UF=df_UF.merge(categ)
                df_UF=df_UF.merge(commute_share)
                df_UF=pd.concat([df_UF.drop(columns='Trip_Distance'),df_UF['Trip_Distance']],axis=1)

                df_UF['City']=city1
                if len(df_UF.columns==df_all.columns):
                       df_all=pd.concat([df_all,df_UF])
                       print(city1, 'added.')
                       print(len(df_all), 'rows in the combined dataframe')
        df_UF=df_all.copy()
    else:
            df=pd.read_csv('../outputs/Combined/' + city + '_UF.csv',dtype={'Ori_geocode': str, 'Des_geocode': str,'Res_geocode': str })
            df['Commute_Trip']=0
            df.loc[df['Trip_Purpose_Agg']=='Home↔Work','Commute_Trip']=1
            if country=='Germany':
                   df.loc[(df['Training'].isin(['Apprenticeship/Business','Craftsman/Technical'])) & (df['Education']!='University'),'Education']='Apprenticeship'

            commute_share=pd.DataFrame(df.groupby(['HH_PNR'])['Commute_Trip'].mean()).reset_index()
            categ=df[['HH_PNR','Season','Age','Sex','Occupation','Education','HHSize']].drop_duplicates()
            df_trip_daily=pd.DataFrame(df.groupby(['Res_geocode','HH_PNR'])['Trip_Distance'].sum())
            df_trip_daily.reset_index(inplace=True,drop=False)
            df_UF_Features=df.loc[:,['Res_geocode', 
                            'DistSubcenter_res', 'DistCenter_res',
                            'UrbPopDensity_res', 'UrbBuildDensity_res',
                            'IntersecDensity_res', 'street_length_res', 'LU_UrbFab_res','bike_lane_share_res',
                            'LU_Comm_res',]].groupby('Res_geocode').mean().drop_duplicates() 
            df_UF_Features.reset_index(inplace=True)
            df_UF=df_UF_Features.merge(df_trip_daily)
            df_UF=df_UF.merge(categ)
            df_UF=df_UF.merge(commute_share)
            df_UF=pd.concat([df_UF.drop(columns='Trip_Distance'),df_UF['Trip_Distance']],axis=1)
            df_UF['City']=city

    #commute_exists=commute_share.copy()
    df_UF['Commute_Trip']=np.ceil(df_UF['Commute_Trip'])
    df_UF.reset_index(inplace=True,drop=True)
    Occ_dict={'Employed_FullTime':'Employed','Employed_PartTime':'Employed','Employed':'Employed','Trainee':'Employed',
                'Student_School':'Student_School','Student_3rdLevel':'Student_3rdLevel','Pre-School':'Pre-School','Retired':'Retired',
                'Unemployed':'Unemployed/Other','Other':'Unemployed/Other','Home_Partner':'Unemployed/Other'}
    Edu_dict={'University':'University','Secondary':'Secondary','Secondary+BAC':'Secondary','Secondary+Matura':'Secondary',
                'Apprenticeship':'Apprenticeship',
                'Elementary':'Primary/None','Pre-School':'Primary/None','No diploma yet':'Primary/None','Unknown':'Primary/None','Other':'Primary/None'}
    
    df_UF['Sex']=df_UF['Sex']-1 # change from [1,2] to [0,1], for plotting purposes
    df_UF['Occupation']=df_UF['Occupation'].map(Occ_dict)
    df_UF['Education']=df_UF['Education'].map(Edu_dict)
    df_UF.drop(columns=['HH_PNR','City'],inplace=True)
    df_agg=df_UF.copy()
    N=len(df_UF)

    df_agg.sort_values(by='Res_geocode',inplace=True)
    df_agg.dropna(subset=['Trip_Distance'],inplace=True)

    target='Trip_Distance'

    N_non_feature=1 # how many non-features are at the start of the df
    cols=df_agg.columns
    newcols=(df_agg.columns[:N_non_feature].tolist()) + ('FeatureD' +'_'+ cols[N_non_feature:-1]).tolist() + (df_agg.columns[-1:].tolist())
    # change column names
    df_agg.set_axis(newcols,axis=1,inplace=True)
    df_agg = df_agg.reset_index(drop=True)
    df0=df_agg.copy()

    # convert  all categorical variables to dummies
    df_Cat=df_agg.select_dtypes('object')[[col for col in df_agg.select_dtypes('object').columns if "FeatureD" in col]]
    for col in df_Cat:
        dum=pd.get_dummies(df_agg[[col]])
        df_agg = pd.concat([df_agg, dum], axis = 1)
        # remove the original categorical columns
    df_agg.drop(df_Cat.columns.tolist(),axis=1,inplace=True)


    X=df_agg.drop(columns=['Res_geocode',target])
    y=df_agg['Trip_Distance']
#     # for German cities with very small N postcodes, define a smaller number of splits for cross validation
#     if city in ['Potsdam','Magdeburg','Kassel']:
#         cv = RepeatedKFold(n_splits=3,n_repeats=10,random_state=2)
#     else:
#         cv = RepeatedKFold(n_splits=5,n_repeats=10,random_state=2
    cv = RepeatedKFold(n_splits=5,n_repeats=10,random_state=2)
    
    # Define the parameter space to be considered
    PS = {"learning_rate": [0.01, 0.05,0.1,0.2], 
                "n_estimators": [100,200, 300],
                "max_depth":[2,3,4]}
    
    fp='../outputs/ML_Results/'+city+'_HPO_dist_daily_summary.csv'
    if os.path.isfile(fp):
        print('HPs already identified')
        HPO_summary=pd.read_csv(fp)
        n_parameter_all = HPO_summary['N_est'][0]
        lr_parameter_all = HPO_summary['LR'][0]
        md_parameter_all = HPO_summary['MD'][0]
    else:
        tuning_all = GridSearchCV(estimator=XGBRegressor(verbosity=0), param_grid=PS, cv=cv, scoring="r2",return_train_score=True)
        tuning_all.fit(X,y)

        print('best hyper-parameters identified by HPO')
        print(tuning_all.best_params_)
        print('r2 with best hyper-parameters')
        print(tuning_all.best_score_)
        cv_res_all=tuning_all.cv_results_

        n_parameter_all = tuning_all.best_params_['n_estimators']
        lr_parameter_all = tuning_all.best_params_['learning_rate']
        md_parameter_all = tuning_all.best_params_['max_depth']

        # save results of HPO, and full model r2
        r8=['rkf_gridSearch','agg_postcode','5splits_10repeats',tuning_all.best_params_['learning_rate'],tuning_all.best_params_['max_depth'],tuning_all.best_params_['n_estimators'],round(tuning_all.best_score_,3),round(cv_res_all['std_test_score'][tuning_all.best_index_],3),N] #
        HPO_summary=pd.DataFrame([r8],columns=['CV_Type','Sample','CV_params','LR','MD','N_est','R2_best','SD_best','N_obs']) # the last element in this case is the sd of f1 scores in the fold which produced best results

    y_predict = pd.DataFrame()
    y_predict2 = pd.DataFrame()
    y_test = pd.DataFrame()
    y_test2 = pd.DataFrame()
    summ_table_list=[]

    shap_values= pd.DataFrame() 

    model = XGBRegressor(
    max_depth=md_parameter_all, 
    n_estimators=n_parameter_all, 
    learning_rate=lr_parameter_all)

    # if doing the repeated cv and creation of shap values
    r2ml=[]
    r2lr=[]

    form_str="Trip_Distance ~  FeatureD_HHSize + FeatureD_Sex + FeatureD_Education + FeatureD_Occupation + FeatureD_Age + FeatureD_Season + FeatureD_Commute_Trip + FeatureD_DistCenter_res + FeatureD_DistSubcenter_res  + FeatureD_UrbPopDensity_res + FeatureD_UrbBuildDensity_res  + FeatureD_IntersecDensity_res + FeatureD_street_length_res + FeatureD_LU_Comm_res +  FeatureD_LU_UrbFab_res + FeatureD_bike_lane_share_res"

    writer = pd.ExcelWriter('../outputs/ML_Results/dist_daily/'  + city + '.xlsx', engine='openpyxl')
    for train_idx, test_idx in cv.split(X): # select here 
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
            summ_table['param']=summ_table['param'].str.replace('FeatureO_','')

            st_list_fold=[summ_table.drop(columns='param').to_numpy()]
            summ_table_list.append(st_list_fold)

            summ_table.to_excel(writer, sheet_name='summ' + id,index=False)
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()
    
    mdarray=np.array(summ_table_list).squeeze()
    means=np.nanmean(mdarray,axis=0)
    means_df=pd.DataFrame(data=np.hstack((np.reshape(summ_table['param'].to_numpy(),(len(summ_table),1)),means)),columns=summ_table.columns.values)
    means_df.to_csv('../outputs/ML_Results/dist_daily/'  + city + '_mean.csv',index=False)

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
    HPO_summary.to_csv('../outputs/ML_Results/' + city + '_HPO_dist_daily_summary.csv',index=False)

    r2ml=pd.DataFrame(r2ml)
    r2ml.columns=['GBDT']
    r2ml['LR']=r2lr
    #r2ml.to_csv('../outputs/ML_Results/' + city + '_HPO_dist_daily_r2.csv',index=False)

    X_disp=[re.sub('FeatureD_','', x) for x in X.columns]

    shap_values=shap_values.sort_index()
    shap_values.reset_index(inplace=True)
    shap_values=shap_values.groupby('index').mean().reset_index()
    shap_values.drop(columns=['index'],inplace=True)

    shap.summary_plot(shap_values.sort_index().to_numpy(), X.sort_index(),feature_names=X_disp,max_display=14,show=False)
    plt.title('Feature Influence for Trip Distance, ' + city + ', R2: ' + round(r2_model,3).astype(str))
    plt.xlabel("SHAP value (impact on distance, in m)")
    plt.savefig('../outputs/ML_Results/result_figures/dist_daily/' + city + '_FI_distance.png',facecolor='w',dpi=65,bbox_inches='tight')
    plt.close()
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([X_disp, shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)

    n=importance_df[:10].index

    X.sort_index(inplace=True)
    data=X.sort_index().iloc[:,n]
    values=shap_values.sort_index().iloc[:,n]

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

    fig = plt.figure(figsize=(12,15))

    for i in range(0,8):
            ax1 = fig.add_subplot(421+i)
            xs=data.iloc[:,i]
            ys=values.iloc[:,i]
            x=xl[i]
            y1=y0[i]
            y2=yl[i]
            xlab=data.columns[i]

            #ax1.scatter(xs+np.random.normal(0, 0.05, len(data)),ys,alpha=0.9,s=8)
            ax1.scatter(xs,ys,alpha=0.9,s=8)
            plt.plot(x,y1,'k:',label='zero')
            #plt.plot(x,y2,'k',label='mean')
            plt.legend(loc="upper left",prop={'size':12})
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
    plt.suptitle("SHAP values for most important UF features, " + city,y=0.92,size=16)
    plt.savefig('../outputs/ML_Results/result_figures/dist_daily/' + city + '_main.png',facecolor='w',dpi=65,bbox_inches='tight')
    plt.close()

    # save shap_values, to enable later re-creation and editing of shap plots
    with open('../outputs/ML_Results/shap/dist_daily/' + city + '.pkl', 'wb') as f:
            pickle.dump(shap_values, f)

    with open('../outputs/ML_Results/shap/dist_daily/' + city + '_importance.pkl', 'wb') as g:
            pickle.dump(importance_df, g)
    
    with open('../outputs/ML_Results/shap/dist_daily/' + city + '_df.pkl', 'wb') as h:
            pickle.dump(df_agg, h)


cities=pd.Series(['France_other','Germany_other'])
#cities=pd.Series(cities_all)
cities.apply(dist_daily)