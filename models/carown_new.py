# load in required packages
import numpy as np
import pandas as pd
import geopandas as gpd
import shap
import re
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, cross_validate, GroupKFold, StratifiedGroupKFold, RepeatedKFold, StratifiedKFold, GridSearchCV, KFold
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import pickle
import os

cities_all=['Berlin','Dresden','Düsseldorf','Frankfurt am Main','Kassel','Leipzig','Magdeburg','Potsdam','Clermont','Dijon','Lille','Lyon','Montpellier','Nantes','Nimes','Paris','Toulouse','Madrid','Wien','France_other','Germany_other']
countries=['Germany','Germany','Germany','Germany','Germany','Germany','Germany','Germany','France','France','France','France','France','France','France','France','France','Spain','Austria','France','Germany']

def carown_model(city):
    country=countries[cities_all.index(city)]
    print(city, country)

    if city=='Germany_other':
        city0='Dresden'
        df0=pd.read_csv('../outputs/Combined/' + city0 + '_co_UF.csv')
        print(len(df0.columns), 'columns in the data for ', city0)
        df0['City']=city0
        df_all=df0.copy()

        cities0=['Leipzig','Magdeburg','Potsdam','Frankfurt am Main','Düsseldorf','Kassel']
        for city1 in cities0:
            print(city1)
            df1=pd.read_csv('../outputs/Combined/' + city1 + '_co_UF.csv')
            print(len(df1.columns), 'columns in the data for ', city1)
            df1['City']=city1
            if len(df1.columns==df_all.columns):
                df_all=pd.concat([df_all,df1])
                print(city1, 'added.')
                print(len(df_all), 'rows in the combined dataframe')
        df_all['HHNR']=df_all['City']+'_'+df_all['HHNR'].astype(int).astype(str)

        df=df_all.copy()
    elif city=='France_other':
        city0='Clermont'
        df0=pd.read_csv('../outputs/Combined/' + city0 + '_co_UF.csv')
        print(len(df0.columns), 'columns in the data for ', city0)
        df0['City']=city0
        df_all=df0.copy()

        cities0=['Toulouse']
        for city1 in cities0:
            print(city1)
            df1=pd.read_csv('../outputs/Combined/' + city1 + '_co_UF.csv')
            print(len(df1.columns), 'columns in the data for ', city1)
            df1['City']=city1
            if len(df1.columns==df_all.columns):
                df_all=pd.concat([df_all,df1])
                print(city1, 'added.')
                print(len(df_all), 'rows in the combined dataframe')
        df_all['HHNR']=df_all['City']+'_'+df_all['HHNR'].astype(int).astype(str)
        df=df_all.copy()
    else: df=pd.read_csv('../outputs/Combined/' + city + '_co_UF.csv')

    df=df.loc[:,( 'HHNR','Res_geocode',#'Dist_group', # IDs, trip geocodes, home-Res_geocode
    'HHSize','IncomeDetailed_Numeric','HHType_simp','maxAgeHH',# household details, omit  'IncomeDetailed', 'HHType', 
    'UniversityEducation', 'InEmployment', 'AllRetired', # personal-based details
    'UrbPopDensity', 'UrbBuildDensity','DistSubcenter', 'DistCenter', 'transit_accessibility',
    'bike_lane_share', 'IntersecDensity', 'street_length', 'LU_UrbFab', 'LU_Comm',# 'LU_Urban',
    # target: car ownership
    'CarOwnershipHH')
    ]

    df.loc[df['HHType_simp'].isin(['Single_Female_Parent','Single_Male_Parent']),'HHType_simp']='Single_Parent'
    df=df.loc[df['UrbPopDensity']<80000,]   
    # remove high building density outliers (For Leipzig)
    df=df.loc[df['UrbBuildDensity']<1e8,]   
    df=df.loc[df['maxAgeHH']>0,]  
    df.drop(columns='HHType_simp',inplace=True)
    df.dropna(inplace=True)

    if city in ['Berlin','Paris']:
        df.drop(columns=['UrbBuildDensity'],inplace=True)
        form_str="CarOwnershipHH ~ FeatureO_HHSize + FeatureO_IncomeDetailed_Numeric + FeatureO_maxAgeHH  + FeatureO_UniversityEducation + FeatureO_InEmployment + FeatureO_AllRetired + FeatureO_UrbPopDensity + FeatureO_DistSubcenter +  FeatureO_DistCenter + FeatureO_bike_lane_share + FeatureO_IntersecDensity +  FeatureO_street_length +  FeatureO_LU_UrbFab +  FeatureO_LU_Comm + FeatureO_transit_accessibility"
    elif city in ['France_other']:
           df.drop(columns=['UrbBuildDensity','transit_accessibility','IntersecDensity'],inplace=True)
           form_str="CarOwnershipHH ~ FeatureO_HHSize + FeatureO_IncomeDetailed_Numeric + FeatureO_maxAgeHH  + FeatureO_UniversityEducation + FeatureO_InEmployment + FeatureO_AllRetired + FeatureO_UrbPopDensity + FeatureO_DistSubcenter +  FeatureO_DistCenter + FeatureO_bike_lane_share +  FeatureO_street_length +  FeatureO_LU_UrbFab +  FeatureO_LU_Comm"
    elif city =='Germany_other':
        df.drop(columns=['LU_UrbFab'],inplace=True)
        form_str="CarOwnershipHH ~ FeatureO_HHSize + FeatureO_IncomeDetailed_Numeric + FeatureO_maxAgeHH  + FeatureO_UniversityEducation + FeatureO_InEmployment + FeatureO_AllRetired + FeatureO_UrbPopDensity +  FeatureO_UrbBuildDensity  + FeatureO_DistSubcenter +  FeatureO_DistCenter + FeatureO_bike_lane_share + FeatureO_IntersecDensity +  FeatureO_street_length +  FeatureO_LU_Comm + FeatureO_transit_accessibility"
    else:
        form_str="CarOwnershipHH ~ FeatureO_HHSize + FeatureO_IncomeDetailed_Numeric + FeatureO_maxAgeHH  + FeatureO_UniversityEducation + FeatureO_InEmployment + FeatureO_AllRetired + FeatureO_UrbPopDensity +  FeatureO_UrbBuildDensity  + FeatureO_DistSubcenter +  FeatureO_DistCenter + FeatureO_bike_lane_share + FeatureO_IntersecDensity +  FeatureO_street_length  + FeatureO_LU_UrbFab + FeatureO_LU_Comm + FeatureO_transit_accessibility"
    
    # identify the feature columns
    N_non_feature=2 # how many non-features are at the start of the df
    cols=df.columns
    newcols=(df.columns[:N_non_feature].tolist()) + ('FeatureO' +'_'+ cols[N_non_feature:-1]).tolist() + (df.columns[-1:].tolist())
    # change column names
    df.set_axis(newcols,axis=1,inplace=True)
    df = df.reset_index(drop=True)
    df0=df.copy()

    # convert  all categorical variables to dummies
    df_Cat=df.select_dtypes('object')[[col for col in df.select_dtypes('object').columns if "FeatureO" in col]]
    for col in df_Cat:
        dum=pd.get_dummies(df[[col]])
        df = pd.concat([df, dum], axis = 1)
        # remove the original categorical columns
    df.drop(df_Cat.columns.tolist(),axis=1,inplace=True)
    # HPO with full dataset, grouping by individual person
    target = 'CarOwnershipHH'
    N=len(df)
    # Define the parameter space to be considered
    PS = {"learning_rate": [0.1 ,0.15,0.2,0.3], 
                    "n_estimators": [100, 200, 300, 400],
                    "max_depth":[4, 5]}

    X=df[[col for col in df.columns if "FeatureO" in col]]
    y = df[target]

    tf=(X < 0).all(0)
    print(len(tf[tf]),' columns with value below zero')
    if len(tf[tf])>0:
        print(tf[tf].index.values)
        raise Exception("Some columns have values below zero")

    kf = list(KFold(n_splits=9,shuffle=True).split(X,y))

    # define grid search cross validator
    # if not already done!! #
    # if file '../ML_Results/' + city + '_HPO_carown_summary.csv' does not exist, run the HPO and create it. 
    # other wise pd.read_csv the file and extract variables ['LR','MD','N']
    fp='../outputs/ML_Results/'+city+'_HPO_carown_new_summary.csv'
    if os.path.isfile(fp):
        print('HPs already identified')
        HPO_summary=pd.read_csv(fp)
        n_parameter_all = HPO_summary['N'][0]
        lr_parameter_all = HPO_summary['LR'][0]
        md_parameter_all = HPO_summary['MD'][0]
    else:
        tuning_all = GridSearchCV(estimator=XGBClassifier(verbosity=0,use_label_encoder=False), param_grid=PS, cv=kf, scoring="f1_weighted",return_train_score=True)
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
        r8=['kf_gridSearch','full','9splits',tuning_all.best_params_['learning_rate'],tuning_all.best_params_['max_depth'],tuning_all.best_params_['n_estimators'],round(tuning_all.best_score_,3),round(cv_res_all['std_test_score'][tuning_all.best_index_],3),N] #

        # also include other results lists here if HPO is done for more than one cv type or sample
        HPO_summary=pd.DataFrame([r8],columns=['CV_Type','Sample','CV_params','LR','MD','N','F1_best','SD_best','N_obs']) # the last element in this case is the sd of f1 scores in the fold which produced best results

    # now redo the CV and calculate the SHAP values with the best HPs
    cv = KFold(n_splits=9,shuffle=True, random_state=2)

    y_predict = pd.DataFrame()
    y_predict2 = pd.DataFrame()
    y_test = pd.DataFrame()
    y_test2 = pd.DataFrame()

    summ_table_list=[]

    shap_values= pd.DataFrame()

    model = XGBClassifier(
        max_depth=md_parameter_all, 
        n_estimators=n_parameter_all, 
        learning_rate=lr_parameter_all)
    
    #form_str="CarOwnershipHH ~ FeatureO_HHSize + FeatureO_IncomeDetailed_Numeric + FeatureO_HHType_simp + FeatureO_maxAgeHH  + FeatureO_UniversityEducation + FeatureO_InEmployment + FeatureO_AllRetired + FeatureO_UrbPopDensity +  FeatureO_UrbBuildDensity  + FeatureO_DistSubcenter +  FeatureO_DistCenter + FeatureO_bike_lane_share + FeatureO_IntersecDensity +  FeatureO_street_length +  FeatureO_LU_UrbFab +  FeatureO_LU_Comm + FeatureO_transit_accessibility"
    writer = pd.ExcelWriter('../outputs/ML_Results/carown_LR_new/'  + city + '.xlsx', engine='openpyxl')
    for train_idx, test_idx in cv.split(X): # select here 
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        df_train, df_test = df0.iloc[train_idx], df0.iloc[test_idx]
        id=test_idx[0].astype(str)
        print('id',id)
        y_test_fold2=df_test['CarOwnershipHH']

        # train & predict
        model.fit(X_train, y_train, verbose=False, eval_set=[(X_train, y_train), (X_test, y_test_fold)])
        y_predict_fold = pd.Series(model.predict(X_test), index=X_test.index)

        # explain
        explainer = shap.TreeExplainer(model)
        shap_values_fold = explainer.shap_values(X_test)

        shap_values_fold = pd.DataFrame(shap_values_fold, index=X_test.index, columns=X.columns)     

        y_predict = pd.concat([y_predict, y_predict_fold], axis=0)
        y_test = pd.concat([y_test, y_test_fold], axis=0)

        shap_values = pd.concat([shap_values, shap_values_fold], axis=0)

        try:
            log_reg = smf.logit(form_str, data=df_train).fit()
            yhat=np.asarray(round(log_reg.predict(df_test.drop(columns='CarOwnershipHH'))))
            y_predict_fold2 = pd.Series(yhat, index=df_test.index)
            y_predict2 = pd.concat([y_predict2, y_predict_fold2], axis=0)
            y_test2 = pd.concat([y_test2, y_test_fold2], axis=0)

            coeff=log_reg.params.reset_index()
            coeff.rename(columns={'index':'param',0:'coefficient'},inplace=True)

            pval=log_reg.pvalues.reset_index()
            pval.rename(columns={'index':'param',0:'p'},inplace=True)

            summ_table=pd.concat([coeff,pval['p']],axis=1)
            summ_table['param']=summ_table['param'].str.replace('FeatureO_','')

            st_list_fold=[summ_table.drop(columns='param').to_numpy()]
            summ_table_list.append(st_list_fold)

            summ_table.to_excel(writer, sheet_name='summ' + id,index=False)
        except Exception as err:
            print('Logit Model Error')
            print(type(err))
            print(err)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    writer.close()

    mdarray=np.array(summ_table_list).squeeze()
    means=np.nanmean(mdarray,axis=0)
    means_df=pd.DataFrame(data=np.hstack((np.reshape(summ_table['param'].to_numpy(),(len(summ_table),1)),means)),columns=summ_table.columns.values)
    means_df.to_csv('../outputs/ML_Results/carown_LR_new/'  + city + '_mean.csv',index=False)

    y_test = y_test.squeeze(axis=1)
    y_test2 = y_test2.squeeze(axis=1)
    y_predict = y_predict.squeeze(axis=1)
    y_predict2 = y_predict2.squeeze(axis=1)
    f1_model_ML = metrics.f1_score(y_test, y_predict, average='weighted')
    f1_model_LR = metrics.f1_score(y_test2, y_predict2, average='weighted')
    print('Model f1, ML: ' + city)
    print(f1_model_ML)
    print('Model f1, LR: ' + city)
    print(f1_model_LR)

    HPO_summary['F1_full_ML']=f1_model_ML
    HPO_summary['F1_full_LR']=f1_model_LR
    HPO_summary['City']=city
    HPO_summary.to_csv('../outputs/ML_Results/' + city + '_HPO_carown_new_summary.csv',index=False)

    # save shap_values, to enable later re-creation and editing of shap plots
    with open('../outputs/ML_Results/shap/carown_new/' + city + '.pkl', 'wb') as f:
        pickle.dump(shap_values, f)

    # optionally here, check which variables are more important than random noise, then downselect X to those variables, and go back to HPO (or CV) and run once more from there.
    X_disp=[re.sub('FeatureO_','', x) for x in X.columns]

    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([X_disp, shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)

    with open('../outputs/ML_Results/shap/carown_new/' + city + '_importance.pkl', 'wb') as f:
        pickle.dump(importance_df, f)

    shap.summary_plot(shap_values.sort_index().to_numpy(), X.sort_index(),feature_names=X_disp,show=False)
    plt.title('Overall Feature Influence Car Ownership ' + city)
    plt.savefig('../outputs/ML_Results/result_figures/carown_new/' + city + '_FI_carown.png',facecolor='w',dpi=65,bbox_inches='tight')
    plt.close()

    cm = metrics.confusion_matrix(y_test, y_predict,normalize='true')
    # note the high confusion between bike/foot trips, and the high number of transit trips labelled as bike or car, and the high number of bike trips labelled as foot or car
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion matrix, car ownership, ' + city + '. F1: ' + str(round(f1_model_ML,3)))
    plt.savefig('../outputs/ML_Results/result_figures/carown_new/' + city + '_carown_CM.png',facecolor='w',dpi=65,bbox_inches='tight')
    plt.close() 

    col_dict= {'DistCenter':'Dist. to city center','IntersecDensity':'Instersection density','street_length':'Avg. street length',
               'UniversityEducation':'University education','AllRetired':'Retired','bike_lane_share':'Cycle lane share',
               'transit_accessibility':'Transit Accessibility',
       'UrbBuildDensity':'Built-up density','UrbPopDensity':'Population density', 'DistSubcenter':'Dist. to subcenter',
       'LU_Urban':'Urban area','LU_UrbFab':'Urban fabric area','LU_Comm':'Commercial area',
       'IncomeDetailed_Numeric':'Income','HHSize':'Household size','maxAgeHH':'Max householder age','InEmployment':'Employed',
       'HHType_simp_Single_Female':'Single female household','HHType_simp_MultiAdult':'Multi-adult household','HHType_simp_MultiAdult_Kids':'Multi-adult household with kids'}
    X_lab=[*map(col_dict.get, X_disp)]

    shap.summary_plot(shap_values.sort_index().to_numpy(), X.sort_index(),feature_names=X_lab,max_display=10,show=False)
    plt.title('Overall Feature Influence Car Ownership ' + city, size=14)
    plt.xlabel("SHAP (probability of car ownership)", size=12)
    plt.savefig('../outputs/ML_Results/result_figures/carown_new/' + city + '_FI_small.png',facecolor='w',dpi=65,bbox_inches='tight')
    plt.close() 
    
    n=importance_df[:10].index
    shap_values.sort_index(inplace=True)
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

    citylab=city
    if city == 'Germany_other': citylab='Germany, other'
    if city == 'France_other': citylab='France, other'

    lab_dict= {'DistCenter':'Dist. to city center (km)','IntersecDensity':'Instersection density','street_length':'Avg. street length','bike_lane_share':'Cycle lane share',
       'UrbBuildDensity':'Built-up density','UrbPopDensity':'Population density', 'DistSubcenter':'Dist. to subcenter','LU_Urban':'Urban area','LU_UrbFab':'Urban fabric area',
       'transit_accessibility':'Transit Accessibility','HHType_simp_Single_Female':'Single female household','UniversityEducation':'University education',
       'IncomeDetailed_Numeric':'Income','HHSize':'Household size','maxAgeHH':'Max householder age','InEmployment':'Employed'}

    importance_df['column_label']=importance_df['column_name'].map(lab_dict)
    
    if 'FeatureO_DistCenter' in data.columns:
        fig = plt.figure()
        i=data.columns.get_loc('FeatureO_DistCenter')
        ax1 = fig.add_subplot(111) 
        xs=data.iloc[:,i]
        ys=values.iloc[:,i]

        x=xl[i]
        y1=y0[i]
        y2=yl[i]
        xlab=importance_df['column_label'].iloc[i]

        ax1.scatter(xs,ys,alpha=0.3,s=8)
        plt.plot(x,y1,'k:',label='zero')
        plt.plot(x,y2,'k',label='mean')
        #plt.legend(loc="upper left",prop={'size':12})
        if i%2==0:
            ax1.set_ylabel('SHAP value',size=13)
        else:
            ax1.set_ylabel('')
        ax1.set_xlabel(xlab,size=14)


        ax2 = ax1.twinx() 
        if len(xs.unique())==2:
            ax2.hist(xs,bins=[-0.5,0.5,1.5], align='mid',color='gray',alpha=0.15)
            ax2.set_xticks([-.5,0,0.5,1,1.5])
        else:
            ax2.hist(xs,bins=30,color='gray',alpha=0.15)
            ax2.set_ylim(0,len(data))
        ax2.set_yticks([])
        

        if city == 'Berlin': let='a'
        elif city == 'Paris': let='b'
        # elif city == 'Madrid': let='c'
        # elif city == 'Wien': let='d'
        elif city == 'Germany_other': let='c'
        elif city == 'France_other': let='d'
        else: let='0'

        plt.title(let + ') ' + citylab,fontsize=16)
        ax1.set_ylabel('SHAP (car ownership probability)',size=13)
        plt.savefig('../outputs/ML_Results/result_figures/carown_new/' + city + '_d2c.png',facecolor='w',dpi=65,bbox_inches='tight')
        plt.close() 

    fig = plt.figure(figsize=(11,11))
    i=0
    for i in range(0,6):
        ax1 = fig.add_subplot(321+i)
        xs=data.iloc[:,i]
        ys=values.iloc[:,i]
        x=xl[i]
        y1=y0[i]
        y2=yl[i]
        xlab=importance_df['column_label'].iloc[i]

        ax1.scatter(xs,ys,alpha=0.3,s=8)
        plt.plot(x,y1,'k:',label='zero')
        plt.plot(x,y2,'k',label='mean')
        #plt.legend(loc="upper left",prop={'size':12})
        if i%2==0:
            ax1.set_ylabel('SHAP (car own prob.)',size=13)
        else:
            ax1.set_ylabel('')
        ax1.set_xlabel(xlab,size=13)

        ax2 = ax1.twinx() 
        if len(xs.unique())==2:
            ax2.hist(xs,bins=[-0.5,0.5,1.5], align='mid',color='gray',alpha=0.15)
            ax2.set_xticks([-.5,0,0.5,1,1.5])
        else:
            ax2.hist(xs,bins=30,color='gray',alpha=0.15)
            ax2.set_ylim(0,len(data))
        ax2.set_yticks([])
    plt.suptitle("SHAP and feature values for features influencing car ownership, " + city.replace('_',', '),y=0.92,size=16)
    plt.savefig('../outputs/ML_Results/result_figures/carown_new/' + city + '_FI_detail6.png',facecolor='w',dpi=65,bbox_inches='tight')
    plt.close() 

#cities=pd.Series(['Berlin','Dresden','Düsseldorf','Frankfurt am Main','Kassel','Leipzig','Magdeburg','Potsdam','Clermont','Paris','Toulouse','France_other','Germany_other'])
cities=pd.Series(['France_other','Germany_other'])

cities.apply(carown_model) # args refers to the size threshold above which to divide large units into their smaller sub-components, e.g. 10km2