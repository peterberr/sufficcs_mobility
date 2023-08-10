import pickle

var_dict_H = {'Res_geocode':'PLZ','HHNR':'HHNR', # Variables related to administrative boundary units and identifiers
'HHSize':'V_ANZ_PERS','IncomeDetailed':'V_EINK', 'IncomeHarmonised':'V_EINK','IncomeDetailed_Numeric':'V_EINK','HHType':'E_HHTYP',  # household size and income
'MBikeAvailable':'V_ANZ_MOT125','MopScootAvailable':'V_ANZ_MOPMOT','CarOwnershipHH':'V_ANZ_PKW_PRIV','CompanyCarHH':'V_ANZ_PKW_DIENST', # ownership of transport vehicles, availability of car/bike is in the person file
'Time2Bus':'V_GEHZEIT_BUS_HH','Time2Tram':'V_GEHZEIT_STRAB_HH','Time2SBahn':'V_GEHZEIT_SBAHN_HH','Time2UBahn':'V_GEHZEIT_UBAHN_HH','Time2Train':'V_GEHZEIT_NFZUG_HH', # Variables related to proximity to transit
'HH_Weight':'GEWICHT_HH'}

value_dict_H={'Res_geocode': {}, 'HHNR': {}, 'HHSize':{},
'IncomeDetailed': {1:'Under500',2:'500-900',3:'900-1500',4:'1500-2000',5:'2000-2600',6:'2600-3000',7:'3000-3600',8:'3600-4600',9:'4600-5600',10:'Over5600'},
'IncomeHarmonised': {1:'Under1000',2:'Under1000',3:'1000-2000',4:'1000-2000',5:'2000-3000',6:'2000-3000',7:'3000-4500',8:'3000-4500',9:'Over4500',10:'Over4500'},
'IncomeDetailed_Numeric': {1:400,2:700,3:1200,4:1750,5:2300,6:2800,7:3300,8:4100,9:5100,10:6000},
'HHType': {1:'Multiperson_Kids',2:'Multiperson_NoKids',3:'SinglePerson_Under65',4:'SinglePerson_Over65'},
'MBikeAvailable': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1,-10:0},
'MopScootAvailable': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1,-10:0},
'CarOwnershipHH': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:0,-10:0},
'CompanyCarHH': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:0,-10:0},
'Time2Bus':{},'Time2Tram':{},'Time2SBahn':{},'Time2UBahn':{},'Time2Train':{},
'HH_Weight':{}
}

na_dict_H={'IncomeDetailed':'Unknown','IncomeHarmonised':'Unknown','HHType':'Unknown','MBikeAvailable':0,'MopScootAvailable':0,'CarOwnershipHH':0,'CompanyCarHH':0}

var_dict_P = {'HHNR':'HHNR','Person':'PNR', # Variables related to administrative boundary units and identifiers
'Age':'V_ALTER','Sex':'V_GESCHLECHT', # i don't see a 'relationship' variable in srv. the trip month and day coming from the trip file
'MobilityConstraints':'V_EINSCHR_NEIN','BikeAvailable':'V_RAD_VERFUEG', 'CarAvailable':'V_PKW_VERFUEG',
'Occupation':'V_ERW', 'Education':'V_SCHULAB','Training':'V_BAUSB',
'DrivingLicense':'V_FUEHR_PKW','TransitSubscription': 'E_OEV_FK', # every city has a more detailed variable for the transit subscription coded as 'V_OEV_FK_XXX'. # i don't see a work from home variable for srv surveys
# parking availability variables seem to only exist for Regensburg in Germany
# next variable checks if they participated in mobility on the day of the survey. not sure how to interpret this, but possible we need to remove all that did not participate. also unsure how it interacts with 'V_WOHNUNG' - left home. Probably after we drop all trips without a mode all will be well
'MobilityParticipation':'E_MOBIL', #
'Per_Weight':'GEWICHT_P'
}

value_dict_P={'HHNR': {}, 'Person': {}, # Variables related to administrative boundary units and identifiers
'Age': {}, 'Sex': {}, # relation to reference person, age, sex (1=M; 2=F)
'MobilityConstraints':{0:1,1:0},
'BikeAvailable': {1:1, 2:1,2:1,3:0,-8:0,-9:0,-10:0},
'CarAvailable': {1:1, 2:1,2:1,3:0,-8:0,-9:0,-10:0},
'Occupation':{1:'Other',2:'Home_Partner',3:'Retired',4:'Other',5:'Unemployed',6:'Student_School',7:'Student_3rdLevel',8:'Trainee',9:'Employed_FullTime',10:'Employed_PartTime',11:'Employed_PartTime',12:'Other',70:'Other',-10:'Other'}, # surprisingly low number of 3rd level students in Dresden
'Education':{1:'Elementary',2:'Secondary',3:'University',4:'No diploma yet'}, # there are some inconsitencies with the Education file which will be addressed in the combine script. apprenticeships are not mentioned in SRV education, although they would be covered in Training
'Training':{1:'Apprenticeship/Business',2:'Craftsman/Technical',3:'University',4:'None',-9:'None',-10:'None'},
'DrivingLicense':{1:1,2:0,-8:0,-10:0},
'TransitSubscription':{1:0,2:0,3:1,60:1,70:0,-8:0,-10:0},
'MobilityParticipation':{1:1,2:0,-7:0},
'Per_Weight':{}
}

na_dict_P={'Occupation':'Other','Education':'Unknown','Training':'None','BikeAvailable':0,'CarAvailable':0,
'DrivingLicense':0,'TransitSubscription':0}

var_dict_W={'HHNR':'HHNR','Person':'PNR','Trip':'WNR', # Variables related to administrative boundary units and identifiers
'Ori_Plz':'V_START_PLZ','Des_Plz':'V_ZIEL_PLZ','Trip_Valid':'E_WEG_GUELTIG',
'Date':'STICHTAG_DATUM', 'Day':'STICHTAG_WTAG','Time':'V_BEGINN_STUNDE','Minute':'V_BEGINN_MINUTE',
'Trip_Purpose':'E_QZG_17','Ori_Reason':'E_START_ZWECK','Des_Reason':'V_ZWECK',
'Trip_Purpose_Agg':'E_QZG_17',
'N_accomp_HH':'V_BEGLEITUNG_HH','N_others_Car':'V_F_ANZAHL','Mode_Detailed':'E_HVM','Ori_Reason_Detailed':'E_START_ZWECK','Des_Reason_Detailed':'V_ZWECK',
'Mode':'E_HVM','Trip_Distance':'V_LAENGE','Trip_Distance_GIS':'GIS_LAENGE','Trip_Distance_GIS_valid':'GIS_LAENGE_GUELTIG','Trip_Duration':'E_DAUER','Trip_Weight':'GEWICHT_W',
}

value_dict_W={'HHNR': {}, 'Person': {},'Trip': {}, # Variables related to administrative boundary units and identifiers
'Ori_Plz':{},'Des_Plz':{},'Trip_Valid':{-1:1,0:0},
'Date':{}, 'Day':{},'Time':{},'Minute':{},
'Trip_Purpose':{1:'Home-Work',2:'Home-Companion',3:'Home-School',4:'Home-Work',5:'Home-Shopping',6:'Home-Leisure',7:'Home-Other',8:'Work-Home',9:'Companion-Home',
10:'School-Home',11:'Work-Home',12:'Shopping-Home',13:'Leisure-Home',14:'Other-Home',15:'Other-Work',16:'Work-Other',17:'Other',-7:'Other'},
'Trip_Purpose_Agg':{1:'Home↔Work',2:'Home↔Companion',3:'Home↔School',4:'Home↔Work',5:'Home↔Shopping',6:'Home↔Leisure',7:'Other',8:'Home↔Work',9:'Home↔Companion',
10:'Home↔School',11:'Home↔Work',12:'Home↔Shopping',13:'Home↔Leisure',14:'Other',15:'Other',16:'Other',17:'Other',-7:'Other'},
'Ori_Reason':{1:'Work',2:'Other',3:'Kindergarten',4:'School',5:'School',6:'University/College',7:'University/College',8:'Shopping_Daily',9:'Shopping_Other',
10:'Service_Facility',11:'Companion',12:'Leisure_Culture',13:'Leisure_Food',14:'Leisure_Visit',15:'Leisure_Outdoors',16:'Leisure_Sport',17:'Leisure_Other',18:'Home',70:'Other',-7:'Other'},
'Des_Reason':{1:'Work',2:'Other',3:'Kindergarten',4:'School',5:'School',6:'University/College',7:'University/College',8:'Shopping_Daily',9:'Shopping_Other',
10:'Service_Facility',11:'Companion',12:'Leisure_Culture',13:'Leisure_Food',14:'Leisure_Visit',15:'Leisure_Outdoors',16:'Leisure_Sport',17:'Leisure_Other',18:'Home',70:'Other',-7:'Other'},
'N_accomp_HH':{},'N_others_Car':{},'Mode_Detailed':{},'Ori_Reason_Detailed':{},'Des_Reason_Detailed':{},
'Mode':{1:'Foot',2:'Bike',18:'Bike',19:'Bike',3:'2_3_Wheel',4:'Car',5:'Car',6:'Car',7:'Car',8:'Car',9:'Car',10:'Transit',11:'Transit',12:'Transit',13:'Transit',14:'Transit',15:'Transit',16:'Car'}, # 
'Trip_Distance':{},'Trip_Distance_GIS':{},'Trip_Distance_GIS_valid':{-1:1,0:0},'Trip_Duration':{},'Trip_Weight':{}
}

na_dict_W={'Trip_Purpose':'Other','Trip_Purpose_Agg':'Other','Ori_Reason':'Other','Des_Reason':'Other','Mode':'Other'}

# Combine variable, value, and na dictionaries
var_all = {'HH': var_dict_H,'P':var_dict_P,'W':var_dict_W}
value_all = {'HH': value_dict_H,'P':value_dict_P,'W':value_dict_W}
na_all = {'HH': na_dict_H,'P':na_dict_P,'W':na_dict_W}

# save combined dictionaries
with open('../dictionaries/Germany_var.pkl', 'wb') as f:
    pickle.dump(var_all, f)
with open('../dictionaries/Germany_val.pkl', 'wb') as f:
    pickle.dump(value_all, f)
with open('../dictionaries/Germany_na.pkl', 'wb') as f:
    pickle.dump(na_all, f)