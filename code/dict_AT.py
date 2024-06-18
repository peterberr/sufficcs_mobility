import pickle

var_dict_H = {'Res_geocode':'hh_wohngemeinde','HHNR':'hh_nr', # Variables related to administrative boundary units and identifiers
'HHSize':'hh_gr','N_Adult':'hh_grue18','N_6-17':'hh_gr617','N_<6':'hh_gru6','IncomeDescriptive':'hh_wirtschsituation','IncomeDescriptiveNumeric':'hh_wirtschsituation',  # household size and income
'BikeAvailable':'fzg_radges','EBikeAvailable':'fzg_e_rad','2_3WAvailable':'fzg_mot','CarAvailable':'fzg_pkw','CarOwnershipHH':'fzg_pkw','CarOwnershipHH_num':'fzg_pkw',  # ownership of transport vehicles, including electric bikes
'Time2Transit':'hh_oev_entf', # Variables related to proximity to transit
'HH_Weight':'hh_hochrechnungsfaktor'}

value_dict_H={'Res_geocode': {}, 'HHNR': {}, 'HHSize':{},'N_Adult':{},'N_6-17':{},'N_<6':{},
'IncomeDescriptive': {1:'VeryBad',2:'Bad',3:'Moderate',4:'Good',5:'VeryGood'},
'IncomeDescriptiveNumeric': {1:1,2:2,3:3,4:4,5:5},
'BikeAvailable': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1,11:1,12:1,13:1,14:1,15:1,-90:0},
'EBikeAvailable': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1,-90:0},
'2_3WAvailable': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1,-90:0},
'CarAvailable': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1,-90:0},
'CarOwnershipHH': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1,-90:0},'CarOwnershipHH_num':{},
'Time2Transit':{},
'HH_Weight':{}
}

na_dict_H={'IncomeDescriptive':'Unknown','BikeAvailable':0,'EBikeAvailable':0,'2_3WAvailable':0,'CarAvailable':0, 'CarOwnershipHH':0,'CarOwnershipHH_num':0}

var_dict_P = {'HHNR':'hh_nr','Person':'pers_nr', # Variables related to administrative boundary units and identifiers
'Age':'pers_alter','Sex':'pers_geschlecht', # i don't see a 'relationship' variable in srv. the trip month and day coming from the trip file
'BikeAvailable':'pers_fahrrad', 'CarAvailable':'pers_pkw','2_3WAvailable':'pers_krad',
'Occupation':'pers_beruf', 'Education':'pers_bildung',
'DrivingLicense':'pers_fs_pkw','TransitSubscription': 'pers_oev_zeitkarte', 'Work/Study_AtHome':'pers_arbeittele',
'Work/Study_CarParkAvailable':'pers_arbeit_parkpl',
'Per_Weight':'pers_hochrechnungsfaktor'
}

value_dict_P={'HHNR': {}, 'Person': {}, # Variables related to administrative boundary units and identifiers
'Age': {}, 'Sex': {}, # relation to reference person, age, sex (1=M; 2=F)
'BikeAvailable': {1:1, 2:0},
'CarAvailable': {1:1, 2:0},
'2_3WAvailable': {1:1, 2:0},
'Occupation':{1:'Student',2:'Employed',3:'Retired',4:'Other'}, 
'Education':{1:'No diploma yet',2:'Secondary',3:'Apprenticeship',4:'Secondary+Matura',5:'University'}, 
'Work/Study_AtHome':{1:1, 2:0},
'Work/Study_CarParkAvailable':{1:1, 2:0,3:0},
'DrivingLicense':{1:1, 2:0},
'TransitSubscription':{1:1, 2:0},
'Per_Weight':{}
}

na_dict_P={'Occupation':'Other','Education':'Unknown','BikeAvailable':0,'CarAvailable':0,'2_3WAvailable': 0,
'DrivingLicense':0,'TransitSubscription':0,'Work/Study_CarParkAvailable':0,'Work/Study_AtHome':0}

var_dict_W={'HHNR':'hh_nr','Person':'pers_nr','Trip':'weg_nr','ReportingDay':'perstag_stnr', # Variables related to administrative boundary units and identifiers
'Ori_geocode':'weg_startgemeinde','Des_geocode':'weg_zielgemeinde',
'Season':'perstag_jzeit', 'Day':'perstag_tag','Time':'weg_startzeit',
'Trip_Purpose':'weg_zweck','Ori_Reason':'weg_quellzweck','Des_Reason':'weg_zielzweck',
'Mode_Detailed':'weg_vm_haupt','Mode':'weg_vm_haupt_kl',
'Trip_Distance':'weg_laenge','Trip_Duration':'weg_dauer','Trip_Weight':'weg_hochrechnungsfaktor_woche'
}

value_dict_W={'HHNR': {}, 'Person': {},'Trip': {},'ReportingDay':{}, # Variables related to administrative boundary units and identifiers
'Ori_geocode':{},'Des_geocode':{},
'Season':{1:'Spring',2:'Summer',3:'Autumn',4:'Winter'}, 'Day':{},'Time':{},
'Trip_Purpose':{10:'Work',20:'Work',30:'School',40:'Companion',50:'Shopping',60:'Personal',70:'Leisure',80:'Personal',880:'Other'},
'Ori_Reason':{10:'Work',20:'Work',30:'School',40:'Companion',50:'Shopping',60:'Personal',70:'Leisure',80:'Personal',870:'Home',880:'Other'},
'Des_Reason':{10:'Work',20:'Work',30:'School',40:'Companion',50:'Shopping',60:'Personal',70:'Leisure',80:'Personal',870:'Home',880:'Other'},
'Mode_Detailed':{100:'Foot',200:'Bike',300:'Taxi',302:'Moped',301:'CarDriver',400:'CarPassenger',501:'CityBus',502:'UBahn',503:'Train',504:'Coach',8801:'Truck',8802:'Plane',8803:'Ship',8888:'Other',-90:'Other'},
'Mode':{1:'Foot',2:'Bike',3:'Car',4:'Car',5:'Transit',88:'Other',-90:'Other'}, # 
'Trip_Distance':{},'Trip_Duration':{},'Trip_Weight':{}
}

na_dict_W={'Trip_Purpose':'Other','Ori_Reason':'Other','Des_Reason':'Other','Mode':'Other'}

# Combine variable, value, and na dictionaries
var_all = {'HH': var_dict_H,'P':var_dict_P,'W':var_dict_W}
value_all = {'HH': value_dict_H,'P':value_dict_P,'W':value_dict_W}
na_all = {'HH': na_dict_H,'P':na_dict_P,'W':na_dict_W}

# save combined dictionaries
with open('../dictionaries/Austria_var.pkl', 'wb') as f:
    pickle.dump(var_all, f)
with open('../dictionaries/Austria_val.pkl', 'wb') as f:
    pickle.dump(value_all, f)
with open('../dictionaries/Austria_na.pkl', 'wb') as f:
    pickle.dump(na_all, f)