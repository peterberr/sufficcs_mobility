import pickle

var_dict_H = {'Sector_Zone':'ZFM','Sample':'ECH', # Variables related to administrative boundary units and identifiers
'BikeAvailable':'M21', 'CarAvailable':'M6','CarOwnershipHH':'M6','CarOwnershipHH_num':'M6','2_3WAvailable':'M14','HouseType':'M1','HouseTenure':'M2','HouseTenureAgg':'M2','HH_Weight':'COE0'}

# in Montpellier (and other cities potentially), electric bikes are a subset of bikes, so unneccessary to create an additional 'electric bike available' variable
value_dict_H={'Sector_Zone': {}, 'Sample': {}, # Variables related to administrative boundary units and identifiers
'BikeAvailable': {0:0, 1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1,15:1},
'CarAvailable': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1,7:1,8:1,9:1},
'CarOwnershipHH': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1,7:1,8:1,9:1},'CarOwnershipHH_num':{},
'2_3WAvailable': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1},
'HouseType': {1:'SFH', 2:'SFH', 3:'MFH_small', 4:'MFH_large', 5:'Unknown'},
'HouseTenure': {1:'Owner', 2:'Renter_Social', 3:'Renter_Private', 4:'Rent_Free', 5:'Renter_Private',6:'Unknown',7:'Unknown',8:'Unknown'},
'HouseTenureAgg': {1:'Owner', 2:'Renter', 3:'Renter', 4:'Renter', 5:'Renter', 6:'Unknown', 7:'Unknown', 8:'Unknown'},
'HH_Weight':{}
}

na_dict_H={'BikeAvailable':0,'CarAvailable':0,'CarOwnershipHH':0,'CarOwnershipHH_num':0,'2_3WAvailable':0,
'HouseType':'Unknown','HouseTenure':'Unknown','HouseTenureAgg':'Unknown'}

var_dict_P = {'Sector_Zone':'ZFP','Sample':'ECH','Person':'PER', # Variables related to administrative boundary units and identifiers
'Relationship':'P3','Age':'P4','Sex':'P2',
'Month':'MOIS', 'Season':'MOIS', 'Day':'JOUR', 
'Occupation':'P9', 'Education':'P8',
'DrivingLicense':'P7','TransitSubscription':'P12','Work/Study_AtHome':'P14',
'Work/Study_CarParkAvailable1':'P17','Work/Study_CarParkAvailable2':'P18', # these two 'parking available' variables need combined with an OR operator
'Work/Study_BikeParkAvailable':'P18A','Per_Weight':'COEP'
}

value_dict_P={'Sector_Zone': {}, 'Sample': {}, 'Person': {}, # Variables related to administrative boundary units and identifiers
'Relationship':{1:'Ref',2:'Partner',3:'Child',4:'Other not related',5:'Other related',6:'Other not related',7:'Other not related'},'Age': {}, 'Sex': {}, # relation to reference person, age, sex (1=M; 2=F)
'Month': {}, 'Season': {1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn',12:'Winter'}, 'Day': {},
'Occupation':{1:'Employed_FullTime',2:'Employed_PartTime',3:'Trainee',4:'Student_3rdLevel',5:'Student_School',6:'Unemployed',7:'Retired',8:'Home_Partner',9:'Other'},
'Education':{0:'No diploma yet',1:'Elementary',2:'Secondary',3:'Secondary',4:'Secondary+BAC',5:'University',6:'University',7:'Apprenticeship',8:'Apprenticeship',9:'Other',90:'Other',93:'Secondary',97:'Apprenticeship'}, # there are some inconsitencies with the Education file which will be addressed in the combine script
'DrivingLicense':{1:1,2:0,3:0},
'TransitSubscription':{1:1,2:1,3:1,4:0,5:1,6:1},
'Work/Study_AtHome':{1:1,2:0},
'Work/Study_CarParkAvailable1':{1:0,2:1,3:1,4:1},'Work/Study_CarParkAvailable2':{1:0,2:1,3:1,4:1}, # these two 'parking available' variables need combined with an OR operator
'Work/Study_BikeParkAvailable':{1:1,2:1,3:1,4:1,5:0,6:1,7:1},'Per_Weight':{}
}

na_dict_P={'Occupation':'Other','Education':'Unknown','Work/Study_AtHome':0,'Work/Study_BikeParkAvailable':0,
'DrivingLicense':0,'TransitSubscription':0,'Relationship':'Other'}


var_dict_W={'Sector_Zone':'ZFD','Sample':'ECH','Person':'PER','Trip':'NDEP', # Variables related to administrative boundary units and identifiers
'Ori_Sec_Zone':'D3','Des_Sec_Zone':'D7',
'Time':'D4','Ori_Reason1':'D2A','Ori_Reason2':'D2B','Des_Reason1':'D5A','Des_Reason2':'D5B',
'Mode':'MODP','Mode_disagg':'MODP','Trip_Distance':'D12','Trip_Duration':'D9','N_Stops':'D6','N_Legs':'D10'
}

value_dict_W={'Sector_Zone': {}, 'Sample': {}, 'Person': {},'Trip':{},
'Ori_Sec_Zone':{},'Des_Sec_Zone':{}, # these O-D variables need further modification to split out sectors and zones
'Time':{},'Ori_Reason1':{},'Ori_Reason2':{},'Des_Reason1':{},'Des_Reason2':{}, # these reasons need further editing to combine trip purposes of independent travellers and accompanied travellers
'Mode':{1:'Foot',10:'Bike',11:'Bike',12:'Bike',13:'2_3_Wheel',14:'2_3_Wheel',15:'2_3_Wheel',16:'2_3_Wheel',17:'2_3_Wheel',18:'2_3_Wheel',21:'Car',22:'Car',
31:'Transit',32:'Transit',33:'Transit',37:'Transit',38:'Transit',39:'Transit',41:'Transit',42:'Transit',51:'Transit',71:'Transit',
61:'Car',81:'Car',82:'Car',91:'Other',92:'Other',93:'Foot',94:'Foot',95:'Other'}, # assume 93 (rollerblades/scooters) and 94 (wheelchairs) are same as walking by foot, which is generally true from an energy perspective
'Trip_Distance':{},'Trip_Duration':{},'N_Stops':{},'N_Legs':{},

'Mode_disagg':{1:'Foot',10:'Bike',11:'Bike',12:'Bike',13:'2_3_Wheel',14:'2_3_Wheel',15:'2_3_Wheel',16:'2_3_Wheel',17:'2_3_Wheel',18:'2_3_Wheel',21:'Car',22:'Car',
31:'Bus',32:'Tram',33:'UBahn',37:'Car',38:'Bus',39:'Bus',41:'Bus',42:'Bus',51:'Train',71:'Car',
61:'Car',81:'Car',82:'Car',91:'Other',92:'Other',93:'Foot',94:'Foot',95:'Other'}, # assume 93 (rollerblades/scooters) and 94 (wheelchairs) are same as walking by foot, which is generally true from an energy perspective
}

na_dict_W={}

var_dict_T={'Sector_Zone':'ZFT','Sample':'ECH','Person':'PER','Trip':'NDEP','Leg':'T1', # Variables related to administrative boundary units and identifiers
'Ori_Sec_Zone':'T4','Des_Sec_Zone':'T5',
'Mode':'T3','Mode_disagg':'T3','N_People':'T8','N_Adult':'T8A','N_Child':'T8B','Trip_Distance':'T13'
}

# includes mode_disagg, which includes modes of UBahn, SBahn, Train, Bus, Tram. Assume demand response transit (#37) and employer transport (#71) are by car, and that unspecified 'other networks' (38,39) and departmental public transit (41) are by bus
value_dict_T={'Sector_Zone': {}, 'Sample': {}, 'Person': {},'Trip':{},'Leg':{},
'Ori_Sec_Zone':{},'Des_Sec_Zone':{}, # these O-D variables need further modification to split out sectors and zones
'Mode':{1:'Foot',10:'Bike',11:'Bike',12:'Bike',13:'2_3_Wheel',14:'2_3_Wheel',15:'2_3_Wheel',16:'2_3_Wheel',17:'2_3_Wheel',18:'2_3_Wheel',21:'Car',22:'Car',
31:'Transit',32:'Transit',33:'Transit',37:'Car',38:'Transit',39:'Transit',41:'Transit',42:'Transit',51:'Transit',71:'Car',
61:'Car',81:'Car',82:'Car',91:'Other',92:'Other',93:'Foot',94:'Foot',95:'Other'}, # assume 93 (rollerblades/scooters) and 94 (wheelchairs) are same as walking by foot, which is generally true from an energy perspective
'N_People':{},'N_Adult':{},'N_Child':{},'Trip_Distance':{},

'Mode_disagg':{1:'Foot',10:'Bike',11:'Bike',12:'Bike',13:'2_3_Wheel',14:'2_3_Wheel',15:'2_3_Wheel',16:'2_3_Wheel',17:'2_3_Wheel',18:'2_3_Wheel',21:'Car',22:'Car',
31:'Bus',32:'Tram',33:'UBahn',37:'Car',38:'Bus',39:'Bus',41:'Bus',42:'Bus',51:'Train',71:'Car',
61:'Car',81:'Car',82:'Car',91:'Other',92:'Other',93:'Foot',94:'Foot',95:'Other'}, # assume 93 (rollerblades/scooters) and 94 (wheelchairs) are same as walking by foot, which is generally true from an energy perspective
'N_People':{},'N_Adult':{},'N_Child':{},'Trip_Distance':{}
}

na_dict_T={}

# Combine variable, value, and na dictionaries
var_all = {'HH': var_dict_H,'P':var_dict_P,'W':var_dict_W,'T':var_dict_T}
value_all = {'HH': value_dict_H,'P':value_dict_P,'W':value_dict_W,'T':value_dict_T}
na_all = {'HH': na_dict_H,'P':na_dict_P,'W':na_dict_W,'T':na_dict_T}

# save combined dictionaries
with open('../dictionaries/Dijon_var.pkl', 'wb') as f:
    pickle.dump(var_all, f)
with open('../dictionaries/Dijon_val.pkl', 'wb') as f:
    pickle.dump(value_all, f)
with open('../dictionaries/Dijon_na.pkl', 'wb') as f:
    pickle.dump(na_all, f)

# save combined dictionaries
with open('../dictionaries/Lille_var.pkl', 'wb') as f:
    pickle.dump(var_all, f)
with open('../dictionaries/Lille_val.pkl', 'wb') as f:
    pickle.dump(value_all, f)
with open('../dictionaries/Lille_na.pkl', 'wb') as f:
    pickle.dump(na_all, f)

# save combined dictionaries
with open('../dictionaries/Lyon_var.pkl', 'wb') as f:
    pickle.dump(var_all, f)
with open('../dictionaries/Lyon_val.pkl', 'wb') as f:
    pickle.dump(value_all, f)
with open('../dictionaries/Lyon_na.pkl', 'wb') as f:
    pickle.dump(na_all, f)

# save combined dictionaries
with open('../dictionaries/Montpellier_var.pkl', 'wb') as f:
    pickle.dump(var_all, f)
with open('../dictionaries/Montpellier_val.pkl', 'wb') as f:
    pickle.dump(value_all, f)
with open('../dictionaries/Montpellier_na.pkl', 'wb') as f:
    pickle.dump(na_all, f)

# save combined dictionaries
with open('../dictionaries/Nantes_var.pkl', 'wb') as f:
    pickle.dump(var_all, f)
with open('../dictionaries/Nantes_val.pkl', 'wb') as f:
    pickle.dump(value_all, f)
with open('../dictionaries/Nantes_na.pkl', 'wb') as f:
    pickle.dump(na_all, f)

# save combined dictionaries
with open('../dictionaries/Nimes_var.pkl', 'wb') as f:
    pickle.dump(var_all, f)
with open('../dictionaries/Nimes_val.pkl', 'wb') as f:
    pickle.dump(value_all, f)
with open('../dictionaries/Nimes_na.pkl', 'wb') as f:
    pickle.dump(na_all, f)