import pickle

var_dict_H = {'Grid_Cell':'RESC','Commune':'RESCOMM','HHNR':'NQUEST', # Variables related to administrative boundary units and identifiers
'Week':'SEM',
'BikeAvailable':'NB_VELO', 'CarAvailable':'NB_VD','CarOwnershipHH':'NB_VD','2_3WAvailable':'NB_2RM',
'HHSize':'MNP','IncomeDetailed':'REVENU','IncomeHarmonised':'REVENU',
'HouseType':'TYPELOG','HouseTenure':'OCCUPLOG','HouseTenureAgg':'OCCUPLOG','HH_Weight':'POIDSM'}

value_dict_H={'Grid_Cell': {}, 'Commune': {}, 'HHNR':{}, # Variables related to administrative boundary units and identifiers
'Week':{},
'BikeAvailable': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1},
'CarAvailable': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1},
'CarOwnershipHH': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1},
'2_3WAvailable': {0:0, 1:1, 2:1, 3:1, 4:1, 5:1},
'HHSize': {},
'IncomeDetailed': {1: 'Under800', 2: '800-1200', 3: '1200-1600', 4: '1600-2000', 5:'2000-2400', 6:'2400-3000', 7:'3000-3500', 8:'3500-4500', 9:'4500-5500',10:'Over5500',11:'Unkown',12:'Unknown'},
'IncomeHarmonised': {1: 'Under1000', 2: '1000-2000', 3: '1000-2000', 4: '1000-2000', 5: '2000-3000', 6: '2000-3000', 7: '3000-4500', 8: '3000-4500', 9:'Over4500',10:'Over4500',11:'Unkown',12:'Unknown'},
'HouseType': {1:'SFH', 2:'SFH', 3:'SFH', 4:'MFH_small', 5:'MFH_large', 6:'MFH_large', 7:'Unknown'},
'HouseTenure': {1:'Owner', 2:'Owner', 3:'Renter_Social', 4:'Renter_Private', 5:'Renter_Private', 6:'Renter_Private', 7:'Rent_Free', 8:'Rent_Free',9:'Unknown'},
'HouseTenureAgg': {1:'Owner', 2:'Owner', 3:'Renter', 4:'Renter', 5:'Renter', 6:'Renter', 7:'Renter', 8:'Renter',9:'Unknown'},
'HH_Weight':{}
}
# values to fill in with if NA
na_dict_H={'BikeAvailable':0,'CarAvailable':0,'CarOwnershipHH':0,'2_3WAvailable':0,'IncomeDetailed':'Unknown','IncomeHarmonised':'Unknown',
'HouseType':'Unknown','HouseTenure':'Unknown','HouseTenureAgg':'Unknown'}

var_dict_P = {'Grid_Cell':'RESC','Commune':'RESCOMM','HHNR':'NQUEST','Person':'NP', # Variables related to administrative boundary units and identifiers
'Relationship':'LIENPREF','Age':'AGE','Sex':'SEXE','NoMobilityConstraints':'HANDI',
'Day':'JDEP', 
'Occupation':'OCCP', 'Education':'DIPL',
'DrivingLicense':'PERMVP','TransitSubscription':'ABONTC','Work/Study_AtHome':'ULTRAV',
'Work/Study_CarParkAvailable':'PKVPTRAV',
'Work/Study_BikeParkAvailable':'PKVLTRAV','Per_Weight':'POIDSP'
}

value_dict_P={'Grid_Cell': {}, 'Commune': {},'HHNR':{},'Person':{},  # Variables related to administrative boundary units and identifiers
'Relationship':{1:'Ref',2:'Partner',3:'Child',4:'Other related',5:'Other related',6:'Other related',7:'Other not related',8:'Other not related'},'Age': {}, 'Sex': {}, # relation to reference person, age, sex (1=M; 2=F)
'NoMobilityConstraints':{0:1,1:0},
'Day': {},
'Occupation':{0:'Retired',1:'Employed_FullTime',2:'Employed_PartTime',3:'Student_3rdLevel',4:'Trainee',5:'Student_School',6:'Unemployed',7:'Retired',8:'Unemployed',9:'Home_Partner'},
'Education':{1:'No diploma yet',2:'Elementary',3:'Secondary',4:'Secondary',5:'Secondary+BAC',6:'University',7:'University',8:'Apprenticeship',9:'University'}, # there are some inconsitencies with the Education file which will be addressed in the combine script
'DrivingLicense':{1:1,2:0,3:0},
'TransitSubscription':{1:0,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1,15:1},
'Work/Study_AtHome':{1:0,2:1,3:0},
'Work/Study_CarParkAvailable':{1:1,2:0},
'Work/Study_BikeParkAvailable':{1:1,2:0},'Per_Weight':{}
}

na_dict_P={'Occupation':'Other','Education':'Unknown','Work/Study_AtHome':0,
'Work/Study_CarParkAvailable':0,'Work/Study_BikeParkAvailable':0,'Relationship':'Other'}

var_dict_W = {'Grid_Cell':'RESC','Commune':'RESCOMM','HHNR':'NQUEST','Person':'NP','Trip':'ND', # Variables related to administrative boundary units and identifiers
'Ori_Cell':'ORC','Des_Cell':'DESTC','Ori_Comm':'ORCOMM','Des_Comm':'DESTCOMM',
'Time':'ORH','Ori_Reason1':'ORMOT_H9','Ori_Reason2':'ACCMOT_H9','Des_Reason1':'DESTMOT_H9','Des_Reason2':'ACCMOT_H9',
'Mode':'MODP_H12','Trip_Distance':'DPORTEE','Trip_Duration':'DUREE','N_Stops':'NBAT','N_Legs':'NBTRAJ'
}

value_dict_W={'Grid_Cell': {}, 'Commune': {},'HHNR':{},'Person':{},'Trip':{},  # Variables related to administrative boundary units and identifiers
'Ori_Cell':{},'Des_Cell':{}, 'Ori_Comm':{},'Des_Comm':{},
'Time':{},'Ori_Reason1':{},'Ori_Reason2':{},'Des_Reason1':{},'Des_Reason2':{}, # these reasons need further editing to combine trip purposes of independent travellers and accompanied travellers
'Mode':{1:'Transit',2:'Transit',3:'Transit',4:'Transit',5:'Transit',6:'Car',7:'Car',8:'Car',9:'2_3_Wheel',10:'Bike',11:'Other',12:'Foot'},
'Trip_Distance':{},'Trip_Duration':{},'N_Stops':{},'N_Legs':{}
}

na_dict_W={}

# Combine variable, value, and na dictionaries
var_all = {'HH': var_dict_H,'P':var_dict_P,'W':var_dict_W}
value_all = {'HH': value_dict_H,'P':value_dict_P,'W':value_dict_W}
na_all = {'HH': na_dict_H,'P':na_dict_P,'W':na_dict_W}

# save combined dictionaries
with open('../dictionaries/Paris_var.pkl', 'wb') as f:
    pickle.dump(var_all, f)
with open('../dictionaries/Paris_val.pkl', 'wb') as f:
    pickle.dump(value_all, f)
with open('../dictionaries/Paris_na.pkl', 'wb') as f:
    pickle.dump(na_all, f)