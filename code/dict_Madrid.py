import pickle

var_dict_H = {'Sector':'CZ208','Zone':'ZT1259','HHNR':'ID_HOGAR', # Variables related to administrative boundary units and identifiers
'HHSize':'A1PER','HHS2':'N_MIEMBROS_POR_HOGAR','N_Vehicles':'B1NVE','Type_V1':'V1B11TIPO','Type_V2':'V2B11TIPO1','Type_V3':'V3B11TIPO1','Type_V4':'V4B11TIPO1','Type_V5':'V5B11TIPO1',
'HH_Weight':'ELE_HOGAR_NUEVO'}

value_dict_H={'Sector': {},'Zone': {},'HHNR': {}, # Variables related to administrative boundary units and identifiers
'HHSize':{},'HHS2':{},
'N_Vehicles':{},'Type_V1':{},'Type_V2':{},'Type_V3':{},'Type_V4':{},'Type_V5':{},
'HH_Weight':{}
}

na_dict_H={}

var_dict_P = {'HHNR':'ID_HOGAR','Person':'ID_IND', # Variables related to administrative boundary units and identifiers
'Age':'EDAD_FIN','Sex':'C2SEXO',
'Month':'DMES', 'Season':'DMES', 'Day':'DIASEM', 
'Occupation':'C8ACTIV', 'Education':'C7ESTUD',
'DrivingLicense':'C6CARNE','TransitSubscription':'C14ABONO',
'MobilityConstraints':'CPMR','Per_Weight':'ELE_G_POND'
}

value_dict_P={'HHNR': {}, 'Person': {}, # Variables related to administrative boundary units and identifiers
'Age': {}, 'Sex': {}, # relation to reference person, age, sex (1=M; 2=F)
'Month': {}, 'Season': {1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn',12:'Winter'}, 'Day': {},
'Occupation':{1:'Employed_FullTime',2:'Employed_PartTime',3:'Retired',4:'Unemployed',5:'Unemployed',6:'Student',7:'Home_Partner',8:'Home_Partner',9:'Other'},
'Education':{1:'No diploma yet',2:'Elementary',3:'Secondary',4:'Secondary',5:'Apprenticeship',6:'Apprenticeship',7:'University'}, 
'DrivingLicense':{1:0,2:0,3:0,4:1,5:1},
'TransitSubscription':{1:1,2:0},
'MobilityConstraints':{1:1,2:0},'Per_Weight':{}
}

na_dict_P={'Occupation':'Other','Education':'Unknown',
'DrivingLicense':0,'TransitSubscription':0,'MobilityConstraints':0}


var_dict_W={'HHNR':'ID_HOGAR','Person':'ID_IND','Trip':'ID_VIAJE', # Variables related to administrative boundary units and identifiers
'Ori_Zone':'VORIZT1259','Des_Zone':'VDESZT1259',
'Time_Departure':'VORIHORAINI','Time_Arrival':'VDESHORAFIN','Ori_Reason':'VORI','Des_Reason':'VDES','Trip_Purpose0':'MOTIVO_PRIORITARIO',
'Mode':'MODO_PRIORITARIO','Trip_Distance':'DISTANCIA_VIAJE','N_Legs':'N_ETAPAS_POR_VIAJE','Trip_Weight':'ELE_G_POND_ESC2',
}

value_dict_W={'HHNR': {}, 'Person': {},'Trip':{},
'Ori_Zone':{},'Des_Zone':{}, # these O-D variables need further modification to split out sectors and zones
'Time_Departure':{},'Time_Arrival':{},'Ori_Reason':{},'Des_Reason':{},
'Trip_Purpose0':{1:'Home',2:'Work',3:'Work',4:'School',5:'Shopping',6:'Personal',7:'Companion',8:'Leisure',9:'Leisure',10:'Personal',11:'Other',12:'Other'}, # these reasons need further editing to combine trip purposes of independent travellers and accompanied travellers
'Mode':{1:'Transit',2:'Transit',3:'Transit',4:'Transit',5:'Transit',6:'Transit',7:'Transit',8:'Transit',9:'Transit',
10:'Car',11:'Car',12:'Car',13:'Car',14:'Car',15:'Car',16:'Car',
17:'2_3_Wheel',18:'2_3_Wheel',19:'2_3_Wheel',
20:'Bike',21:'Bike',22:'Bike',23:'Other',24:'Foot'}, 
'Trip_Distance':{},'N_Legs':{},'Trip_Weight':{}
}

na_dict_W={'Mode':'Other','Trip_Purpose0':'Other'}

# Combine variable, value, and na dictionaries
var_all = {'HH': var_dict_H,'P':var_dict_P,'W':var_dict_W}
value_all = {'HH': value_dict_H,'P':value_dict_P,'W':value_dict_W}
na_all = {'HH': na_dict_H,'P':na_dict_P,'W':na_dict_W}

# save combined dictionaries
with open('../dictionaries/Madrid_var.pkl', 'wb') as f:
    pickle.dump(var_all, f)
with open('../dictionaries/Madrid_val.pkl', 'wb') as f:
    pickle.dump(value_all, f)
with open('../dictionaries/Madrid_na.pkl', 'wb') as f:
    pickle.dump(na_all, f)
