from load_data import *
import pandas as pd 
import numpy as np
import os, sys, datetime

os.system("cp test.py rest.py")

path_to_localmax = "/csv/localmax/"
# loop through bottom and top of the hour for all cases

rootdir = '/mnt/data/SHAVE_cases'
DATA_HOME='/mnt/data/SHAVE_cases'
OUT_HOME='/mnt/data/michaelm/practicum/cases'
delta = 0.15

# gather cases
def get_cases(rootdir):
    cases_df = pd.DataFrame(columns={'casedate','multi_n'})
    casedirs = []
    for subdir, dirs, files in os.walk(rootdir):
        idx=0
        for dir in dirs:
            if dir[:2] == '20':
                casedate = dir
                for subdir, dirs, files in os.walk('{}/{}'.format(rootdir,dir)):
                    for dir in dirs:
                        if dir[:5] == 'multi':
                            multi_n = dir[5]
                            row = {'casedate':casedate,'multi_n':multi_n}
                            cases_df.loc[idx] = row
        break
    return cases_df

# given a date, retrieve positions for max ref storm
def get_storm_info(date,multi_n):

    LOCALMAX_PATH = '{}/{}/multi{}/csv/'.format(OUT_HOME,date,multi_n)
    case_df = pd.DataFrame(columns={"timedate","Latitude","Longitude","Storm","ref"})
    i=+1
    with open('{}/{}/finished{}'.format(DATA_HOME,date,multi_n)) as f:
        for line in f:
            latN_d, lonW_d, latS_d, lonE_d = line.split()
            latN_d = float(latN_d)
            lonW_d = float(lonW_d)
            latS_d = float(latS_d)
            lonE_d = float(lonE_d)            
            break
    f.close()

    # builds dataframe of case centers 
    for subdir, dirs, files in os.walk(LOCALMAX_PATH):
        files = sorted(files)
        length=len(sorted(files))-1
        # Loop through times
        for idx, file in enumerate(files):
            # build dataframe of locations
            timedate=file[-19:-4]
            minutes = timedate[-4:-2]
            if (minutes == '30' or minutes == '00') and idx != 0 and idx != length:
                print("Finding localmax for ", timedate)
                df = pd.read_csv('{}/MergedReflectivityQCCompositeMaxFeatureTable_{}.csv'.format(LOCALMAX_PATH, timedate))
                if df.empty:
                    print("Empty dataframe!")  
                else:
                    # List of valid clusters
                    valid_clusters = {}
                    keys = range(df.shape[0])
                    for i in keys:
                        valid_clusters[i] = True
                    # find max
                    for idx, val in enumerate(df["MergedReflectivityQCCompositeMax"]):
                        if valid_clusters[idx] == False:
                            continue
                        if val < 40 or df['Size'].iloc[idx] < 20:
                            valid_clusters[idx] = False
                            continue
                        
                        lat = df['#Latitude'].iloc[idx]
                        lon = df['Longitude'].iloc[idx]
                        latN = lat + delta
                        latS = lat - delta
                        lonW =  lon - delta
                        lonE =  lon + delta
                        # Don't include clusters too close to domain edge
                        if latN > latN_d or latS <= latS_d or lonW < lonW_d or lonE >= lonE_d:
                            valid_clusters[idx] = False
                            continue

                        for idx2, val2 in enumerate(df["MergedReflectivityQCCompositeMax"]):
                            if idx2 == idx or valid_clusters[idx2] == False:
                                continue
                            lat2 = df['#Latitude'].iloc[idx2]
                            lon2 = df['Longitude'].iloc[idx2]
                            if lat2 < latN and lat2 > latS and lon2 > lonW and lon2 < lonE:                     
                                valid_clusters[idx2] = False
                                continue     
                    # valid_clusters is complete
                    # # add valid rows to case_df
                    for key in valid_clusters.keys():
                        if valid_clusters[key] == False:
                            continue
                        else:
                            row_idx = key+1
                            row = {"timedate":timedate,"Latitude":df['#Latitude'].iloc[row_idx],"Longitude":df['Longitude'].iloc[row_idx],'Storm':df['RowName'].iloc[row_idx]}
                            case_df.loc[len(case_df.index)] = row

    case_df = case_df.sort_values(['timedate'])
    return case_df

# Build swaths
def w2accumulator(case_df, multi_n, fields):
    date = case_df['casedate']
    multi_n = case_df['multi_n']
    multi = '{}/{}/multi{}'.format(OUT_HOME,date,multi_n)
    os.system('rm -r {}/Mer* {}/ME* {}/Re* {}/tar*'.format(multi,multi,multi,multi))

    for field in fields:
        os.system('w2accumulator -i /mnt/data/SHAVE_cases/{}/multi{}/code_index.xml -g {} -o /mnt/data/michaelm/practicum/cases/{}/multi{}/uncropped -C 1 -t 30 --verbose="severe"'.format(date, multi_n, field,date, multi_n))
        print('w2accumulator -i /mnt/data/SHAVE_cases/{}/multi{}/code_index.xml -g {} -o /mnt/data/michaelm/practicum/cases/{}/multi{}/uncropped -C 1 -t 30 --verbose="severe"'.format(date, multi_n, field,date, multi_n))
        if field[8:] == 'Shear':
            os.system('w2accumulator -i /mnt/data/SHAVE_cases/{}/multi{}/code_index.xml -g {} -o /mnt/data/michaelm/practicum/cases/{}/multi{}/uncropped -C 3 -t 30 --verbose="severe"'.format(date, multi_n, field,date, multi_n))


def get_NSE(date,multi_n,fields):
    for field in fields:
        os.system('mkdir {}/{}/NSE'.format(OUT_HOME,date))
        os.sytem('ln -s /mnt/data/SHAVE_cases/{}/multi{}/NSE/{} /mnt/data/michaelm/practicum/cases/{}/NSE'.format(date,multi_n,field,date))

# Run localmax on composite reflectivity
def localmax(date, multi_n):
    os.system('makeIndex.pl /mnt/data/SHAVE_cases/{}/multi{} code_index.xml'.format(date,multi_n))
    os.system('w2localmax -i /mnt/data/SHAVE_cases/{}/multi{}/code_index.xml -I MergedReflectivityQCComposite -o /mnt/data/michaelm/practicum/cases/{}/multi{} -s -d "40 60 5"'.format(date,multi_n,date,multi_n))
    os.system('makeIndex.pl /mnt/data/michaelm/practicum/cases/{}/multi{} code_index.xml'.format(date,multi_n))
    os.system('w2table2csv -i {}/{}/multi{}/code_index.xml -T MergedReflectivityQCCompositeMaxFeatureTable -o {}/{}/multi{}/csv -h'.format(OUT_HOME,date,multi_n,OUT_HOME,date,multi_n))


def cropconv(case_df, date, nse_fields, fields_accum, multi_n):
    for idx, row in case_df.iterrows():
        multi = '{}/{}/multi{}'.format(OUT_HOME,date,multi_n)
        lon = row['Longitude']
        lat = row['Latitude']
        delta = 0.15

        lonNW = lon - delta
        latNW = lat + delta
        lonSE = lon + delta
        latSE = lat - delta
        
        time1 = row['timedate']
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time2 = (date_1+datetime.timedelta(minutes=1)).strftime('%Y%m%d-%H%M%S')
        print(time1[-8:])
        print(time2)

        # crop input
        #########################
        os.system("makeIndex.pl {}/{}/multi{}/uncropped code_index.xml {} {}".format(OUT_HOME,date,multi_n, time1, time2)) # make index for uncropped
        for field in fields_accum:
            os.system('w2cropconv -i {}/{}/multi{}/uncropped/code_index.xml -I {} -o /mnt/data/michaelm/practicum/cases/{}/multi{}/storm{:02d} -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME,date, multi_n, field, date, multi_n, idx, latNW, lonNW,latSE,lonSE))
            print('w2cropconv -i {}/{}/multi{}/uncropped/code_index.xml -I {} -o /mnt/data/michaelm/practicum/cases/{}/multi{}/storm{:02d} -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME,date, multi_n, field, date, multi_n, idx, latNW, lonNW,latSE,lonSE))
        #########################

        # crop target
        #########################
        time1 = (date_1+datetime.timedelta(minutes=30)).strftime('%Y%m%d-%H%M%S')
        date_1 = datetime.datetime.strptime(time1, "%Y%m%d-%H%M%S")
        time2 = (date_1+datetime.timedelta(minutes=1)).strftime('%Y%m%d-%H%M%S')        
        os.system("makeIndex.pl {}/{}/multi{}/uncropped code_index.xml {} {}".format(OUT_HOME,date,multi_n, time1, time2))
        os.system('w2cropconv -i {}/{}/multi{}/uncropped/code_index.xml -I MESH_Max_30min -o /mnt/data/michaelm/practicum/cases/{}/multi{}/storm{:02d}/target_MESH_Max_30min -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(OUT_HOME,date, multi_n, date, multi_n, idx, latNW, lonNW,latSE,lonSE))
        ########################

        # NSE
        os.system("makeIndex.pl {}/{}/NSE code_index.xml {} {}".format(DATA_HOME,date, time1, time2))
        for field in nse_fields:
            os.system('w2cropconv -i {}/{}/NSE/code_index.xml -I {} -o /mnt/data/michaelm/practicum/cases/{}/multi{}/storm{:02d}/NSE -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(DATA_HOME,date, field, date, multi_n, idx, latNW, lonNW,latSE,lonSE))
        
        os.system('w2cropconv -i {}/{}/multi{}/code_index.xml -I  MergedReflectivityQC -o /mnt/data/michaelm/practicum/cases/{}/multi{}/storm{:02d} -t "{} {}" -b "{} {}" -s "0.005 0.005" -R -n --verbose="severe"'.format(DATA_HOME,date, multi_n, date, multi_n, idx, latNW, lonNW,latSE,lonSE))

fields = ['MergedLLShear','MergedMLShear','MESH','Reflectivity_0C','Reflectivity_-10C','Reflectivity_-20C', 'MergedReflectivityQCComposite']
fields_accum = ['MergedLLShear_Max_30min','MergedMLShear_Max_30min','MESH_Max_30min','Reflectivity_0C_Max_30min','Reflectivity_-10_Max_30min','Reflectivity_-20_Max_30min', 'MergedReflectivityQCComposite_Max_30min',
                'MergedLLShear_Min_30min','MergedMLShear_Min_30min']


nse_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km']

#cases_df = get_cases(rootdir)
#cases_df.to_csv('{}/cases.csv'.format(OUT_HOME))

cases_df = pd.read_csv('{}/cases.csv'.format(OUT_HOME))
cases_df = cases_df.sort_values(['casedate'])
i=0
for idx, case in cases_df.iterrows():

    date = case['casedate'] 
    multi_n = case['multi_n']
    
    print(date,"multi",multi_n)
    # get swaths
    w2accumulator(case, multi_n, fields)
    # collect NSE fields
    #get_NSE(date,multi_n,NSE_fields)

    # run localmax
    #localmax(date,multi_n)

    # gather individual storm info
    storm_df = get_storm_info(date, multi_n)
    print(storm_df)
    cropconv(storm_df, date, nse_fields, fields_accum, multi_n)


