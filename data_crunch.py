import os
import sys

rootdir ="/mnt/data/SHAVE_cases"
variables = ["MergedReflectivityQC","MergedDZDR","MergedDSMZ","MergedDRHO","MergedDKDP","MergedAzShear", "MergedDivShear"]

dates = []
numbers = []
with open('/mnt/data/SHAVE_cases/Capstones/CaseMatching.txt') as f:
    for line in f:
        dateRadar, number = line.split()
        date = dateRadar[:8]
        dates.append(date)
        numbers.append(number)
    f.close()

pathname, subdirs, files = os.walk(rootdir):

            
dictionary = dict(zip(dates,numbers))