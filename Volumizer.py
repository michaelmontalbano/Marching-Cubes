from netCDF4 import Dataset
import pandas as pd
import numpy as np
import pygalmesh
import meshio
import meshplex
import networkx as nx
import itertools
import time
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from datetime import date, datetime, timedelta
import sys, time, argparse, os
from sys import argv
import math as m
import os

colors = ['r','b','g','m','y','w','k','y','b','b','y','r']
features_list = ["zdrColumn", "kdpColumn", "hail volume", "zdrArc", "zdrRing", "rhoRing", "30 dBZ outline","MARC","storm top divergence","kdpEnhancement","llm","mlm"]
#dates = ["20120329","20120601","20120622","20120705","20120716","20120718","20120720","20120724","20120726","20120727","20120730","20120803","20120808","20120809","20120810","20130515","20130517","20130522","20130528","20130530","20130610","20130612","20130614","20130617","20130618","20130620","20130624","20130625","20130626","20130628","20130710","20130711","20130718","20130805","20130808", "20140521", "20140528", "20140601", "20140611", "20140616","20140617","20140618", "20140619", "20140623", "20140626", "20140629", "20140702", "20140708", "20140712", "20140714", "20140723", "20140726", "20140727", "20140804", "20150611", "20150612", "20150615", "20150617", "20150623", "20150625", "20150629", "20150630", "20150713", "20150714", "20150715", "20150716", "20150720", "20150721", "20150724", "20150725", "20150728", "20150801", "20150802", "20150803", "20150806", "20150807"]

dates = ["20120329"]

clr_dict = {features_list[i]: colors[i] for i in range(len(features_list))}


np.set_printoptions(threshold=sys.maxsize)

SHAVE_path = "/mnt/data/SHAVE_cases/"
#casesPath = 'C:\\Users\\User\\weather\\cases\\'
casesPath = "/mnt/data/SHAVE_cubes/"
shavePath = "/mnt/data/SHAVE_cases/"


date = "20130515"
time1 = "23"
time2 = "221000"
field3D = "MergedDZDR"
convUnit = 0.5*0.5*1 
DATE_TIME_FORMAT = '%Y%m%d-%H%M%S'
DATE_FORMAT = '%Y%m%d'
maxHeight = 20
maxDist = 0.08
tracking = 1

features = np.arange(0,12,1) # run for all features

# field to feed into marching_cubes in case of exception
A = np.zeros((3,3,3))
for i in np.arange(0,1):
    for j in np.arange(0,1):
        for k in np.arange(0,1):
            A[i,j,k] = 1
 
def buildDataFrame(features):
    titles = []
    titles.append("time")
    for idx, feature in enumerate(features):
        fields = setFeature(idx)[0]
        maxVal = feat + "_" + interest + "_max"
        depth = feat + '_depth'
        width = feat + '_width'
        volume = feat + '_volume_at_max'
        centroid = feat + '_centroid'
        volumes = feat+ "_volumes"
        numVol = feat + "_numVol"
        V = feat + "_totalVolume" 

        titles.append(feat+"_Volumes")
        titles.append(numVol)
        titles.append(V)
        titles.append(maxVal)
        titles.append(depth)
        titles.append(width)
        titles.append(feat+"_specificVolume")
        titles.append(centroid)

        for field in fields:
            title = feat + "_" + field + "_corr"
            titles.append(title)
        
        titles.append(feat+"meshes")
        titles.append(feat+"objects")

    print("length:",len(titles))
    print(titles)
    df = pd.DataFrame(columns=titles)
    return df

def setFeature(chosen):
    global chosenFields, chosenThresholds, chosenHeights, interest, feat
    if chosen == 0:
        title = "zdrColumn"
        chosenFields = ["MergedDZDR", "MergedReflectivityQC", "MergedDRHO"]
        chosenThresholds = [0.5, 50,0.80]
        chosenHeights = ["0","0","0"]
        topHeights = ["0","0","0"]
        interest = "MergedDZDR"
        feat = title
    if chosen == 1:
        title = "kdpColumn"
        chosenFields = ["MergedDKDP", "MergedReflectivityQC"]
        chosenThresholds = [2, 50]
        chosenHeights = ["surface","surface"]
        topHeights = ["0","0","0"]
        interest = "MergedDKDP"
        feat = title
    if chosen == 2:
        title = "kdpEnhancement"
        chosenFields = ["MergedDKDP", "MergedReflectivityQC","MergedDRHO"]
        chosenThresholds = [2, 50, 0.93] 
        chosenHeights = ["surface","surface","surface"]
        topHeights = ["0","0","0"]        
        interest = "MergedDKDP"
        feat = title
    if chosen == 3:
        title =  "zdrArc"
        chosenFields = ["MergedDZDR", "MergedReflectivityQC", "MergedDRHO"]
        chosenThresholds = [2, 40, -0.85] 
        chosenHeights = ["-0","-0","-0"] 
        topHeights = ["-0","-0","-0"]
        offset = [-1, -1, -1]
        interest = "MergedDZDR"
        feat = title
    if chosen == 4:
        title = "zdrRing"
        chosenFields = ["MergedDZDR", "MergedReflectivityQC"]
        chosenThresholds = [1, 40]
        chosenHeights = ["0","0"]
        topHeights = ["0","0","0"]
        interest = "MergedDZDR"
        feat = title
    if chosen == 5:
        title =  "rhoRing"
        chosenFields = ["MergedDRHO", "MergedReflectivityQC"]
        chosenThresholds = [0.96, 40]
        chosenHeights = ["0","0"] 
        topHeights = ["0","0","0"]
        interest = "MergedDRHO"
        feat = title
    if chosen == 6:
        title = "hail volume"
        chosenFields = ["MergedDZDR", "MergedReflectivityQC"]
        chosenThresholds = [4, 45]
        chosenHeights = ["-0","-0"]
        topHeights = ["0","0","0"] 
        interest = "MergedDZDR"
        feat = title
    if chosen == 7:
        title = "llm"
        chosenFields = ["MergedAzShear", "MergedReflectivityQC"]
        chosenThresholds = [0.004,20]
        chosenHeights = ["-0","-0"]
        topHeights = ["0","0","0"] 
        interest = "MergedAzShear"
        feat = title
    if chosen == 8:
        title = "mlm"
        chosenFields = ["MergedAzShear", "MergedReflectivityQC"]
        chosenThresholds = [0.004,20]
        chosenHeights = ["0","0"]
        topHeights = ["0","0","0"] 
        interest = "MergedAzShear"
        feat = title
    if chosen == 9:
        title = "dBZ"
        chosenFields = ["MergedReflectivityQC"]
        chosenThresholds = [30]
        chosenHeights = ["surface"]
        topHeights = ["0","0","0"]
        interest = "MergedReflectivityQC"
        feat = title
    if chosen == 10:
        title = "MARC"
        chosenFields = ["MergedDivShear"]
        chosenThresholds = [0.004]
        chosenHeights = ["0"]
        topHeights = ["0"]
        interest = "MergedDivShear"
        feat = title
    # maybe make dbz < 25
    if chosen == 11:
        title = "storm top divergence"
        chosenFields = ["MergedDivShear"]
        chosenThresholds = [0.004]
        chosenHeights = ["0"]
        topHeights = ["0"]
        interest = "MergedDivShear"
        feat = title                

    return [chosenFields, chosenThresholds, chosenHeights, topHeights, interest, feat]

def getHgtWghtedVolume():
    '''
    '''

def getFields(timeStep,caseDate,fields3D, feature):
    '''
    input: file location and field sought
    output: 3D np array
    '''
    caseDate = "{:04d}{:02d}{:02d}".format(caseDate.year,caseDate.month,caseDate.day)
    currDate = "{:04d}{:02d}{:02d}".format(timeStep.year,timeStep.month,timeStep.day)
    hour = "{:02d}".format(timeStep.hour)
    dateTime = "{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}".format(timeStep.year,timeStep.month,timeStep.day,timeStep.hour,timeStep.minute,timeStep.second)
    
    fields = []
    for idx, field in enumerate(fields3D):
        try:
            name = casesPath+caseDate+"\\multi0\\"+field+"\\full3D\\"+dateTime+".netcdf"
            f = Dataset(name,mode='r')
        except:
            caseDate = datetime.strptime(caseDate,'%Y%m%d')
            caseDate = caseDate + timedelta(days=-1)
            caseDate = "{:04d}{:02d}{:02d}".format(caseDate.year,caseDate.month,caseDate.day)             
            name = casesPath+caseDate+"\\multi0\\"+field+"\\full3D\\"+dateTime+".netcdf"
            f = Dataset(name,mode='r')                  

        var = f.variables[field][:,:,:]
        # Specifications for difficult variables
        if feature == "MARC" and field == "MergedDivShear":
                var = np.where(var<-10,0,var)
                var = var*-1
        # if field == "MergedDRHO":
        #     if threshold[idx] < 0:
        #         pass
        #     else:
        #         maxField = np.ones(var.shape)
        #         var = maxField - var
        if feature == "hail volume" and field == "MergedDZDR":
            maxField = np.ones(var.shape)*4
            var = maxField - var
        fields.append(var)
    return fields

def getHeights(timeStep, caseDate, fields3D, temp):
    caseDate = "{:04d}{:02d}{:02d}".format(caseDate.year,caseDate.month,caseDate.day)
    currDate = "{:04d}{:02d}{:02d}".format(timeStep.year,timeStep.month,timeStep.day)
    hour = "{:02d}".format(timeStep.hour)
    dateTime = "{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}".format(timeStep.year,timeStep.month,timeStep.day,timeStep.hour,timeStep.minute,timeStep.second)

    zone = "above"
    heights = []
    for idx, height in enumerate(temp):
        # insert code to convert time. should be in EchoTop
        if height == 'surface':
            try:
                name = casesPath+caseDate+"\\multi0\\"+fields3D[idx]+"\\full3D\\"+dateTime+".netcdf"
                f = Dataset(name,mode='r')
            except:
                caseDate = datetime.strptime(caseDate,'%Y%m%d')
                caseDate = caseDate + timedelta(days=-1)
                caseDate = "{:04d}{:02d}{:02d}".format(caseDate.year,caseDate.month,caseDate.day)
                name = casesPath+caseDate+"\\multi0\\"+fields3D[idx]+"\\full3D\\"+dateTime+".netcdf"
                f = Dataset(name,mode='r') 

            var = f.variables[fields3D[idx]][:,:,:]
            heights.append(np.zeros(var.shape[1:]))
        #elif height == "18 dBZ EchoTop":
            #heights.append(findEchoTop(casesPath,caseDate,"MergedReflectivityQC",dateTime,18))
        else:
            if temp[idx][0] == "-":
                zone = "below"          # if the negative sign appears, we are looking below the height.
                height = temp[idx][1:]
            try:
                name = shavePath+caseDate+"\\NSE\\Heightof"+height+"C\\nseanalysis\\"+currDate+"-"+hour+"0000.netcdf"
                f = Dataset(name,mode='r')
            except:
                caseDate = datetime.strptime(caseDate,'%Y%m%d')
                caseDate = caseDate + timedelta(days=-1)
                caseDate = "{:04d}{:02d}{:02d}".format(caseDate.year,caseDate.month,caseDate.day)
                name = shavePath+caseDate+"\\NSE\\Heightof"+height+"C\\nseanalysis\\"+currDate+"-"+hour+"0000.netcdf"
                f = Dataset(name,mode='r')                 
            var = f.variables["Heightof"+height+"C"][:,:]/1000
            heights.append(var)
    return heights, zone    

def mask(fields, heights, thresholds, zone):
    '''
    consider allowing whole field input
    this masks the 3D field that falls below a threshold above a given temp
    '''
    top = np.ones(len(fields))*maxHeight  # create ceiling for the space. Can be ammended by changing maxHeight, as a var in the setFeature function

    masks = []
    for idx, field in enumerate(fields):
        if zone == "below":
            hgtMask = np.arange(field.shape[0])[:,None,None]<heights[idx]
        else:
            hgtMask = np.arange(field.shape[0])[:,None,None]>heights[idx]
            topHeights = np.ones((heights[0].shape))*top[idx]
            topMask = np.arange(field.shape[0])[:,None,None]<topHeights[idx] # filter for heights above the specified top height
            hgtMask = hgtMask * topMask
        fieldMask = np.ma.masked_greater(field,thresholds[idx]).mask
        masks.append([hgtMask,fieldMask])
       
    return masks

def getNew()
    mask both

def getGrid(timeStep, caseDate, var):
    caseDate = "{:04d}{:02d}{:02d}".format(caseDate.year,caseDate.month,caseDate.day)
    currDate = "{:04d}{:02d}{:02d}".format(timeStep.year,timeStep.month,timeStep.day)
    hour = "{:02d}".format(timeStep.hour)
    dateTime = "{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}".format(timeStep.year,timeStep.month,timeStep.day,timeStep.hour,timeStep.minute,timeStep.second)

    file = casesPath+caseDate+"\\finished0"
    words = []
    try:
        with open(casesPath+caseDate+'\\finished0','r') as f:
            for line in f:
                words.extend(line.split())
    except:
        caseDate = datetime.strptime(caseDate,'%Y%m%d')
        caseDate = caseDate + timedelta(days=-1)
        caseDate = "{:04d}{:02d}{:02d}".format(caseDate.year,caseDate.month,caseDate.day)        
        with open(casesPath+caseDate+'\\finished0','r') as f:
            for line in f:
                words.extend(line.split())

    latNW = float(words[0])
    lonNW = float(words[1])
    latSE = float(words[2])
    lonSE = float(words[3])
    latLon = np.ones((var.shape[1],var.shape[2],2))
    spacing = 0.005
    for i in range(0,var.shape[1]):
        for j in range(0,var.shape[2]):
            latLon[i,j,0] = latNW - spacing*i
            latLon[i,j,1] = lonNW + spacing*j
    return latLon

def getMax(subgraphs, field):
    '''
    input: either the graph or the mesh
    output: max value and 3D index and the graph index
    '''
    var = field
    i = 0
    max_Point = 0
    for idx, subgraph in enumerate(subgraphs):
        i=i+1
        verts = subgraph[0]
        faces = subgraph[1]
        sub = verts[faces]
        maxVal = -10000000
        for group in sub:
            for point in group:
                lon = int(point[0])
                lat = int(point[1])
                hgt = int(point[2])
                val = var[hgt][lat][lon]
                if val > maxVal:
                    maxVal = val
                    max_Point = point
                    graph_index = idx
    return maxVal, max_Point, idx


def getCorrespondingData(fields, point):
    '''
    input: var, 3D index and fields sought
    output: [values of fields sought]
    '''
    lon = int(point[0])
    lat = int(point[1])
    hgt = int(point[2])

    values = []
    for idx, field in enumerate(fields):
        values.append(field[hgt][lat][lon])
    return values

def getDepth(graph):
    '''
    input: graph
    output: max depth
    '''
    verts = graph[0]
    faces = graph[1]
    sub = verts[faces]
    top = 0
    bot = 1000
    for group in sub:
        for point in group:
            hgt = int(point[2])
            if hgt > top:
                top = hgt
            if hgt < bot:
                bot = hgt
    depth = top - bot
    return depth

def getCentroid(graph):
    '''
    input: graph
    output: centroid
    '''
    sumLon = 0
    sumLat = 0
    sumHgt = 0
    verts = graph[0]
    faces = graph[1]
    sub = verts[faces]
    n = 0
    for group in sub:
        for point in group:
            lon = int(point[0])
            lat = int(point[1])
            hgt = int(point[2])    
            sumLon = sumLon + lon
            sumLat = sumLat + lat 
            sumHgt = sumHgt + hgt         
            n=n+1
    
    centroid = [sumLon/n, sumLat/n, sumHgt/n]          
    return centroid

def findEchoTop(casesPath,caseDate,field,dateTime,threshold):
    name = casesPath+caseDate+"\\multi0\\"+field+"\\full3D\\"+dateTime+".netcdf"
    f = Dataset(name,mode='r')       
    var = f.variables[field][:,:,:]
    echoTops = np.ones((var.shape[1],var.shape[2]))*-999    # initialize NP array with fill value -999
    for i in range(0,var.shape[1]):
        for j in range(0,var.shape[2]):
            for k in np.arange(var.shape[0]-1,0,-1):   # work down from top
                if var[k][i][j] > threshold:                # check if above dBZ threshold
                    echoTops[i][j] = k                      # if so, break out and move to the next lat,lon
                    break
    return echoTops

def filterForHeight(fields,masks,height):
    heights = np.ones
    hgtMask = np.arange(fields[0])[:,None,None]<height
    masks.append(hgtMask)
    return masks

def createMesh(graphs, feature):
    mesh_container = []
    object_container = []
    nets = []
    subgraphs = []
    for graph in graphs:
        verts = graph[0]
        faces = graph[1]
        linelist = []
        for idx, vert in enumerate(faces):  
            for i,x in enumerate(vert):
                l = [np.ndarray.tolist(verts[faces[idx][i]]), np.ndarray.tolist(verts[faces[idx][(i+1)%len(vert)]])] # connect the verts of the triangle
                linelist.append(l)  # add to the line list

        nets.append([verts,faces])

        tmp = [tuple(tuple(j) for j in i) for i in linelist]
        graph = nx.Graph(tmp)
        graphs = []
        for idx, graph in enumerate(sorted(nx.connected_components(graph),key = len, reverse = True)):
            graphs.append((graph))

        edges = []
        for face in faces:
            edges.extend(list(itertools.combinations(face, 2)))
        g = nx.from_edgelist(edges)

        components = list(nx.algorithms.components.connected_components(g))
        component_to_faces = dict()
        for component in components:
            component_to_faces[tuple(component)] = [face for face in faces if set(face) <= component] # <= operator tests for subset relations 

        objects = []
        meshes = []
        subgraphs = []
        for component, component_faces in component_to_faces.items():
            subgraphs.append([verts,np.asarray(component_faces)])
            if feature == "30 dBZ outline":
                meshes.append(Poly3DCollection(verts[np.asarray(component_faces)],alpha=0.1))
            else:
                meshes.append(Poly3DCollection(verts[np.asarray(component_faces)],alpha=0.3))
            objects.append(component_faces)
        mesh_container.append(meshes)
        object_container.append(objects)
    return mesh_container, object_container, subgraphs
    # subgraphs[0] contains the verts and faces of the first connected graph

def createGraphs(fields):
    nets = []
    graphs = []
    for field in fields:
        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(field.transpose(),0)
        except:
            print("An exception has occurred, field below threshold everywhere.")
            verts, faces, normals, values = measure.marching_cubes_lewiner(A.transpose(),0)
        
        graphs.append([verts, faces])
        nets.append([verts,faces])
    return nets, graphs # nets = graphs, so simplify

def getVolume(nets, subgraphs):
    volumes = []
    for subgraph in subgraphs:
        verts = subgraph[0]
        faces = subgraph[1]
        points = np.array(verts)  
        cells = [("triangle", np.array(faces))]
        meshio.write_points_cells('out.vtu',points,cells)
        #vMesh = pygalmesh.generate_volume_mesh_from_surface_mesh("out.vtu") unnecessary, removal halves the time
        mesh = meshplex.MeshTri(points,np.array(faces))
        # mask cell volume below some threshold here
        V = sum(mesh.cell_volumes*convUnit)
        #V = mesh.cell_volumes*convUnit
        #if V < 0.5:
        #    V = 0
        volumes.append(V)
    return volumes

def track(subgraphs, latLon, currTime, caseDate):
    df = pd.read_csv(casesPath + "/" + caseDate + "/features.csv")
    time = df["EpochTime"]
    for idx, item in enumerate(time):
        time[idx] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item))
    t = nearest(time, datetime.strptime(currTime, DATE_TIME_FORMAT))
    idx = np.where(time==t)[0][0]

    # now we grab the storm's coord using the time-index
    lat = df["Latitude"][idx]
    lon = df["Longitude"][idx]
    coord = [lat,lon]
    nearGraphs = []
    print("beginning number of graphs:",len(subgraphs))
    i=1
    for idx, subgraph in enumerate(subgraphs):
        i=i+1
        print(i)
        verts = subgraph[0]
        faces = subgraph[1]
        sub = verts[faces]
        minDist = 10
        for group in sub:
            for point in group:
                lon = int(point[0])
                lat = int(point[1])
                pos = latLon[lat,lon]
                dist = distance(pos,coord)
                if dist < minDist:
                    minDist = dist
        if minDist > maxDist:
           #print("woah",idx)
            pass
        else:
            print("hey there!")
            nearGraphs.append(subgraph)
    print("Final number of graphs:", len(nearGraphs))
    return nearGraphs
            
def distance(pos,coord):
    return m.sqrt((pos[0]-coord[0])**2 + (pos[1]-coord[1])**2)   

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def display(mesh_containers, objects, type):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r','b','g','c','y','m','w']
    #meshes[0][0][0].set_facecolor('r')
    if type == 'zdr_column':
        color = 'r'
    for idx, mesh_container in enumerate(mesh_containers):
        colidx = idx % len(colors)
        print("index:",idx)
        for i, meshes in enumerate(mesh_container):
            for index, mesh in enumerate(meshes):
                mesh.set_facecolor(colors[colidx])
                mesh.set_edgecolor('k')
                ax.add_collection3d(mesh)
        print("loop complete")

    print(len(mesh_containers[0][0]))    
    ax.set_xlim(0,100)
    ax.set_ylim(0,100)
    ax.set_zlim(0,8)
    plt.tight_layout()
    plt.show()

def plot_df(df,xlabel,ylabel):
    plt.plot(df[xlabel],df[ylabel])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def update_timestep(time, feature):
    startTime = time.time()
    feature = feature
    time = datetime.strptime(time, DATE_TIME_FORMAT)
    fields, thesh, heights, topHeights, interest, feat = setFeature(feature)

#     depth_title = feature + '_depth'
#     width_title = feature + '_width'
#     fetVolume = feature + '_feature_volume'
#     centroid_title = feature + '_centroid'   
# 'time':timeStep,'volumes':[V], 'V':sum(V),title:[maxValue], depth_title:[depth],width_title:[width],fetVolume:[specificVol],centroid_title:[centroid],'meshes':[meshes],'objects':[objects]}))
#     col = ["time", "volumes", "V", ""]
#     df = pd.DataFrame(columns=col)

#def getData(timeStep, caseDate, chosenFields,chosenHeights):



def getData(caseDate):
    daymonthyear = caseDate[:8] 
    caseDate = datetime.strptime(caseDate,DATE_FORMAT)
    done = 0
    i = 0
    df = buildDataFrame(features_list)
    path = "C:\\Users\\User\\weather\\cases\\20130515\\multi0\\MergedDZDR\\full3D\\"

    #path = "/mnt/data/SHAVE_cubes/" + daymonthyear + "/" + "multi" + dictionary[key] + "/" + variable  + "/full3D"

    df_shave = pd.read_excel("/mnt/data/SHAVE_cubes/" + caseDate + "/features.csv") #daymonthyear goes here
    times = [file[:15] for file in os.listdir(path)]
    for idx, item in enumerate(times):
        times[idx] = datetime.strptime(item, DATE_TIME_FORMAT) 

    for ind in df_shave.index:
        t_0 = df_shave["0.5DegTiltTime"][ind]
        t_0 = datetime.strptime(t_0, DATE_TIME_FORMAT)
        timeStep = nearest(times, t_0)
        values = []
        values.append(timeStep)

        for idx, feature in enumerate(features):
            # build dataframe
            presets = setFeature(idx)[0]    # set the feature
            currentTime = "{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}".format(timeStep.year,timeStep.month,timeStep.day,timeStep.hour,timeStep.minute,timeStep.second)
            fields = getFields(timeStep, caseDate, chosenFields, feature)               # fields = 
            heights, zone = getHeights(timeStep, caseDate, chosenFields, chosenHeights)
            masks = mask(fields, heights, chosenThresholds, zone)                       # here we get the mask
            grid = getGrid(timeStep, caseDate, fields[0])                               # now we get the lat-lon grid based on the shape of var
            meshes = []
            objects = []
            finalField = np.ones((fields[0].shape))     

            for idx, field in enumerate(fields):
                field = field*masks[idx][0]*masks[idx][1]                               # masks multiply fields to get the masked field
                finalField = field*finalField

            nets, graphs = createGraphs([finalField])                                   # this is run through the graphing algorithm to retrieve graphs of the objects
            mesh_container, object_container, subgraphs = createMesh(graphs, feature)   # meshes are created for plotting, subgraphs for tracking

            if tracking == 1:
                subgraphs_2 = track(subgraphs, grid, currentTime, caseDate)
                mesh_container, object_container, trash = createMesh(subgraphs_2, feature)   

            meshes = []
            meshes.append(mesh_container)   # these are the graphs 
            objects = []
            objects.append(object_container)
            # display(meshes, objects, 'zdr_column')

            V = getVolume(nets, subgraphs)  # gather volume


            field = fields[chosenFields.index(interest)]
            maxValue, maxPoint, index  = getMax(subgraphs, field)
            corr_data = getCorrespondingData(fields, maxPoint)
            depth = getDepth(subgraphs[index])
            centroid = getCentroid(subgraphs[index])
            width = V[index]/depth
            specificVol = V[index]

            # To get the volume of the subgraph containing the maxValue:
            # 

            # Find the largest value pixel and its volume
            # retrieve statistics on corresponding values at that pixel
            # retrieve statistics on the nature of that volume
            # (centroid, depth)
            # record statistics on the feature, # of volumes, total volume
            #for feature in features:
            # max value, corresponding data at that max value

            values.append(V)
            values.append(len(V))
            values.append(sum(V))
            values.append(maxValue)
            values.append(depth)
            values.append(width)
            values.append(specificVol)
            values.append([centroid])
            for idx, datum in enumerate(corr_data):
                values.append(corr_data[idx])
            values.append([meshes])
            values.append([objects])  # all the data for this feature
        print("length:",len(values))
        print(len(df.columns))
        df.loc[i] = values
        timeStep = timeStep + timedelta(minutes=1)
        i=i+1   
        if i == 4:
            return df     
    return df

# def get_Dates():
#     cases = []
#     for key in dictionary:
#         # key = date, dictionary[key] = number
#         name = "multi" + dictionary[key] 
#     start_dir = rootdir + "/" + key 
#     if find_dir(name,start_dir) != None:
#         start_dir = start_dir + "/" + name
#         name = "MergedDZDR"
#         if find_dir(name,start_dir) != None:    
#             cases.append(key)
#             f = rootdir + "/" + key + "/" + "finished" + dictionary[key]

def main():
    # get the list of casedates
    # get the list of start time and end-time
    for date in dates:
        caseDate = date
        df = getData(caseDate)
        df2 = pd.read_excel("/mnt/data/SHAVE_cubes/" + caseDate + "/features.csv",sep="|")
        df2 = df2.iloc[:4]
        result = pd.concat([df2, df], axis=1, sort=False)
        result.to_csv("/mnt/data/SHAVE_cubes/" + caseDate + "/feature_stats.csv")
    #    df.to_csv('zdr_columns_20130515.csv')



main()






statistics = []
data = []
# for feature in features:
#     df, df2 = main(start, end, feature)
#     df3 = df[['time','volumes',feature+'_depth',feature+'_centroid']].reset_index()
#     df4 = df2.join(df3,how='inner')
#     data.append(df)
#     statistics.append(df4)
# print(statistics[0])


# statistics[0].to_excel("C:\\Users\\User\\weather\\cases\\20130515\\statistics.xlsx")

# df = pd.read_excel('C:\\Users\\User\\weather\\cases\\20130515\\tracking_data.xlsx')

# for column in statistics[0]:
#     df[column] = 0

# for index, row in df.iterrows():
#     time = df["0.5DegTiltTime"][index]
#     time = datetime.strptime(time, DATE_TIME_FORMAT)
#     t = nearest(statistics[0]["time"].tolist(),time)
#     idx = np.where(statistics[0]["time"]==t)[0][0]

#     for column in statistics[0]:
#         try:
#             df[column].loc[index] = statistics[0][column][idx]
#         except:
#             df[column] = df[column].astype(object)
#             df[column].loc[index] = statistics[0][column][idx]

# df.to_excel("C:\\Users\\User\\weather\\cases\\20130515\\statistics.xlsx")
#     # add statistics for nearest timestep




fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for idx, feature in enumerate(features):
    #rowNumber = data[idx].loc[data[idx]['time'] == end].index
    for index, mesh_container in enumerate(data[idx].iloc[0]['meshes']):
        for i, meshes in enumerate(mesh_container):
            for j, mesh in enumerate(meshes):
                print(mesh)
                if feature == "30 dBZ outline":
                    mesh.set_facecolor(clr_dict[feature])
                    mesh.set_edgecolor('w')
                else:
                    mesh.set_facecolor(clr_dict[feature])
                    mesh.set_edgecolor('k')
                ax.add_collection3d(mesh)

ax.set_xlim(0,170)
ax.set_ylim(0,108)
ax.set_zlim(0,20)
plt.tight_layout()
plt.show()


#    display(data[idx]['meshes'][0],data[idx]['objects'][0],data[idx]['feature'][0])

def display(mesh_containers, objects, type):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r','b','g','c','y','m','w']
    #meshes[0][0][0].set_facecolor('r')
    if type == 'zdr_column':
        color = 'r'
    for idx, mesh_container in enumerate(mesh_containers):
        colidx = idx % len(colors)
        print("index:",idx)
        for i, meshes in enumerate(mesh_container):
            for index, mesh in enumerate(meshes):
                mesh.set_facecolor(colors[colidx])
                mesh.set_edgecolor('k')
                ax.add_collection3d(mesh)

    print(len(mesh_containers[0][0]))    
    ax.set_xlim(0,100)
    ax.set_ylim(0,100)
    ax.set_zlim(0,8)
    plt.tight_layout()
    plt.show()

#main()

# if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description='Finds volume of fields meeting thresholds above a given height, creates a time-ordered csv of these volumes.')
#    parser.add_argument("--ds", type=str, nargs=1, help='Start time of dataset, format: 20130515-220000.')
#    parser.add_argument("--de", type=str, nargs=1, help='End time of dataset.')
#    parser.add_argument("--dv", metavar="variables",   type=str, nargs='*',
#                         default = [], action = 'append', help='List of variables, e.g. MergedReflectivityQC MergedDZDR.')
#    parser.add_argument("--dt", metavar='thresholds', type = float, nargs = '*',
#                         default = [], action = 'append', help = "List of thresholds, e.g. 0.5 1.0.")
#    parser.add_argument("--dh", metavar='heights', type=str, nargs='*',
#                         default = [], action='append', help='Temperature height threshold variable must be above, e.g. 0 0.')
#    args = parser.parse_args(argv[1:])
   

       
#    main(args.ds, args.de, args.dv,args.dt,args.dh)

# command = "python mask.py --ds 20130515-220100 --de 20130516-013000 --dv MergedDZDR MergedReflectivityQC --dt 0.5 50 -dh 0 0"
# os.system(command)
#main()

#zdr = oldMask(date,time,"MergedDZDR",0.5,["0"])
#nets, mesh_container, object_container = volume_display([zdr])
#V = getVolume(nets)
#print(V)
#data = mask(date,time,["MergedReflectivityQC","MergedDZDR"],10,"0")

