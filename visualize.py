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

choice = 2

colors = ['r','b','g','m','y','w','k','y','b','b','y']
featuresList = ["zdrColumn", "kdpColumn", "hail volume", "zdrArc", "zdrRing", "rhoRing", "30 dBZ outline","mesocyclone","MARC","storm top divergence","kdpEnhancement"]

clr_dict = {featuresList[i]: colors[i] for i in range(len(featuresList))}


np.set_printoptions(threshold=sys.maxsize)

#SHAVE_path = "/mnt/data/SHAVE_cases/"
casesPath = "C:\\Users\\User\\weather\\cases\\"
date = "20150807"
time1 = "23"
time2 = "235400"
field3D = "MergedDZDR"
convUnit = 0.5*0.5*1 
DATE_TIME_FORMAT = '%Y%m%d-%H%M%S'
maxHeight = 20
maxDist = 0.08
tracking = 1

# field to feed into marching_cubes in case of exception
A = np.zeros((3,3,3))
for i in np.arange(0,1):
    for j in np.arange(0,1):
        for k in np.arange(0,1):
            A[i,j,k] = 1

def setFeature(chosen):
    global chosenFields, chosenThresholds, chosenHeights
    if chosen == "zdrColumn":
        chosenFields = ["MergedDZDR", "MergedReflectivityQC"]
        chosenThresholds = [0.5, 50]
        chosenHeights = ["0","0"]
        topHeights = ["0","0","0"]
    if chosen == "kdpColumn":
        chosenFields = ["MergedDKDP", "MergedReflectivityQC"]
        chosenThresholds = [2, 50]
        chosenHeights = ["surface","surface"]
        topHeights = ["0","0","0"]
    if chosen == "kdpEnhancement":
        chosenFields = ["MergedDKDP", "MergedReflectivityQC","MergedDRHO"]
        chosenThresholds = [2, 50, 0.93] 
        chosenHeights = ["surface","surface","surface"]
        topHeights = ["0","0","0"]        
    if chosen == "zdrArc":
        chosenFields = ["MergedDZDR", "MergedReflectivityQC", "MergedDRHO"]
        chosenThresholds = [2, 40, -0.85] 
        chosenHeights = ["-0","-0","-0"] 
        topHeights = ["-0","-0","-0"]
    if chosen == "zdrRing":
        chosenFields = ["MergedDZDR", "MergedReflectivityQC"]
        chosenThresholds = [1, 40]
        chosenHeights = ["0","0"]
        topHeights = ["0","0","0"]
    if chosen == "rhoRing":
        chosenFields = ["MergedDRHO", "MergedReflectivityQC"]
        chosenThresholds = [0.96, 40]
        chosenHeights = ["0","0"] 
        topHeights = ["0","0","0"]
    if chosen == "hail volume":
        chosenFields = ["MergedDZDR", "MergedReflectivityQC"]
        chosenThresholds = [4, 45]
        chosenHeights = ["-0","-0"]
        topHeights = ["0","0","0"] 

    if chosen == "mesocyclone":
        chosenFields = ["MergedAzShear", "MergedReflectivityQC"]
        chosenThresholds = [0.004,20]
        chosenHeights = ["0","0"]
        topHeights = ["0","0","0"] 



    if chosen == "30 dBZ outline":
        chosenFields = ["MergedReflectivityQC"]
        chosenThresholds = [50]
        chosenHeights = ["surface"]
        topHeights = ["0","0","0"]
    if chosen == "MARC":
        chosenFields = ["MergedDivShear"]
        chosenThresholds = [0.004]
        chosenHeights = ["0"]
        topHeights = ["0"]
    # maybe make dbz < 25
    if chosen == "storm top divergence":
        chosenFields = ["MergedDivShear"]
        chosenThresholds = [0.004]
        chosenHeights = ["0"]
        topHeights = ["0"]                

    return chosenFields, chosenThresholds, chosenHeights

if choice == 1:
    features = ["30 dBZ outline"]
if choice == 2:
    features = ["30 dBZ outline","mesocyclone",]
# features =  ["zdrColumn", "kdpColumn","zdrRing","30 dBZ outline", "hail volume"]
# features = ["storm top divergence","30 dBZ outline","zdrColumn","hail volume"]
# features = ["storm top divergence","kdpEnhancement","zdrColumn"]
#features = ["mesocyclone"]
#features = ["30 dBZ outline"]
#features = ["zdrColumn"]
#features = [""]
#features= ["kdpEnhancement"]
def oldMask(date, time, fields3D, threshold, temp):
    '''
    consider allowing whole field input
    this masks the 3D field that falls below a threshold above a given temp
    '''
    name1 = casesPath+date+"\\NSE\\Heightof"+"0"+"C\\nseanalysis\\"+date+"-"+time+"0000.netcdf"
    name2 = casesPath+date+"\\multi0\\"+fields3D+"\\full3D\\"+date+"-"+time+"1600.netcdf"

    f1 = Dataset(name1,mode='r')
    f2 = Dataset(name2,mode='r')
    minHeight = f1.variables['Heightof0C'][:,:]/1000 # (lat, lon)
    var = f2.variables[fields3D][:,:,:]          # (ht, lat, lon)

    # if it's 1D, we start with idx=0, check hgtof0C, then go up until 
    # we are above hgtof0C. Once above, check ZDR
    # if either condition is unmet, set = to 0
    # for i in np.arange(0,nLat,1):
    #     for j in np.arange(0,nLon,1):
    #         levelof0 = minHeight[i,j]
    #         print(levelof0)
    #         for hgt, value in enumerate(var):
    #             if hgt < levelof0:
    #                 var[hgt,i,j] = 0
    hgtMask = np.arange(var.shape[0])[:,None,None]>minHeight
    zdrMask = np.ma.masked_greater(var,0.5).mask
    zdrAccepted = var*zdrMask*hgtMask
    return zdrAccepted

#make it so we take the last hour for heightof0C
# time refers to an hour
# sample command:
# mask(20130515,23, ["MergedDZDR", "MergedDRHO"], [1.0, -0.99], ["Heightof0C","Heightof0C"])
# ask Kiel how you could make it work for both Heightof0C or 01.00
# could this be shrunk into two loops for efficiency? 
def mask(timeStep,caseDate,fields3D, threshold, temp, feature):
    '''
    consider allowing whole field input
    this masks the 3D field that falls below a threshold above a given temp
    '''
    caseDate = "{:04d}{:02d}{:02d}".format(caseDate.year,caseDate.month,caseDate.day)
    currDate = "{:04d}{:02d}{:02d}".format(timeStep.year,timeStep.month,timeStep.day)
    hour = "{:02d}".format(timeStep.hour)
    dateTime = "{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}".format(timeStep.year,timeStep.month,timeStep.day,timeStep.hour,timeStep.minute,timeStep.second)
    top = np.ones(len(temp))*maxHeight

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
                name = casesPath+caseDate+"\\NSE\\Heightof"+height+"C\\nseanalysis\\"+currDate+"-"+hour+"0000.netcdf"
                f = Dataset(name,mode='r')
            except:
                caseDate = datetime.strptime(caseDate,'%Y%m%d')
                caseDate = caseDate + timedelta(days=-1)
                caseDate = "{:04d}{:02d}{:02d}".format(caseDate.year,caseDate.month,caseDate.day)
                name = casesPath+caseDate+"\\NSE\\Heightof"+height+"C\\nseanalysis\\"+currDate+"-"+hour+"0000.netcdf"
                f = Dataset(name,mode='r')                 
            var = f.variables["Heightof"+height+"C"][:,:]/1000
            heights.append(var)

    # Read in 3D fields
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

    # Read in heights

    masks = []
    for idx, field in enumerate(fields):
        if zone == "below":
            hgtMask = np.arange(var.shape[0])[:,None,None]<heights[idx]
        else:
            hgtMask = np.arange(var.shape[0])[:,None,None]>heights[idx]
            topHeights = np.ones((heights[0].shape))*top[idx]
            topMask = np.arange(var.shape[0])[:,None,None]<topHeights[idx] # filter for heights above the specified top height
            hgtMask = hgtMask * topMask
        fieldMask = np.ma.masked_greater(field,threshold[idx]).mask
        masks.append([hgtMask,fieldMask])

    #generate lat,lon grid
    file = casesPath+caseDate+"\\finished0"
    words = []
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
    np.set_printoptions(threshold=sys.maxsize)        
    #print(latLon[107][169][1])

    return fields,masks, latLon

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

def track(subgraphs, latLon, currTime):
    df = pd.read_excel("C:\\Users\\User\\weather\\cases\\20130515\\tracking_data.xlsx")
    time = df["0.5DegTiltTime"]
    for idx, item in enumerate(time):
        time[idx] = datetime.strptime(item, DATE_TIME_FORMAT) 
    t = nearest(time, datetime.strptime(currTime, DATE_TIME_FORMAT))
    idx = np.where(time==t)[0][0]

    # now we grab the storm's coord using the time-index
    lat = df["Latitude(degrees)"][idx]
    lon = df["Longitude(degrees)"][idx]
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
    ax.set_xlim(20,80)
    ax.set_ylim(20,100)
    ax.set_zlim(0,8)
    plt.title("ZDR Column")
    plt.tight_layout()
    plt.show()


def plot_df(df,xlabel,ylabel):
    plt.plot(df[xlabel],df[ylabel])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

#    for hgt, place in enumerate(var):
def main(start_Time, end_Time, feature):
    #setFeature('zdrColumn')
    startTime = time.time()
    feature = feature
    endTime = datetime.strptime( end_Time, DATE_TIME_FORMAT)
    timeStep = datetime.strptime(start_Time, DATE_TIME_FORMAT)
    caseDate = datetime.strptime(start_Time, DATE_TIME_FORMAT)
    topHeights = [20,20]
    fields, thesh, heights = setFeature(feature)
    df = pd.DataFrame([])

    # startDate = 20130515-230100
    # endDate =   20130516-013000
    done = 0
    i = 0
    while done == 0:
        # create topHeight filter
        # count the number of columns and their volume (perhaps)
        # check paper to see what the dBZ threshold should be 
        currentTime = "{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}".format(timeStep.year,timeStep.month,timeStep.day,timeStep.hour,timeStep.minute,timeStep.second)

        fields, masks, latLon = mask(timeStep,caseDate,chosenFields ,chosenThresholds,chosenHeights,feature) # matrices[0][0] = field, matrices[0][1][0] = fieldMask, matrices[0][1][1] = hgtMask
        #    masks = filterForHeight(fields,masks,10)
        meshes = []
        objects = []
        finalField = np.ones((fields[0].shape))
        for idx, field in enumerate(fields):
            field = field*masks[idx][0]*masks[idx][1]
            finalField = field*finalField
        nets, graphs = createGraphs([finalField])
        mesh_container, object_container, subgraphs = createMesh(graphs, feature)
        if tracking == 1:
            graphList = track(subgraphs, latLon, currentTime)
            mesh_container, object_container, trash = createMesh(graphList, feature)        
        meshes = []
        meshes.append(mesh_container)   # these are the graphs 
        objects = []
        objects.append(object_container)
        #display(meshes, objects, 'zdr_column')


        V = getVolume(nets, subgraphs)
        df = df.append(pd.DataFrame({'time':timeStep,'volumes':[V], 'V':sum(V),'meshes':[meshes],'objects':[objects],"feature":feature}))
        timeStep = timeStep + timedelta(minutes=1)
        i=i+1
        if timeStep == endTime:
            done = 1
    print(df)
        
    return df

#    df.to_csv('zdr_columns_20130515.csv')
    # print("--- %s seconds ---" % (time.time() - startTime))


start ="20130515-235300"
end =  "20130515-235400"


data = []
for feature in features:
    df = main(start, end, feature)
    plt.plot(df['time'],df['V'],clr_dict[feature])
    plt.title(feature)
    data.append(df)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

for idx, feature in enumerate(features):
    
    #rowNumber = data[idx].loc[data[idx]['time'] == end].index
    print(data[idx].iloc[0])
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

ax.set_xlim(80,130)
ax.set_ylim(20,70)
ax.set_zlim(0,20)
plt.tight_layout()
plt.show()


#    display(data[idx]['meshes'][0],data[idx]['objects'][0],data[idx]['feature'][0])

# def display(mesh_containers, objects, type):
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     colors = ['r','b','g','c','y','m','w']
#     #meshes[0][0][0].set_facecolor('r')
#     if type == 'zdr_column':
#         color = 'r'
#     for idx, mesh_container in enumerate(mesh_containers):
#         colidx = idx % len(colors)
#         print("index:",idx)
#         for i, meshes in enumerate(mesh_container):
#             for index, mesh in enumerate(meshes):
#                 mesh.set_facecolor(colors[colidx])
#                 mesh.set_edgecolor('k')
#                 ax.add_collection3d(mesh)

#     print(len(mesh_containers[0][0]))    
#     ax.set_xlim(0,100)
#     ax.set_ylim(0,100)
#     ax.set_zlim(0,8)
#     plt.tight_layout()
#     plt.show()

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

