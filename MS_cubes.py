from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys
from MarchingSquare import *
from MS_Graph import *
import numpy as np
from netCDF4 import Dataset
import collections
import itertools
import pprint
import networkx as nx
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=sys.maxsize)
import time
import collections
import pandas as pd
import pygalmesh
import meshio
import meshplex

#B = np.zeros((12,12))
A = np.zeros((12,12,12))
#A[A<1] = -1
for i in np.arange(0,2):
    for j in np.arange(0,2):
        for k in np.arange(0,2):
            A[i,j,k] = 10

            

for i in np.arange(8,9):
    for j in np.arange(8,9):
        for k in np.arange(8,9):
            A[i,j,k] = 10

#A[4,4,4] = 10

start_time = time.time()
#name = r"C:\localdata\multi_20190523\MergedReflectivityQC\full3D\20190523-020516.netcdf"
name = r"C:\Users\User\weather\cases\multi_20190523\MergedReflectivityQC\full3d\20190523-020516.netcdf"
#name = r"C:\Users\User\weather\cases\20130515\MergedReflectivityQC\full3D\20130515-233500.netcdf"
#name = r"C:\Users\User\weather\cases\20130515\MergedAzShear\full3D\20130516-000600.netcdf"

f = Dataset(name,mode='r')
var = f.variables['MergedReflectivityQC'][:,:,:] 
#var = f.variables['MergedAzShear'][:,:,:]

pretend = 1
if pretend:
    var = A
    threshold = 1
else:
    threshold = 40
#print(var.shape[1])
#a = np.zeros((140,140))
#res = np.vstack((a.reshape(1,140,140),var))
#print(var.shape)

verts, faces, normals, values = measure.marching_cubes_lewiner(var.transpose(),threshold)

# create linelist
#def makeLineList():
linelist = []
for idx, vert in enumerate(faces):  
    for i,x in enumerate(vert):
        l = [np.ndarray.tolist(verts[faces[idx][i]]), np.ndarray.tolist(verts[faces[idx][(i+1)%len(vert)]])] # connect the verts of the triangle
        linelist.append(l)  # add to the line list

# Creates graph
tmp = [tuple(tuple(j) for j in i) for i in linelist]
graph = nx.Graph(tmp)
graphs = []
i=0
open('output.txt','w').close()
for idx, graph in enumerate(sorted(nx.connected_components(graph),key = len, reverse = True)):
    graphs.append((graph))
    print("Graph ",idx," corresponds to vertices: ",graph,'\n\n',file=open("output.txt","a"))         
    i+=1

# put faces in the right form, also clean up this code


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
for component, component_faces in component_to_faces.items():
    meshes.append(Poly3DCollection(verts[np.asarray(component_faces)]))
    objects.append(component_faces)



print("--- %s seconds ---" % (time.time() - start_time))
print(len(graphs))

#mesh = pygalmesh.generate_mesh(verts[np.asarray(objects[0])])



fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')


# Fancy indexing: `verts[faces]` to generate a collection of triangles
# each element describes a polygon as a sequence of N_ i points (x, y, z)
#temp = faces[0].copy()
#faces[0] = faces[4]
#faces[4] = temp
#mesh = Poly3DCollection(verts[faces[0:8]])

colors = ['r','b','g','c','y','m','w']
print(len(meshes))
for idx, mesh in enumerate(meshes):
    colidx = idx % len(colors)
    mesh.set_facecolor(colors[colidx])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
 

# mesh[0].set_edgecolor('k')
# mesh[0].set_facecolor('r')
# ax.add_collection3d(mesh[0])

# mesh[1].set_edgecolor('k')
# mesh[1].set_facecolor('b')
# ax.add_collection3d(mesh[1])

points = np.array(verts)
cells = [
    ("triangle", np.array(faces))
]
meshio.write_points_cells('out.vtu',points,cells)

vMesh = pygalmesh.generate_volume_mesh_from_surface_mesh("out.vtu")

mesh = meshplex.MeshTri(points,np.array(faces))

V = sum(mesh.cell_volumes)

print(V)

# ax.set_xlim(0,140)
# ax.set_ylim(0,140)
# ax.set_zlim(0,15)

# ax.set_xlim(60,100)
# ax.set_ylim(40,80)
# ax.set_zlim(0,12)

ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.set_zlim(0,12)

plt.tight_layout()
plt.show()

# 