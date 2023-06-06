# import necessary libraries
from __future__ import division
from dolfin import *
import os
import pathlib
import pickle
import subprocess
from dolfin_utils.meshconvert import meshconvert
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI as pyMPI

def max_ufl_fenics_mesh_func(a, b):
    return (a+b+abs(a-b)) / 2.

def min_ufl_fenics_mesh_func(a, b):
    return (a+b-abs(a-b)) / 2.

def local_project(v, V, u=None):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return

def mpi4py_comm(comm):
    try:
        return comm.tompi4py()
    except AttributeError:
        return comm

def peval(f, x):
    try:
        y_local = f(x)
    except RuntimeError:
        y_local = np.inf*np.ones(f.value_shape())
    
    comm = mpi4py_comm(f.function_space().mesh().mpi_comm())
    y_global = np.zeros_like(y_local)
    comm.Allreduce(y_local, y_global, op=pyMPI.MIN)
    return y_global

def print0(text):
    if MPI.rank(MPI.comm_world) == 0:
        print(text)

def generate_savedir(namedir):
    savedir = "./"+namedir+"/"
    create_savedir(savedir)

    return savedir

def create_savedir(savedir):
    if MPI.rank(MPI.comm_world) == 0:
        if os.path.isdir(savedir) == False:
            pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
    MPI.barrier(MPI.comm_world)

def fenics_mesher(fenics_mesh, subdir, mesh_name):

    def generate_mesh_dir(subdir):
        mesh_dir = "./"+subdir+"/"+"meshes"+"/"
        create_savedir(mesh_dir)

        return mesh_dir
    
    mesh_dir = generate_mesh_dir(subdir)

    geo_mesh = XDMFFile(MPI.comm_world, mesh_dir+mesh_name+".xdmf")
    geo_mesh.write(fenics_mesh)
    fenics_mesh.init()

    return fenics_mesh

def mesh_topologier(fenics_mesh_topology, subdir, mesh_topology_name):

    def generate_mesh_topology_dir(subdir):
        mesh_topology_dir = "./"+subdir+"/"+"mesh_topologies"+"/"
        create_savedir(mesh_topology_dir)

        return mesh_topology_dir
    
    mesh_topology_dir = generate_mesh_topology_dir(subdir)

    mesh_topology = XDMFFile(MPI.comm_world, mesh_topology_dir+mesh_topology_name+".xdmf")
    mesh_topology.write(fenics_mesh_topology)

def gmsh_mesher(gmsh_file, subdir, mesh_name):
    
    def generate_mesh_dir(subdir):
        mesh_dir = "./"+subdir+"/"+"meshes"+"/"
        create_savedir(mesh_dir)

        return mesh_dir
    
    mesh_dir = generate_mesh_dir(subdir)
    temp_mesh = Mesh() # create an empty mesh object

    if not os.path.isfile(mesh_dir+mesh_name+".xdmf"):

        if MPI.rank(MPI.comm_world) == 0:

            # Create a .geo file defining the mesh
            geo_file = open(mesh_dir+mesh_name+".geo", "w")
            geo_file.writelines(gmsh_file)
            geo_file.close()

            # Call gmsh to generate the mesh file and call dolfin-convert to generate the .xml file
            try:
                subprocess.call(["gmsh", "-2", "-o", mesh_dir+mesh_name+".msh", mesh_dir+mesh_name+".geo"])
            except OSError:
                print("-----------------------------------------------------------------------------")
                print(" Error: unable to generate the mesh using gmsh")
                print(" Make sure that you have gmsh installed and have added it to your system PATH")
                print("-----------------------------------------------------------------------------")
                return
            meshconvert.convert2xml(mesh_dir+mesh_name+".msh", mesh_dir+mesh_name+".xml", "gmsh")
        
        # Convert the .msh file to a .xdmf file
        MPI.barrier(MPI.comm_world)
        mesh = Mesh(mesh_dir+mesh_name+".xml")
        geo_mesh = XDMFFile(MPI.comm_world, mesh_dir+mesh_name+".xdmf")
        geo_mesh.write(mesh)
        geo_mesh.read(temp_mesh)
    
    else:
        geo_mesh = XDMFFile(MPI.comm_world, mesh_dir+mesh_name+".xdmf")
        geo_mesh.read(temp_mesh)
    
    return temp_mesh

def save_pickle_object(savedir, object, object_filename):
    object2file = open(savedir+object_filename+'.pickle', 'wb')
    pickle.dump(object, object2file, pickle.HIGHEST_PROTOCOL)
    object2file.close()

def load_pickle_object(savedir, object_filename):
    file2object = open(savedir+object_filename+'.pickle', 'rb')
    object = pickle.load(file2object)
    file2object.close()
    return object

def latex_formatting_figure(post_processing_parameters):

    ppp = post_processing_parameters

    plt.rcParams['axes.linewidth'] = ppp["axes_linewidth"] # set the value globally
    plt.rcParams['font.family']    = ppp["font_family"]
    plt.rcParams['text.usetex']    = ppp["text_usetex"] # comment this line out in WSL2, uncomment this line in native Linux on workstation
    
    plt.rcParams['ytick.right']     = ppp["ytick_right"]
    plt.rcParams['ytick.direction'] = ppp["ytick_direction"]
    plt.rcParams['xtick.top']       = ppp["xtick_top"]
    plt.rcParams['xtick.direction'] = ppp["xtick_direction"]
    
    plt.rcParams["xtick.minor.visible"] = ppp["xtick_minor_visible"]

def save_current_figure(savedir, xlabel, xlabelfontsize, ylabel, ylabelfontsize, name):
    plt.xlabel(xlabel, fontsize=xlabelfontsize)
    plt.ylabel(ylabel, fontsize=ylabelfontsize)
    plt.tight_layout()
    plt.savefig(savedir+name+".pdf", transparent=True)
    # plt.savefig(savedir+name+".eps", format='eps', dpi=1000, transparent=True)
    plt.close()

def save_current_figure_no_labels(savedir, name):
    plt.tight_layout()
    plt.savefig(savedir+name+".pdf", transparent=True)
    # plt.savefig(savedir+name+".eps", format='eps', dpi=1000, transparent=True)
    plt.close()

def none_str2nonetype(x):
        if x == "None":
            return None
        else:
            return x