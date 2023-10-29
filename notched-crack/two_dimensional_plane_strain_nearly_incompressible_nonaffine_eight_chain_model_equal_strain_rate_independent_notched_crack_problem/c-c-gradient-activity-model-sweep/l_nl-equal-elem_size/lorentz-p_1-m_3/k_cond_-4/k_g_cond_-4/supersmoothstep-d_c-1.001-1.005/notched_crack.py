# import necessary libraries
from __future__ import division
from dolfin import *
from composite_ufjc_scission import RateIndependentScissionCompositeuFJC
from nonlocal_composite_ufjc_scission_network_fracture import (
    TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetworkNonlocalScissionNetworkFractureProblem,
    peval,
    gmsh_mesher,
    mesh_topologier,
    latex_formatting_figure,
    save_current_figure
)
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from copy import deepcopy
import textwrap

class NotchedCrack(TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetworkNonlocalScissionNetworkFractureProblem):

    def __init__(self, L, H, x_notch_point, r_notch, notch_fine_mesh_layer_level_num=2, fine_mesh_elem_size=0.001, coarse_mesh_elem_size=0.1, l_nl=0.1):

        self.L = L
        self.H = H
        self.x_notch_point = x_notch_point
        self.r_notch = r_notch
        self.notch_fine_mesh_layer_level_num = notch_fine_mesh_layer_level_num
        self.fine_mesh_elem_size = fine_mesh_elem_size
        self.coarse_mesh_elem_size  = coarse_mesh_elem_size
        self.l_nl = l_nl
        
        TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetworkNonlocalScissionNetworkFractureProblem.__init__(self)
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """
        # Two-dimensional geometry parameters
        gp = self.parameters["two_dimensional_geometry"]
        # Updated two-dimensional geometry parameters
        ugp = Parameters("two_dimensional_geometry")

        # x_notch_center_point = self.x_notch_point - self.r_notch
        x_notch_surface_point = self.x_notch_point # x_notch_center_point + np.cos(np.pi/4)*self.r_notch
        y_notch_surface_point = 0 # np.sin(np.pi/4)*self.r_notch
        notch_surface_point = (x_notch_surface_point,y_notch_surface_point)
        x_far_edge_surface_point = self.L
        y_far_edge_surface_point = 0
        far_edge_surface_point = (x_far_edge_surface_point,y_far_edge_surface_point)

        meshpoints_list = [notch_surface_point, far_edge_surface_point]
        meshpoints_num = len(meshpoints_list)
        meshpoints_label_list = []
        meshpoints_color_list = ['black', 'blue']
        meshpoints_name_list = ['notch_surface_point', 'far_edge_surface_opint']
        for meshpoint in meshpoints_list:
            meshpoint_x_coord_string = "{:.4f}".format(meshpoint[0])
            meshpoint_y_coord_string = "{:.4f}".format(meshpoint[1])
            meshpoint_label = r'$'+'('+meshpoint_x_coord_string+','+meshpoint_y_coord_string+')'+'$'
            meshpoints_label_list.append(meshpoint_label)
        
        for meshpoint_indx in range(meshpoints_num):
            ugp.add("meshpoint_indx_"+str(meshpoint_indx)+"_x_coord", meshpoints_list[meshpoint_indx][0])
            ugp.add("meshpoint_indx_"+str(meshpoint_indx)+"_y_coord", meshpoints_list[meshpoint_indx][1])
            ugp.add("meshpoint_indx_"+str(meshpoint_indx)+"_label", meshpoints_label_list[meshpoint_indx])
            ugp.add("meshpoint_indx_"+str(meshpoint_indx)+"_color", meshpoints_color_list[meshpoint_indx])
            ugp.add("meshpoint_indx_"+str(meshpoint_indx)+"_name", meshpoints_name_list[meshpoint_indx])
        
        ugp.add("meshpoint_num", meshpoints_num)
        ugp.add("meshtype", "notched_crack")
        
        gp.assign(ugp)
        
        # Material parameters
        mp = self.parameters["material"]
        # Updated material parameters
        ump = Parameters("material")

        # Fundamental material constants
        k_B          = constants.value(u"Boltzmann constant") # J/K
        N_A          = constants.value(u"Avogadro constant") # 1/mol
        h            = constants.value(u"Planck constant") # J/Hz
        hbar         = h/(2*np.pi) # J*sec
        T            = 298 # absolute room temperature, K
        beta         = 1./(k_B*T) # 1/J
        omega_0      = 1./(beta*hbar) # J/(J*sec) = 1/sec
        zeta_nu_char = 298.9 # 537.6 # 298.9 # 100 # 50
        kappa_nu     = 912.2 # 3197.5 # 912.2 # 2300 # 7500
        nu_b         = "None"
        zeta_b_char  = "None"
        kappa_b      = "None"

        ump.add("k_B", k_B)
        ump.add("N_A", N_A)
        ump.add("h", h)
        ump.add("hbar", hbar)
        ump.add("T", T)
        ump.add("beta", beta)
        ump.add("omega_0", omega_0)
        ump.add("zeta_nu_char", zeta_nu_char)
        ump.add("kappa_nu", kappa_nu)
        ump.add("nu_b", nu_b)
        ump.add("zeta_b_char", zeta_b_char)
        ump.add("kappa_b", kappa_b)

        # composite uFJC scission model
        ump.add("scission_model", "analytical")

        lmbda_nu_crit_min = "None"
        lmbda_nu_crit_max = "None"
        tau            = "None"
        lmbda_nu_check = "None"

        ump.add("lmbda_nu_crit_min", lmbda_nu_crit_min)
        ump.add("lmbda_nu_crit_max", lmbda_nu_crit_max)
        ump.add("tau", tau)
        ump.add("lmbda_nu_check", lmbda_nu_check)

        # Network-level damage
        d_c_lmbda_nu_crit_min = 1.001 # 1.001
        d_c_lmbda_nu_crit_max = 1.005 # 1.005

        ump.add("d_c_lmbda_nu_crit_min", d_c_lmbda_nu_crit_min)
        ump.add("d_c_lmbda_nu_crit_max", d_c_lmbda_nu_crit_max)

        # Non-local interaction length scale
        ump.add("l_nl", self.l_nl)

        # Define the chain segment number statistics in the network
        ump.add("nu_distribution", "uniform")

        nu_list  = [6] # nu = 5 -> nu = 15
        nu_min   = min(nu_list)
        nu_max   = max(nu_list)
        nu_num   = len(nu_list)
        nu_bar   = 6
        Delta_nu = nu_bar-nu_min

        for nu_indx in range(nu_num):
            ump.add("nu_indx_"+str(nu_indx)+"_nu_val", nu_list[nu_indx])
        ump.add("nu_min", nu_min)
        ump.add("nu_max", nu_max)
        ump.add("nu_num", nu_num)
        ump.add("nu_bar", nu_bar)
        ump.add("Delta_nu", Delta_nu)

        # Define chain segment numbers to chunk during deformation
        nu_chunks_list = nu_list
        nu_chunks_num = len(nu_chunks_list)
        nu_chunks_indx_in_nu_list = nu_chunks_list.copy()
        for nu_chunk_indx in range(nu_chunks_num):
            nu_chunks_indx_in_nu_list[nu_chunk_indx] = nu_list.index(nu_chunks_list[nu_chunk_indx])
        nu_chunks_label_list = [r'$\nu='+str(nu_list[nu_chunks_indx_in_nu_list[nu_chunk_indx]])+'$' for nu_chunk_indx in range(nu_chunks_num)]
        nu_chunks_color_list = ['blue']

        for nu_chunk_indx in range(nu_chunks_num):
            ump.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_nu_val", nu_chunks_list[nu_chunk_indx])
            ump.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_indx_val_in_nu_list", nu_chunks_indx_in_nu_list[nu_chunk_indx])
            ump.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_label", nu_chunks_label_list[nu_chunk_indx])
            ump.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_color", nu_chunks_color_list[nu_chunk_indx])

        ump.add("nu_chunks_num", nu_chunks_num)

        composite_ufjc_nu_max = RateIndependentScissionCompositeuFJC(nu=nu_list[-1],
                                                                     zeta_nu_char=zeta_nu_char,
                                                                     kappa_nu=kappa_nu)
        
        lmbda_c_crit_max = composite_ufjc_nu_max.lmbda_c_eq_crit / composite_ufjc_nu_max.A_nu
        ump.add("lmbda_c_tilde_lb", 0.0)
        ump.add("lmbda_c_tilde_ub", lmbda_c_crit_max)
        
        # Define various characteristics of the deformation for the network
        ump.add("network_model", "statistical_mechanics_model")
        ump.add("phenomenological_model", "neo_hookean")
        ump.add("physical_dimension", 2)
        ump.add("physical_dimensionality", "two_dimensional")
        ump.add("incompressibility_assumption", "nearly_incompressible")
        ump.add("macro2micro_deformation_assumption", "nonaffine")
        ump.add("micro2macro_homogenization_scheme", "eight_chain_model")
        ump.add("chain_level_load_sharing", "equal_strain")
        ump.add("rate_dependence", "rate_independent")
        ump.add("two_dimensional_formulation", "plane_strain")
        ump.add("microcircle_quadrature_order", 1)
        ump.add("microsphere_quadrature_order", 1)

        mp.assign(ump)

        # Finite element method parameters
        femp = self.parameters["fem"]

        femp["solver_algorithm"] = "monolithic" # "alternate_minimization" # "monolithic"
        femp["solver_bounded"] = False # True

        femp["u_degree"] = 2

        # Deformation parameters
        dp = self.parameters["deformation"]

        dp["deformation_type"] = "uniaxial"

        dp["K__G"] = 10
        dp["k_cond_val"] = 1.e-4
        dp["k_g_cond_val"] = 1.e-4
        dp["tol_lmbda_c_tilde_val"] = 1e-3

        dp["strain_rate"] = 0.1 # 0.2 # 1/sec
        dp["t_max"] = 4.56 # 5.28 # 6.44 # 8.32 # 6.2 # 5.8 # 4.4 # 30.0 # 33.0 # 30.0 # 13.6 # 13.5 # 16.0 # 100.0 # sec
        dp["t_step"] = 0.02 # 0.005 # 0.01 # 0.02 # sec
        dp["t_step_chunk_num"] = 2

        # Post-processing parameters
        ppp = self.parameters["post_processing"]

        ppp["save_lmbda_c_eq_chunks"] = True
        ppp["save_lmbda_nu_chunks"] = True
        ppp["save_lmbda_c_eq_tilde_chunks"] = True
        ppp["save_lmbda_nu_tilde_chunks"] = True
        ppp["save_lmbda_c_eq_tilde_max_chunks"] = True
        ppp["save_lmbda_nu_tilde_max_chunks"] = True
        ppp["save_upsilon_c_chunks"] = True
        ppp["save_d_c_mesh"] = True
        ppp["save_d_c_chunks"] = True

        ppp["save_u_chunks"] = False
        ppp["save_F_chunks"] = False
        ppp["save_sigma_chunks"] = False

        # ppp["save_lmbda_c_chunks"] = False
        # ppp["save_lmbda_c_eq_chunks"] = False
        # ppp["save_lmbda_nu_chunks"] = False
        # ppp["save_lmbda_c_tilde_chunks"] = False
        # ppp["save_lmbda_c_eq_tilde_chunks"] = False
        # ppp["save_lmbda_nu_tilde_chunks"] = False
        # ppp["save_lmbda_c_tilde_max_chunks"] = False
        # ppp["save_lmbda_c_eq_tilde_max_chunks"] = False
        # ppp["save_lmbda_nu_tilde_max_chunks"] = False
        # ppp["save_upsilon_c_chunks"] = False
        # ppp["save_Upsilon_c_chunks"] = False
        # ppp["save_d_c_chunks"] = False
        # ppp["save_D_c_chunks"] = False
        # ppp["save_epsilon_cnu_diss_hat_chunks"] = False
        # ppp["save_Epsilon_cnu_diss_hat_chunks"] = False
        # ppp["save_epsilon_c_diss_hat_chunks"] = False
        # ppp["save_Epsilon_c_diss_hat_chunks"] = False
        # ppp["save_overline_epsilon_cnu_diss_hat_chunks"] = False
        # ppp["save_overline_Epsilon_cnu_diss_hat_chunks"] = False
        # ppp["save_overline_epsilon_c_diss_hat_chunks"] = False
        # ppp["save_overline_Epsilon_c_diss_hat_chunks"] = False
        # ppp["save_u_chunks"] = False
        # ppp["save_F_chunks"] = False
        # ppp["save_sigma_chunks"] = False

        # ppp["save_lmbda_c_mesh"] = False
        # ppp["save_Upsilon_c_mesh"] = False
        # ppp["save_d_c_mesh"] = False
        # ppp["save_D_c_mesh"] = False
        # ppp["save_F_mesh"] = False
        # ppp["save_sigma_mesh"] = False

        # ppp["save_D_c_mesh"] = True
        # ppp["save_D_c_chunks"] = True

        ppp["save_lmbda_c_mesh"] = False
        ppp["save_lmbda_c_chunks"] = False
        ppp["save_lmbda_c_eq_mesh"] = False
        ppp["save_lmbda_c_eq_chunks"] = False
        ppp["save_lmbda_nu_mesh"] = False
        ppp["save_lmbda_nu_chunks"] = False
        ppp["save_lmbda_c_tilde_mesh"] = False
        ppp["save_lmbda_c_tilde_chunks"] = False
        ppp["save_lmbda_c_eq_tilde_mesh"] = False
        ppp["save_lmbda_c_eq_tilde_chunks"] = False
        ppp["save_lmbda_nu_tilde_mesh"] = False
        ppp["save_lmbda_nu_tilde_chunks"] = False
        ppp["save_lmbda_c_tilde_max_mesh"] = False
        ppp["save_lmbda_c_tilde_max_chunks"] = False
        ppp["save_lmbda_c_eq_tilde_max_mesh"] = False
        ppp["save_lmbda_c_eq_tilde_max_chunks"] = False
        ppp["save_lmbda_nu_tilde_max_mesh"] = False
        ppp["save_lmbda_nu_tilde_max_chunks"] = False
        ppp["save_g_mesh"] = False
        ppp["save_g_chunks"] = False
        ppp["save_upsilon_c_mesh"] = False
        ppp["save_upsilon_c_chunks"] = False
        ppp["save_Upsilon_c_mesh"] = False
        ppp["save_Upsilon_c_chunks"] = False
        ppp["save_d_c_mesh"] = False
        ppp["save_d_c_chunks"] = False
        ppp["save_D_c_mesh"] = False
        ppp["save_D_c_chunks"] = False
        ppp["save_epsilon_cnu_diss_hat_mesh"] = False
        ppp["save_epsilon_cnu_diss_hat_chunks"] = False
        ppp["save_Epsilon_cnu_diss_hat_mesh"] = False
        ppp["save_Epsilon_cnu_diss_hat_chunks"] = False
        ppp["save_epsilon_c_diss_hat_mesh"] = False
        ppp["save_epsilon_c_diss_hat_chunks"] = False
        ppp["save_Epsilon_c_diss_hat_mesh"] = False
        ppp["save_Epsilon_c_diss_hat_chunks"] = False
        ppp["save_overline_epsilon_cnu_diss_hat_mesh"] = False
        ppp["save_overline_epsilon_cnu_diss_hat_chunks"] = False
        ppp["save_overline_Epsilon_cnu_diss_hat_mesh"] = False
        ppp["save_overline_Epsilon_cnu_diss_hat_chunks"] = False
        ppp["save_overline_epsilon_c_diss_hat_mesh"] = False
        ppp["save_overline_epsilon_c_diss_hat_chunks"] = False
        ppp["save_overline_Epsilon_c_diss_hat_mesh"] = False
        ppp["save_overline_Epsilon_c_diss_hat_chunks"] = False
        ppp["save_u_mesh"] = False
        ppp["save_u_chunks"] = False
        ppp["save_F_mesh"] = False
        ppp["save_F_chunks"] = False
        ppp["save_sigma_mesh"] = False
        ppp["save_sigma_chunks"] = False

        ppp["save_u_mesh"] = True
        ppp["save_lmbda_c_tilde_max_mesh"] = True
        # ppp["save_lmbda_c_tilde_max_chunks"] = True
        ppp["save_g_mesh"] = False
        # ppp["save_g_chunks"] = False
        ppp["save_D_c_mesh"] = True
        # ppp["save_D_c_chunks"] = True

    def set_user_parameters_in_lists(self):
        """
        Recast and define particular parameters as attributes in Python
        lists
        """
        gp = self.parameters["two_dimensional_geometry"]
        mp = self.parameters["material"]
        
        # Meshpoints
        self.meshpoint_num = gp["meshpoint_num"]
        meshpoints_list = []
        meshpoints_label_list = []
        meshpoints_color_list = []
        meshpoints_name_list = []

        for meshpoint_indx in range(self.meshpoint_num):
            meshpoint_x_coord = gp["meshpoint_indx_"+str(meshpoint_indx)+"_x_coord"]
            meshpoint_y_coord = gp["meshpoint_indx_"+str(meshpoint_indx)+"_y_coord"]
            meshpoints_label = gp["meshpoint_indx_"+str(meshpoint_indx)+"_label"]
            meshpoints_color = gp["meshpoint_indx_"+str(meshpoint_indx)+"_color"]
            meshpoints_name = gp["meshpoint_indx_"+str(meshpoint_indx)+"_name"]

            meshpoint = (meshpoint_x_coord, meshpoint_y_coord)

            meshpoints_list.append(meshpoint)
            meshpoints_label_list.append(meshpoints_label)
            meshpoints_color_list.append(meshpoints_color)
            meshpoints_name_list.append(meshpoints_name)
        
        self.meshpoints_list = meshpoints_list
        self.meshpoints_label_list = meshpoints_label_list
        self.meshpoints_color_list = meshpoints_color_list
        self.meshpoints_name_list = meshpoints_name_list

        # Chain segment numbers
        self.nu_num = mp["nu_num"]
        nu_list = []

        for nu_indx in range(self.nu_num):
            nu_val = mp["nu_indx_"+str(nu_indx)+"_nu_val"]
            nu_list.append(nu_val)
        
        self.nu_list = nu_list

        # Chain segment number chunks
        self.nu_chunks_num = mp["nu_chunks_num"]
        nu_chunks_list = []
        nu_chunks_indx_in_nu_list = []
        nu_chunks_label_list = []
        nu_chunks_color_list = []

        for nu_chunk_indx in range(self.nu_chunks_num):
            nu_chunks_val = mp["nu_chunk_indx_"+str(nu_chunk_indx)+"_nu_val"]
            nu_chunks_indx_val_in_nu_list = mp["nu_chunk_indx_"+str(nu_chunk_indx)+"_indx_val_in_nu_list"]
            nu_chunks_label = mp["nu_chunk_indx_"+str(nu_chunk_indx)+"_label"]
            nu_chunks_color = mp["nu_chunk_indx_"+str(nu_chunk_indx)+"_color"]

            nu_chunks_list.append(nu_chunks_val)
            nu_chunks_indx_in_nu_list.append(nu_chunks_indx_val_in_nu_list)
            nu_chunks_label_list.append(nu_chunks_label)
            nu_chunks_color_list.append(nu_chunks_color)
        
        self.nu_chunks_list = nu_chunks_list
        self.nu_chunks_indx_in_nu_list = nu_chunks_indx_in_nu_list
        self.nu_chunks_label_list = nu_chunks_label_list
        self.nu_chunks_color_list = nu_chunks_color_list
    
    def set_user_parameters_in_dicts(self):
        """
        Recast and define particular parameters as attributes in Python
        dictionaries
        """
        femp = self.parameters["fem"]

        # Metadata quadrature degree
        self.metadata  = {"quadrature_degree": femp["quadrature_degree"]}

        self.two_dim_vector_indx_dict = {
            "1": 0,
            "2": 1
        }

        # Tensor-to-Voigt notation dictionary
        self.two_dim_tensor2voigt_vector_indx_dict = {
            "11": 0,
            "12": 1,
            "21": 2,
            "22": 3
        }

        # Dictionaries for solver parameters
        # solver_u
        self.solver_u_parameters_dict = {"nonlinear_solver": "snes",
                                         "symmetric": True,
                                         "snes_solver": {"linear_solver": "mumps",
                                                         "method": "newtontr",
                                                         "line_search": "cp",
                                                         "preconditioner": "hypre_amg",
                                                         "maximum_iterations": 200,
                                                         "absolute_tolerance": 1e-8,
                                                         "relative_tolerance": 1e-7,
                                                         "solution_tolerance": 1e-7,
                                                         "report": True,
                                                         "error_on_nonconvergence": False
                                                         }
                                                         }

        # solver_bounded_lmbda_c_tilde
        self.solver_bounded_lmbda_c_tilde_parameters_dict = {"nonlinear_solver": "snes",
                                                             "symmetric": True,
                                                             "snes_solver": {"linear_solver": "umfpack",
                                                                             "method": "vinewtonssls",
                                                                             "line_search": "basic",
                                                                             "maximum_iterations": 200,# 50
                                                                             "absolute_tolerance": 1e-8,
                                                                             "relative_tolerance": 1e-7,# 1e-5
                                                                             "solution_tolerance": 1e-7,# 1e-5
                                                                             "report": True,
                                                                             "error_on_nonconvergence": False
                                                                             }
                                                                             }
        
        # solver_unbounded_lmbda_c_tilde
        self.solver_unbounded_lmbda_c_tilde_parameters_dict = {"nonlinear_solver": "snes",
                                                               "symmetric": True,
                                                               "snes_solver": {"linear_solver": "mumps",
                                                                               "method": "newtontr",
                                                                               "line_search": "cp",
                                                                               "preconditioner": "hypre_amg",
                                                                               "maximum_iterations": 200,
                                                                               "absolute_tolerance": 1e-8,
                                                                               "relative_tolerance": 1e-7,
                                                                               "solution_tolerance": 1e-7,
                                                                               "report": True,
                                                                               "error_on_nonconvergence": False
                                                                               }
                                                                               }
        
        # solver_bounded_monolithic
        self.solver_bounded_monolithic_parameters_dict = {"nonlinear_solver": "snes",
                                                          "symmetric": True,
                                                          "snes_solver": {"linear_solver": "umfpack",
                                                                          "method": "vinewtonssls",
                                                                          "line_search": "basic",
                                                                          "maximum_iterations": 200,#50
                                                                          "absolute_tolerance": 1e-8,
                                                                          "relative_tolerance": 1e-7,# 1e-5
                                                                          "solution_tolerance": 1e-7,# 1e-5
                                                                          "report": True,
                                                                          "error_on_nonconvergence": False
                                                                          }
                                                                          }

        # solver_unbounded_monolithic
        self.solver_unbounded_monolithic_parameters_dict = {"nonlinear_solver": "snes",
                                                            "symmetric": True,
                                                            "snes_solver": {"linear_solver": "mumps",#"mumps",
                                                                            "method": "newtontr",
                                                                            "line_search": "cp",
                                                                            "preconditioner": "hypre_amg",
                                                                            "maximum_iterations": 200,
                                                                            "absolute_tolerance": 1e-8,
                                                                            "relative_tolerance": 1e-4, # 1e-4 seems to work
                                                                            "solution_tolerance": 1e-4, # 1e-4 seems to work
                                                                            "report": True,
                                                                            "error_on_nonconvergence": False
                                                                            }
                                                                            }

    def prefix(self):
        gp = self.parameters["two_dimensional_geometry"]
        return self.modelname + "_" + gp["meshtype"] + "_" + "problem"
    
    def define_mesh(self):
        """
        Define the mesh for the problem
        """
        geofile = \
            """
            Mesh.Algorithm = 8;
            coarse_mesh_elem_size = DefineNumber[ %g, Name "Parameters/coarse_mesh_elem_size" ];
            x_notch_point = DefineNumber[ %g, Name "Parameters/x_notch_point" ];
            r_notch = DefineNumber[ %g, Name "Parameters/r_notch" ];
            L = DefineNumber[ %g, Name "Parameters/L"];
            H = DefineNumber[ %g, Name "Parameters/H"];
            Point(1) = {0, 0, 0, coarse_mesh_elem_size};
            Point(2) = {x_notch_point-r_notch, 0, 0, coarse_mesh_elem_size};
            Point(3) = {0, -r_notch, 0, coarse_mesh_elem_size};
            Point(4) = {0, -H/2, 0, coarse_mesh_elem_size};
            Point(5) = {L, -H/2, 0, coarse_mesh_elem_size};
            Point(6) = {L, H/2, 0, coarse_mesh_elem_size};
            Point(7) = {0, H/2, 0, coarse_mesh_elem_size};
            Point(8) = {0, r_notch, 0, coarse_mesh_elem_size};
            Point(9) = {x_notch_point-r_notch, r_notch, 0, coarse_mesh_elem_size};
            Point(10) = {x_notch_point, 0, 0, coarse_mesh_elem_size};
            Point(11) = {x_notch_point-r_notch, -r_notch, 0, coarse_mesh_elem_size};
            Line(1) = {11, 3};
            Line(2) = {3, 4};
            Line(3) = {4, 5};
            Line(4) = {5, 6};
            Line(5) = {6, 7};
            Line(6) = {7, 8};
            Line(7) = {8, 9};
            Circle(8) = {9, 2, 10};
            Circle(9) = {10, 2, 11};
            Curve Loop(21) = {1, 2, 3, 4, 5, 6, 7, 8, 9};
            Plane Surface(31) = {21};
            Mesh.MshFileVersion = 2.0;
            """ % (self.coarse_mesh_elem_size, self.x_notch_point, self.r_notch, self.L, self.H)
        
        geofile = textwrap.dedent(geofile)

        L_string           = "{:.1f}".format(self.L)
        H_string           = "{:.1f}".format(self.H)
        x_notch_point_string = "{:.1f}".format(self.x_notch_point)
        r_notch_string     = "{:.1f}".format(self.r_notch)
        coarse_mesh_elem_size_string  = "{:.1f}".format(self.coarse_mesh_elem_size)

        mp = self.parameters["material"]
        gp = self.parameters["two_dimensional_geometry"]
        meshname = (
            mp["physical_dimensionality"]
            + "_" + mp["two_dimensional_formulation"]
            + "_" + gp["meshtype"]
            + "_" + L_string
            + "_" + H_string
            + "_" + x_notch_point_string
            + "_" + r_notch_string
            + "_" + coarse_mesh_elem_size_string
        )
        
        return gmsh_mesher(geofile, self.prefix(), meshname)
    
    def define_bc_u(self):
        """
        Return a list of boundary conditions on the displacement
        """
        self.lines = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        self.lines.set_all(0)

        L = self.L
        H = self.H
        x_notch_point = self.x_notch_point
        r_notch = self.r_notch

        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0., DOLFIN_EPS)
        
        class RightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], L, DOLFIN_EPS)

        class BottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], -H/2., DOLFIN_EPS)
        
        class TopBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], H/2., DOLFIN_EPS)

        class Notch(SubDomain):
            def inside(self, x, on_boundary):
                r_notch_sq = (x[0]-(x_notch_point-r_notch))**2 + x[1]**2
                return r_notch_sq <= (r_notch + DOLFIN_EPS)**2

        LeftBoundary().mark(self.lines, 1)
        RightBoundary().mark(self.lines, 2)
        BottomBoundary().mark(self.lines, 3)
        TopBoundary().mark(self.lines, 4)
        Notch().mark(self.lines, 5)

        mesh_topologier(self.lines, self.prefix(), "lines")

        self.u_y_expression = Expression("u_y", u_y=0., degree=0)

        bc_I = DirichletBC(self.V_u.sub(1), Constant(0.), BottomBoundary())
        bc_II = DirichletBC(self.V_u.sub(0), Constant(0.), RightBoundary())
        bc_III  = DirichletBC(self.V_u.sub(1), self.u_y_expression, TopBoundary())

        return [bc_I, bc_II, bc_III]
    
    def define_bc_monolithic(self):
        """
        Return a list of boundary conditions for the monolithic solution
        scheme
        """
        self.lines = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        self.lines.set_all(0)

        L = self.L
        H = self.H
        x_notch_point = self.x_notch_point
        r_notch = self.r_notch

        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0., DOLFIN_EPS)
        
        class RightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], L, DOLFIN_EPS)

        class BottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], -H/2., DOLFIN_EPS)
        
        class TopBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], H/2., DOLFIN_EPS)

        class Notch(SubDomain):
            def inside(self, x, on_boundary):
                r_notch_sq = (x[0]-(x_notch_point-r_notch))**2 + x[1]**2
                return r_notch_sq <= (r_notch + DOLFIN_EPS)**2

        LeftBoundary().mark(self.lines, 1)
        RightBoundary().mark(self.lines, 2)
        BottomBoundary().mark(self.lines, 3)
        TopBoundary().mark(self.lines, 4)
        Notch().mark(self.lines, 5)

        mesh_topologier(self.lines, self.prefix(), "lines")

        self.u_y_expression = Expression("u_y", u_y=0., degree=0)

        bc_I = DirichletBC(self.V.sub(0).sub(1), Constant(0.), BottomBoundary())
        bc_II = DirichletBC(self.V.sub(0).sub(0), Constant(0.), RightBoundary())
        bc_III  = DirichletBC(self.V.sub(0).sub(1), self.u_y_expression, TopBoundary())

        return [bc_I, bc_II, bc_III]
    
    def F_func(self, t):
        """
        Function defining the deformation
        """
        dp = self.parameters["deformation"]

        return 1 + dp["strain_rate"]*(t-dp["t_min"])
    
    # def F_func(self, t):
    #     """
    #     Function defining the deformation
    #     """
    #     dp = self.parameters["deformation"]

    #     t_rel = t - dp["t_min"]
    #     t_def = dp["t_max"] - dp["t_min"]
    #     strain_max = dp["strain_rate"] * t_def
    #     strain_max_prop = 0.5
    #     strain_max_actual = strain_max_prop * strain_max
    #     F_val = (1 + dp["strain_rate"] * t_rel * np.heaviside(strain_max_actual-dp["strain_rate"]*t_rel, 0.5)
    #              + (2*strain_max_actual-dp["strain_rate"]*t_rel) * np.heaviside(dp["strain_rate"]*t_rel-strain_max_actual, 0.5)
    #     )
    #     return F_val
    
    def initialize_lmbda(self):
        lmbda_y        = [] # unitless
        lmbda_y_chunks = [] # unitless

        return lmbda_y, lmbda_y_chunks
    
    def store_initialized_lmbda(self, lmbda):
        lmbda_y_val = 1 # assuming no pre-stretching
        
        lmbda_y        = lmbda[0]
        lmbda_y_chunks = lmbda[1]
        
        lmbda_y.append(lmbda_y_val)
        lmbda_y_chunks.append(lmbda_y_val)
        
        return lmbda_y, lmbda_y_chunks
    
    def calculate_lmbda_func(self, t_val):
        lmbda_y_val = self.F_func(t_val)

        return lmbda_y_val
    
    def store_calculated_lmbda(self, lmbda, lmbda_val):
        lmbda_y        = lmbda[0]
        lmbda_y_chunks = lmbda[1]
        lmbda_y_val    = lmbda_val
        
        lmbda_y.append(lmbda_y_val)
        
        return lmbda_y, lmbda_y_chunks
    
    def store_calculated_lmbda_chunk_post_processing(self, lmbda, lmbda_val):
        lmbda_y        = lmbda[0]
        lmbda_y_chunks = lmbda[1]
        lmbda_y_val    = lmbda_val
        
        lmbda_y_chunks.append(lmbda_y_val)
        
        return lmbda_y, lmbda_y_chunks
    
    def calculate_u_func(self, lmbda):
        lmbda_y        = lmbda[0]
        lmbda_y_chunks = lmbda[1]

        u_y        = [lmbda_y_val-1 for lmbda_y_val in lmbda_y]
        u_y_chunks = [lmbda_y_chunks_val-1 for lmbda_y_chunks_val in lmbda_y_chunks]

        return u_y, u_y_chunks
    
    def save_deformation_attributes(self, lmbda, u):
        lmbda_y        = lmbda[0]
        lmbda_y_chunks = lmbda[1]

        u_y        = u[0]
        u_y_chunks = u[1]

        self.lmbda_y        = lmbda_y
        self.lmbda_y_chunks = lmbda_y_chunks
        self.u_y            = u_y
        self.u_y_chunks     = u_y_chunks

    def set_u_chunks(self):
        self.u_1_chunks = []
        self.u_1_chunks_val = [0. for meshpoint_indx in range(self.meshpoint_num)]
        self.u_2_chunks = []
        self.u_2_chunks_val = [0. for meshpoint_indx in range(self.meshpoint_num)]
    
    def set_F_chunks(self):
        self.F_22_chunks = []
        self.F_22_chunks_val = [0. for meshpoint_indx in range(self.meshpoint_num)]
    
    def set_sigma_chunks(self):
        self.sigma_22_chunks     = []
        self.sigma_22_chunks_val = [0. for meshpoint_indx in range(self.meshpoint_num)]
        self.sigma_22_penalty_term_chunks     = []
        self.sigma_22_penalty_term_chunks_val = [0. for meshpoint_indx in range(self.meshpoint_num)]
        self.sigma_22_less_penalty_term_chunks     = []
        self.sigma_22_less_penalty_term_chunks_val = [0. for meshpoint_indx in range(self.meshpoint_num)]
    
    def set_loading(self):
        """
        Update Dirichlet boundary conditions
        """
        self.u_y_expression.u_y = self.H*self.u_y[self.t_indx]
    
    def u_chunks_post_processing(self):
        for meshpoint_indx in range(self.meshpoint_num):
            MPI.barrier(MPI.comm_world)
            u_chunks_val = peval(self.u, self.meshpoints_list[meshpoint_indx])
            self.u_1_chunks_val[meshpoint_indx] = u_chunks_val[self.two_dim_vector_indx_dict["1"]]
            self.u_2_chunks_val[meshpoint_indx] = u_chunks_val[self.two_dim_vector_indx_dict["2"]]
        self.u_1_chunks.append(deepcopy(self.u_1_chunks_val))
        self.u_2_chunks.append(deepcopy(self.u_2_chunks_val))

    def F_chunks_post_processing(self):
        F_val = project(self.F, self.V_DG_tensor)
        for meshpoint_indx in range(self.meshpoint_num):
            MPI.barrier(MPI.comm_world)
            F_chunks_val = peval(F_val, self.meshpoints_list[meshpoint_indx])
            self.F_22_chunks_val[meshpoint_indx] = F_chunks_val[self.two_dim_tensor2voigt_vector_indx_dict["22"]]
        self.F_22_chunks.append(deepcopy(self.F_22_chunks_val))
    
    def sigma_chunks_post_processing(self):
        sigma_val = self.cauchy_stress_ufl_fenics_mesh_func()
        sigma_penalty_term_val = self.cauchy_stress_penalty_term_ufl_fenics_mesh_func()
        sigma_less_penalty_term_val = sigma_val - sigma_penalty_term_val
        sigma_val = project(sigma_val, self.V_DG_tensor)
        sigma_penalty_term_val = project(sigma_penalty_term_val, self.V_DG_tensor)
        sigma_less_penalty_term_val = project(sigma_less_penalty_term_val, self.V_DG_tensor)
        for meshpoint_indx in range(self.meshpoint_num):
            MPI.barrier(MPI.comm_world)
            sigma_chunks_val = peval(sigma_val, self.meshpoints_list[meshpoint_indx])
            sigma_penalty_term_chunks_val = peval(sigma_penalty_term_val, self.meshpoints_list[meshpoint_indx])
            sigma_less_penalty_term_chunks_val = peval(sigma_less_penalty_term_val, self.meshpoints_list[meshpoint_indx])
            self.sigma_22_chunks_val[meshpoint_indx] = sigma_chunks_val[self.two_dim_tensor2voigt_vector_indx_dict["22"]]
            self.sigma_22_penalty_term_chunks_val[meshpoint_indx] = sigma_penalty_term_chunks_val[self.two_dim_tensor2voigt_vector_indx_dict["22"]]
            self.sigma_22_less_penalty_term_chunks_val[meshpoint_indx] = sigma_less_penalty_term_chunks_val[self.two_dim_tensor2voigt_vector_indx_dict["22"]]
        self.sigma_22_chunks.append(deepcopy(self.sigma_22_chunks_val))
        self.sigma_22_penalty_term_chunks.append(deepcopy(self.sigma_22_penalty_term_chunks_val))
        self.sigma_22_less_penalty_term_chunks.append(deepcopy(self.sigma_22_less_penalty_term_chunks_val))

    def finalization(self):
        """
        Plot the chunked results from the evolution problem
        """

        ppp = self.parameters["post_processing"]

        # plot results
        latex_formatting_figure(ppp)

        MPI.barrier(MPI.comm_world)

        # lmbda_c
        if ppp["save_lmbda_c_chunks"] and self.comm_rank == 0:
            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                lmbda_c___meshpoint_chunk = [lmbda_c_chunk[meshpoint_indx] for lmbda_c_chunk in self.lmbda_c_chunks]
                plt.plot(self.t_chunks, lmbda_c___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_c$', 30, "t-vs-lmbda_c")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                lmbda_c___meshpoint_chunk = [lmbda_c_chunk[meshpoint_indx] for lmbda_c_chunk in self.lmbda_c_chunks]
                plt.plot(self.u_y_chunks, lmbda_c___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$\lambda_c$', 30, "u_y-vs-lmbda_c")

        # lmbda_c_eq
        if ppp["save_lmbda_c_eq_chunks"] and self.comm_rank == 0:
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                lmbda_c_eq___meshpoint_chunk = [lmbda_c_eq_chunk[meshpoint_indx] for lmbda_c_eq_chunk in self.lmbda_c_eq_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    lmbda_c_eq___nu_chunk = [lmbda_c_eq_chunk[nu_chunk_indx] for lmbda_c_eq_chunk in lmbda_c_eq___meshpoint_chunk]
                    plt.plot(self.t_chunks, lmbda_c_eq___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_c^{eq}$', 30, "t-vs-lmbda_c_eq"+"_"+self.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                lmbda_c_eq___meshpoint_chunk = [lmbda_c_eq_chunk[meshpoint_indx] for lmbda_c_eq_chunk in self.lmbda_c_eq_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    lmbda_c_eq___nu_chunk = [lmbda_c_eq_chunk[nu_chunk_indx] for lmbda_c_eq_chunk in lmbda_c_eq___meshpoint_chunk]
                    plt.plot(self.u_y_chunks, lmbda_c_eq___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_y$', 30, r'$\lambda_c^{eq}$', 30, "u_y-vs-lmbda_c_eq"+"_"+self.meshpoints_name_list[meshpoint_indx])
        
        # lmbda_nu
        if ppp["save_lmbda_nu_chunks"] and self.comm_rank == 0:
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                lmbda_nu___meshpoint_chunk = [lmbda_nu_chunk[meshpoint_indx] for lmbda_nu_chunk in self.lmbda_nu_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    lmbda_nu___nu_chunk = [lmbda_nu_chunk[nu_chunk_indx] for lmbda_nu_chunk in lmbda_nu___meshpoint_chunk]
                    plt.plot(self.t_chunks, lmbda_nu___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_{\nu}$', 30, "t-vs-lmbda_nu"+"_"+self.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                lmbda_nu___meshpoint_chunk = [lmbda_nu_chunk[meshpoint_indx] for lmbda_nu_chunk in self.lmbda_nu_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    lmbda_nu___nu_chunk = [lmbda_nu_chunk[nu_chunk_indx] for lmbda_nu_chunk in lmbda_nu___meshpoint_chunk]
                    plt.plot(self.u_y_chunks, lmbda_nu___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_y$', 30, r'$\lambda_{\nu}$', 30, "u_y-vs-lmbda_nu"+"_"+self.meshpoints_name_list[meshpoint_indx])
        
        # lmbda_c_tilde
        if ppp["save_lmbda_c_tilde_chunks"] and self.comm_rank == 0:
            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                lmbda_c_tilde___meshpoint_chunk = [lmbda_c_tilde_chunk[meshpoint_indx] for lmbda_c_tilde_chunk in self.lmbda_c_tilde_chunks]
                plt.plot(self.t_chunks, lmbda_c_tilde___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\tilde{\lambda}_c$', 30, "t-vs-lmbda_c_tilde")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                lmbda_c_tilde___meshpoint_chunk = [lmbda_c_tilde_chunk[meshpoint_indx] for lmbda_c_tilde_chunk in self.lmbda_c_tilde_chunks]
                plt.plot(self.u_y_chunks, lmbda_c_tilde___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$\tilde{\lambda}_c$', 30, "u_y-vs-lmbda_c_tilde")

        # lmbda_c_eq_tilde
        if ppp["save_lmbda_c_eq_tilde_chunks"] and self.comm_rank == 0:
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                lmbda_c_eq_tilde___meshpoint_chunk = [lmbda_c_eq_tilde_chunk[meshpoint_indx] for lmbda_c_eq_tilde_chunk in self.lmbda_c_eq_tilde_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    lmbda_c_eq_tilde___nu_chunk = [lmbda_c_eq_tilde_chunk[nu_chunk_indx] for lmbda_c_eq_tilde_chunk in lmbda_c_eq_tilde___meshpoint_chunk]
                    plt.plot(self.t_chunks, lmbda_c_eq_tilde___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\tilde{\lambda}_c^{eq}$', 30, "t-vs-lmbda_c_eq_tilde"+"_"+self.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                lmbda_c_eq_tilde___meshpoint_chunk = [lmbda_c_eq_tilde_chunk[meshpoint_indx] for lmbda_c_eq_tilde_chunk in self.lmbda_c_eq_tilde_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    lmbda_c_eq_tilde___nu_chunk = [lmbda_c_eq_tilde_chunk[nu_chunk_indx] for lmbda_c_eq_tilde_chunk in lmbda_c_eq_tilde___meshpoint_chunk]
                    plt.plot(self.u_y_chunks, lmbda_c_eq_tilde___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_y$', 30, r'$\tilde{\lambda}_c^{eq}$', 30, "u_y-vs-lmbda_c_eq_tilde"+"_"+self.meshpoints_name_list[meshpoint_indx])
        
        # lmbda_nu_tilde
        if ppp["save_lmbda_nu_tilde_chunks"] and self.comm_rank == 0:
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                lmbda_nu_tilde___meshpoint_chunk = [lmbda_nu_tilde_chunk[meshpoint_indx] for lmbda_nu_tilde_chunk in self.lmbda_nu_tilde_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    lmbda_nu_tilde___nu_chunk = [lmbda_nu_tilde_chunk[nu_chunk_indx] for lmbda_nu_tilde_chunk in lmbda_nu_tilde___meshpoint_chunk]
                    plt.plot(self.t_chunks, lmbda_nu_tilde___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\tilde{\lambda}_{\nu}$', 30, "t-vs-lmbda_nu_tilde"+"_"+self.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                lmbda_nu_tilde___meshpoint_chunk = [lmbda_nu_tilde_chunk[meshpoint_indx] for lmbda_nu_tilde_chunk in self.lmbda_nu_tilde_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    lmbda_nu_tilde___nu_chunk = [lmbda_nu_tilde_chunk[nu_chunk_indx] for lmbda_nu_tilde_chunk in lmbda_nu_tilde___meshpoint_chunk]
                    plt.plot(self.u_y_chunks, lmbda_nu_tilde___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_y$', 30, r'$\tilde{\lambda}_{\nu}$', 30, "u_y-vs-lmbda_nu_tilde"+"_"+self.meshpoints_name_list[meshpoint_indx])
        
        # lmbda_c_tilde_max
        if ppp["save_lmbda_c_tilde_max_chunks"] and self.comm_rank == 0:
            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                lmbda_c_tilde_max___meshpoint_chunk = [lmbda_c_tilde_max_chunk[meshpoint_indx] for lmbda_c_tilde_max_chunk in self.lmbda_c_tilde_max_chunks]
                plt.plot(self.t_chunks, lmbda_c_tilde_max___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\tilde{\lambda}_c^{max}$', 30, "t-vs-lmbda_c_tilde_max")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                lmbda_c_tilde_max___meshpoint_chunk = [lmbda_c_tilde_max_chunk[meshpoint_indx] for lmbda_c_tilde_max_chunk in self.lmbda_c_tilde_max_chunks]
                plt.plot(self.u_y_chunks, lmbda_c_tilde_max___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$\tilde{\lambda}_c^{max}$', 30, "u_y-vs-lmbda_c_tilde_max")

        # lmbda_c_eq_tilde_max
        if ppp["save_lmbda_c_eq_tilde_max_chunks"] and self.comm_rank == 0:
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                lmbda_c_eq_tilde_max___meshpoint_chunk = [lmbda_c_eq_tilde_max_chunk[meshpoint_indx] for lmbda_c_eq_tilde_max_chunk in self.lmbda_c_eq_tilde_max_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    lmbda_c_eq_tilde_max___nu_chunk = [lmbda_c_eq_tilde_max_chunk[nu_chunk_indx] for lmbda_c_eq_tilde_max_chunk in lmbda_c_eq_tilde_max___meshpoint_chunk]
                    plt.plot(self.t_chunks, lmbda_c_eq_tilde_max___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$(\tilde{\lambda}_c^{eq})^{max}$', 30, "t-vs-lmbda_c_eq_tilde_max"+"_"+self.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                lmbda_c_eq_tilde_max___meshpoint_chunk = [lmbda_c_eq_tilde_max_chunk[meshpoint_indx] for lmbda_c_eq_tilde_max_chunk in self.lmbda_c_eq_tilde_max_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    lmbda_c_eq_tilde_max___nu_chunk = [lmbda_c_eq_tilde_max_chunk[nu_chunk_indx] for lmbda_c_eq_tilde_max_chunk in lmbda_c_eq_tilde_max___meshpoint_chunk]
                    plt.plot(self.u_y_chunks, lmbda_c_eq_tilde_max___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_y$', 30, r'$(\tilde{\lambda}_c^{eq})^{max}$', 30, "u_y-vs-lmbda_c_eq_tilde_max"+"_"+self.meshpoints_name_list[meshpoint_indx])
        
        # lmbda_nu_tilde_max
        if ppp["save_lmbda_nu_tilde_max_chunks"] and self.comm_rank == 0:
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                lmbda_nu_tilde_max___meshpoint_chunk = [lmbda_nu_tilde_max_chunk[meshpoint_indx] for lmbda_nu_tilde_max_chunk in self.lmbda_nu_tilde_max_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    lmbda_nu_tilde_max___nu_chunk = [lmbda_nu_tilde_max_chunk[nu_chunk_indx] for lmbda_nu_tilde_max_chunk in lmbda_nu_tilde_max___meshpoint_chunk]
                    plt.plot(self.t_chunks, lmbda_nu_tilde_max___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\tilde{\lambda}_{\nu}^{max}$', 30, "t-vs-lmbda_nu_tilde_max"+"_"+self.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                lmbda_nu_tilde_max___meshpoint_chunk = [lmbda_nu_tilde_max_chunk[meshpoint_indx] for lmbda_nu_tilde_max_chunk in self.lmbda_nu_tilde_max_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    lmbda_nu_tilde_max___nu_chunk = [lmbda_nu_tilde_max_chunk[nu_chunk_indx] for lmbda_nu_tilde_max_chunk in lmbda_nu_tilde_max___meshpoint_chunk]
                    plt.plot(self.u_y_chunks, lmbda_nu_tilde_max___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_y$', 30, r'$\tilde{\lambda}_{\nu}^{max}$', 30, "u_y-vs-lmbda_nu_tilde_max"+"_"+self.meshpoints_name_list[meshpoint_indx])
        
        # g
        if ppp["save_g_chunks"] and self.comm_rank == 0:
            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                g___meshpoint_chunk = [g_chunk[meshpoint_indx] for g_chunk in self.g_chunks]
                plt.plot(self.t_chunks, g___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$g$', 30, "t-vs-g")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                g___meshpoint_chunk = [g_chunk[meshpoint_indx] for g_chunk in self.g_chunks]
                plt.plot(self.u_y_chunks, g___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$g$', 30, "u_y-vs-g")
        
        # upsilon_c
        if ppp["save_upsilon_c_chunks"] and self.comm_rank == 0:
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                upsilon_c___meshpoint_chunk = [upsilon_c_chunk[meshpoint_indx] for upsilon_c_chunk in self.upsilon_c_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    upsilon_c___nu_chunk = [upsilon_c_chunk[nu_chunk_indx] for upsilon_c_chunk in upsilon_c___meshpoint_chunk]
                    plt.plot(self.t_chunks, upsilon_c___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\upsilon_c$', 30, "t-vs-upsilon_c"+"_"+self.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                upsilon_c___meshpoint_chunk = [upsilon_c_chunk[meshpoint_indx] for upsilon_c_chunk in self.upsilon_c_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    upsilon_c___nu_chunk = [upsilon_c_chunk[nu_chunk_indx] for upsilon_c_chunk in upsilon_c___meshpoint_chunk]
                    plt.plot(self.u_y_chunks, upsilon_c___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_y$', 30, r'$\upsilon_c$', 30, "u_y-vs-upsilon_c"+"_"+self.meshpoints_name_list[meshpoint_indx])
        
        # Upsilon_c
        if ppp["save_Upsilon_c_chunks"] and self.comm_rank == 0:
            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                Upsilon_c___meshpoint_chunk = [Upsilon_c_chunk[meshpoint_indx] for Upsilon_c_chunk in self.Upsilon_c_chunks]
                plt.plot(self.t_chunks, Upsilon_c___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\Upsilon_c$', 30, "t-vs-Upsilon_c")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                Upsilon_c___meshpoint_chunk = [Upsilon_c_chunk[meshpoint_indx] for Upsilon_c_chunk in self.Upsilon_c_chunks]
                plt.plot(self.u_y_chunks, Upsilon_c___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$\Upsilon_c$', 30, "u_y-vs-Upsilon_c")

        # d_c
        if ppp["save_d_c_chunks"] and self.comm_rank == 0:
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                d_c___meshpoint_chunk = [d_c_chunk[meshpoint_indx] for d_c_chunk in self.d_c_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    d_c___nu_chunk = [d_c_chunk[nu_chunk_indx] for d_c_chunk in d_c___meshpoint_chunk]
                    plt.plot(self.t_chunks, d_c___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$d_c$', 30, "t-vs-d_c"+"_"+self.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                d_c___meshpoint_chunk = [d_c_chunk[meshpoint_indx] for d_c_chunk in self.d_c_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    d_c___nu_chunk = [d_c_chunk[nu_chunk_indx] for d_c_chunk in d_c___meshpoint_chunk]
                    plt.plot(self.u_y_chunks, d_c___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_y$', 30, r'$d_c$', 30, "u_y-vs-d_c"+"_"+self.meshpoints_name_list[meshpoint_indx])
        
        # D_c
        if ppp["save_D_c_chunks"] and self.comm_rank == 0:
            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                D_c___meshpoint_chunk = [D_c_chunk[meshpoint_indx] for D_c_chunk in self.D_c_chunks]
                plt.plot(self.t_chunks, D_c___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$D_c$', 30, "t-vs-D_c")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                D_c___meshpoint_chunk = [D_c_chunk[meshpoint_indx] for D_c_chunk in self.D_c_chunks]
                plt.plot(self.u_y_chunks, D_c___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$D_c$', 30, "u_y-vs-D_c")
        
        # epsilon_cnu_diss_hat
        if ppp["save_epsilon_cnu_diss_hat_chunks"] and self.comm_rank == 0:
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                epsilon_cnu_diss_hat___meshpoint_chunk = [epsilon_cnu_diss_hat_chunk[meshpoint_indx] for epsilon_cnu_diss_hat_chunk in self.epsilon_cnu_diss_hat_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    epsilon_cnu_diss_hat___nu_chunk = [epsilon_cnu_diss_hat_chunk[nu_chunk_indx] for epsilon_cnu_diss_hat_chunk in epsilon_cnu_diss_hat___meshpoint_chunk]
                    plt.plot(self.t_chunks, epsilon_cnu_diss_hat___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\hat{\varepsilon}_{c\nu}^{diss}$', 30, "t-vs-epsilon_cnu_diss_hat"+"_"+self.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                epsilon_cnu_diss_hat___meshpoint_chunk = [epsilon_cnu_diss_hat_chunk[meshpoint_indx] for epsilon_cnu_diss_hat_chunk in self.epsilon_cnu_diss_hat_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    epsilon_cnu_diss_hat___nu_chunk = [epsilon_cnu_diss_hat_chunk[nu_chunk_indx] for epsilon_cnu_diss_hat_chunk in epsilon_cnu_diss_hat___meshpoint_chunk]
                    plt.plot(self.u_y_chunks, epsilon_cnu_diss_hat___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_y$', 30, r'$\hat{\varepsilon}_{c\nu}^{diss}$', 30, "u_y-vs-epsilon_cnu_diss_hat"+"_"+self.meshpoints_name_list[meshpoint_indx])
        
        # Epsilon_cnu_diss_hat
        if ppp["save_Epsilon_cnu_diss_hat_chunks"] and self.comm_rank == 0:
            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                Epsilon_cnu_diss_hat___meshpoint_chunk = [Epsilon_cnu_diss_hat_chunk[meshpoint_indx] for Epsilon_cnu_diss_hat_chunk in self.Epsilon_cnu_diss_hat_chunks]
                plt.plot(self.t_chunks, Epsilon_cnu_diss_hat___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\hat{E}_{c\nu}^{diss}$', 30, "t-vs-Epsilon_cnu_diss_hat")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                Epsilon_cnu_diss_hat___meshpoint_chunk = [Epsilon_cnu_diss_hat_chunk[meshpoint_indx] for Epsilon_cnu_diss_hat_chunk in self.Epsilon_cnu_diss_hat_chunks]
                plt.plot(self.u_y_chunks, Epsilon_cnu_diss_hat___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$\hat{E}_{c\nu}^{diss}$', 30, "u_y-vs-Epsilon_cnu_diss_hat")
        
        # epsilon_c_diss_hat
        if ppp["save_epsilon_c_diss_hat_chunks"] and self.comm_rank == 0:
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                epsilon_c_diss_hat___meshpoint_chunk = [epsilon_c_diss_hat_chunk[meshpoint_indx] for epsilon_c_diss_hat_chunk in self.epsilon_c_diss_hat_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    epsilon_c_diss_hat___nu_chunk = [epsilon_c_diss_hat_chunk[nu_chunk_indx] for epsilon_c_diss_hat_chunk in epsilon_c_diss_hat___meshpoint_chunk]
                    plt.plot(self.t_chunks, epsilon_c_diss_hat___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\hat{\varepsilon}_c^{diss}$', 30, "t-vs-epsilon_c_diss_hat"+"_"+self.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                epsilon_c_diss_hat___meshpoint_chunk = [epsilon_c_diss_hat_chunk[meshpoint_indx] for epsilon_c_diss_hat_chunk in self.epsilon_c_diss_hat_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    epsilon_c_diss_hat___nu_chunk = [epsilon_c_diss_hat_chunk[nu_chunk_indx] for epsilon_c_diss_hat_chunk in epsilon_c_diss_hat___meshpoint_chunk]
                    plt.plot(self.u_y_chunks, epsilon_c_diss_hat___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_y$', 30, r'$\hat{\varepsilon}_c^{diss}$', 30, "u_y-vs-epsilon_c_diss_hat"+"_"+self.meshpoints_name_list[meshpoint_indx])
        
        # Epsilon_c_diss_hat
        if ppp["save_Epsilon_c_diss_hat_chunks"] and self.comm_rank == 0:
            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                Epsilon_c_diss_hat___meshpoint_chunk = [Epsilon_c_diss_hat_chunk[meshpoint_indx] for Epsilon_c_diss_hat_chunk in self.Epsilon_c_diss_hat_chunks]
                plt.plot(self.t_chunks, Epsilon_c_diss_hat___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\hat{E}_c^{diss}$', 30, "t-vs-Epsilon_c_diss_hat")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                Epsilon_c_diss_hat___meshpoint_chunk = [Epsilon_c_diss_hat_chunk[meshpoint_indx] for Epsilon_c_diss_hat_chunk in self.Epsilon_c_diss_hat_chunks]
                plt.plot(self.u_y_chunks, Epsilon_c_diss_hat___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$\hat{E}_c^{diss}$', 30, "u_y-vs-Epsilon_c_diss_hat")
        
        # overline_epsilon_cnu_diss_hat
        if ppp["save_overline_epsilon_cnu_diss_hat_chunks"] and self.comm_rank == 0:
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                overline_epsilon_cnu_diss_hat___meshpoint_chunk = [overline_epsilon_cnu_diss_hat_chunk[meshpoint_indx] for overline_epsilon_cnu_diss_hat_chunk in self.overline_epsilon_cnu_diss_hat_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    overline_epsilon_cnu_diss_hat___nu_chunk = [overline_epsilon_cnu_diss_hat_chunk[nu_chunk_indx] for overline_epsilon_cnu_diss_hat_chunk in overline_epsilon_cnu_diss_hat___meshpoint_chunk]
                    plt.plot(self.t_chunks, overline_epsilon_cnu_diss_hat___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', 30, "t-vs-overline_epsilon_cnu_diss_hat"+"_"+self.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                overline_epsilon_cnu_diss_hat___meshpoint_chunk = [overline_epsilon_cnu_diss_hat_chunk[meshpoint_indx] for overline_epsilon_cnu_diss_hat_chunk in self.overline_epsilon_cnu_diss_hat_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    overline_epsilon_cnu_diss_hat___nu_chunk = [overline_epsilon_cnu_diss_hat_chunk[nu_chunk_indx] for overline_epsilon_cnu_diss_hat_chunk in overline_epsilon_cnu_diss_hat___meshpoint_chunk]
                    plt.plot(self.u_y_chunks, overline_epsilon_cnu_diss_hat___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_y$', 30, r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', 30, "u_y-vs-overline_epsilon_cnu_diss_hat"+"_"+self.meshpoints_name_list[meshpoint_indx])
        
        # overline_Epsilon_cnu_diss_hat
        if ppp["save_overline_Epsilon_cnu_diss_hat_chunks"] and self.comm_rank == 0:
            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                overline_Epsilon_cnu_diss_hat___meshpoint_chunk = [overline_Epsilon_cnu_diss_hat_chunk[meshpoint_indx] for overline_Epsilon_cnu_diss_hat_chunk in self.overline_Epsilon_cnu_diss_hat_chunks]
                plt.plot(self.t_chunks, overline_Epsilon_cnu_diss_hat___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\overline{\hat{E}_{c\nu}^{diss}}$', 30, "t-vs-overline_Epsilon_cnu_diss_hat")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                overline_Epsilon_cnu_diss_hat___meshpoint_chunk = [overline_Epsilon_cnu_diss_hat_chunk[meshpoint_indx] for overline_Epsilon_cnu_diss_hat_chunk in self.overline_Epsilon_cnu_diss_hat_chunks]
                plt.plot(self.u_y_chunks, overline_Epsilon_cnu_diss_hat___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$\overline{\hat{E}_{c\nu}^{diss}}$', 30, "u_y-vs-overline_Epsilon_cnu_diss_hat")
        

        # overline_epsilon_c_diss_hat
        if ppp["save_overline_epsilon_c_diss_hat_chunks"] and self.comm_rank == 0:
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                overline_epsilon_c_diss_hat___meshpoint_chunk = [overline_epsilon_c_diss_hat_chunk[meshpoint_indx] for overline_epsilon_c_diss_hat_chunk in self.overline_epsilon_c_diss_hat_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    overline_epsilon_c_diss_hat___nu_chunk = [overline_epsilon_c_diss_hat_chunk[nu_chunk_indx] for overline_epsilon_c_diss_hat_chunk in overline_epsilon_c_diss_hat___meshpoint_chunk]
                    plt.plot(self.t_chunks, overline_epsilon_c_diss_hat___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\overline{\hat{\varepsilon}_c^{diss}}$', 30, "t-vs-overline_epsilon_c_diss_hat"+"_"+self.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(self.meshpoint_num):
                fig = plt.figure()
                overline_epsilon_c_diss_hat___meshpoint_chunk = [overline_epsilon_c_diss_hat_chunk[meshpoint_indx] for overline_epsilon_c_diss_hat_chunk in self.overline_epsilon_c_diss_hat_chunks]
                for nu_chunk_indx in range(self.nu_chunks_num):
                    overline_epsilon_c_diss_hat___nu_chunk = [overline_epsilon_c_diss_hat_chunk[nu_chunk_indx] for overline_epsilon_c_diss_hat_chunk in overline_epsilon_c_diss_hat___meshpoint_chunk]
                    plt.plot(self.u_y_chunks, overline_epsilon_c_diss_hat___nu_chunk, linestyle='-', color=self.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=self.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_y$', 30, r'$\overline{\hat{\varepsilon}_c^{diss}}$', 30, "u_y-vs-overline_epsilon_c_diss_hat"+"_"+self.meshpoints_name_list[meshpoint_indx])
        
        # overline_Epsilon_c_diss_hat
        if ppp["save_overline_Epsilon_c_diss_hat_chunks"] and self.comm_rank == 0:
            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                overline_Epsilon_c_diss_hat___meshpoint_chunk = [overline_Epsilon_c_diss_hat_chunk[meshpoint_indx] for overline_Epsilon_c_diss_hat_chunk in self.overline_Epsilon_c_diss_hat_chunks]
                plt.plot(self.t_chunks, overline_Epsilon_c_diss_hat___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\overline{\hat{E}_c^{diss}}$', 30, "t-vs-overline_Epsilon_c_diss_hat")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                overline_Epsilon_c_diss_hat___meshpoint_chunk = [overline_Epsilon_c_diss_hat_chunk[meshpoint_indx] for overline_Epsilon_c_diss_hat_chunk in self.overline_Epsilon_c_diss_hat_chunks]
                plt.plot(self.u_y_chunks, overline_Epsilon_c_diss_hat___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$\overline{\hat{E}_c^{diss}}$', 30, "u_y-vs-overline_Epsilon_c_diss_hat")
        
        # u
        if ppp["save_u_chunks"] and self.comm_rank == 0:
            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                u_1___meshpoint_chunk = [u_1_chunk[meshpoint_indx] for u_1_chunk in self.u_1_chunks]
                plt.plot(self.t_chunks, u_1___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$u_1$', 30, "t-vs-u_1")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                u_1___meshpoint_chunk = [u_1_chunk[meshpoint_indx] for u_1_chunk in self.u_1_chunks]
                plt.plot(self.u_y_chunks, u_1___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$u_1$', 30, "u_y-vs-u_1")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                u_2___meshpoint_chunk = [u_2_chunk[meshpoint_indx] for u_2_chunk in self.u_2_chunks]
                plt.plot(self.t_chunks, u_2___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$u_2$', 30, "t-vs-u_2")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                u_2___meshpoint_chunk = [u_2_chunk[meshpoint_indx] for u_2_chunk in self.u_2_chunks]
                plt.plot(self.u_y_chunks, u_2___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$u_2$', 30, "u_y-vs-u_2")
        
        # F
        if ppp["save_F_chunks"] and self.comm_rank == 0:
            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                F_22___meshpoint_chunk = [F_22_chunk[meshpoint_indx] for F_22_chunk in self.F_22_chunks]
                plt.plot(self.t_chunks, F_22___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$F_{22}$', 30, "t-vs-F_22")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                F_22___meshpoint_chunk = [F_22_chunk[meshpoint_indx] for F_22_chunk in self.F_22_chunks]
                plt.plot(self.u_y_chunks, F_22___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$F_{22}$', 30, "u_y-vs-F_22")
        
        # sigma
        if ppp["save_sigma_chunks"] and self.comm_rank == 0:
            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                sigma_22___meshpoint_chunk = [sigma_22_chunk[meshpoint_indx] for sigma_22_chunk in self.sigma_22_chunks]
                plt.plot(self.t_chunks, sigma_22___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\sigma_{22}$', 30, "t-vs-sigma_22")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                sigma_22___meshpoint_chunk = [sigma_22_chunk[meshpoint_indx] for sigma_22_chunk in self.sigma_22_chunks]
                plt.plot(self.u_y_chunks, sigma_22___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$\sigma_{22}$', 30, "u_y-vs-sigma_22")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                sigma_22_penalty_term___meshpoint_chunk = [sigma_22_penalty_term_chunk[meshpoint_indx] for sigma_22_penalty_term_chunk in self.sigma_22_penalty_term_chunks]
                plt.plot(self.t_chunks, sigma_22_penalty_term___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$(\sigma_{22})_{penalty}$', 30, "t-vs-sigma_22_penalty_term")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                sigma_22_penalty_term___meshpoint_chunk = [sigma_22_penalty_term_chunk[meshpoint_indx] for sigma_22_penalty_term_chunk in self.sigma_22_penalty_term_chunks]
                plt.plot(self.u_y_chunks, sigma_22_penalty_term___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$(\sigma_{22})_{penalty}$', 30, "u_y-vs-sigma_22_penalty_term")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                sigma_22_less_penalty_term___meshpoint_chunk = [sigma_22_less_penalty_term_chunk[meshpoint_indx] for sigma_22_less_penalty_term_chunk in self.sigma_22_less_penalty_term_chunks]
                plt.plot(self.t_chunks, sigma_22_less_penalty_term___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\sigma_{22} - (\sigma_{22})_{penalty}$', 30, "t-vs-sigma_22_less_penalty_term")

            fig = plt.figure()
            for meshpoint_indx in range(self.meshpoint_num):
                sigma_22_less_penalty_term___meshpoint_chunk = [sigma_22_less_penalty_term_chunk[meshpoint_indx] for sigma_22_less_penalty_term_chunk in self.sigma_22_less_penalty_term_chunks]
                plt.plot(self.u_y_chunks, sigma_22_less_penalty_term___meshpoint_chunk, linestyle='-', color=self.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=self.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_y$', 30, r'$\sigma_{22} - (\sigma_{22})_{penalty}$', 30, "u_y-vs-sigma_22_less_penalty_term")


if __name__ == '__main__':

    L, H = 1.0, 1.5
    x_notch_point = 0.5
    r_notch = 0.02
    notch_fine_mesh_layer_level_num = 1
    fine_mesh_elem_size = 0.01 # 0.002
    coarse_mesh_elem_size = 0.01 # 0.25 # 0.1
    l_nl = coarse_mesh_elem_size # coarse_mesh_elem_size # 10*r_notch # 1.25*r_notch # 0.02 = 2*coarse_mesh_elem_size
    problem = NotchedCrack(L, H, x_notch_point, r_notch, notch_fine_mesh_layer_level_num, fine_mesh_elem_size, coarse_mesh_elem_size, l_nl)
    problem.solve()
    problem.finalization()