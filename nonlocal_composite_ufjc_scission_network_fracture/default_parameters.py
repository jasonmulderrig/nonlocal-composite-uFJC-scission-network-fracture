# import necessary libraries
from dolfin import *
import numpy as np
from scipy import constants
from composite_ufjc_scission import RateIndependentScissionCompositeuFJC

def default_parameters():

    p = Parameters("user_parameters")
    subset_list = [
        "pre_processing",
        "problem",
        "two_dimensional_geometry",
        "three_dimensional_geometry",
        "material",
        "fem",
        "deformation",
        "post_processing"
    ]
    for subparset in subset_list:
        subparset_is = eval("default_"+subparset+"_parameters()")
        p.add(subparset_is)
    return p

def default_pre_processing_parameters():
    
    pre_processing = Parameters("pre_processing")

    pre_processing.add("form_compiler_optimize", True)
    pre_processing.add("form_compiler_cpp_optimize", True)
    pre_processing.add("form_compiler_representation", "uflacs")
    pre_processing.add("form_compiler_quadrature_degree", 4)

    return pre_processing

def default_problem_parameters():
    
    problem = Parameters("problem")

    problem.add("hsize", 1e-3)

    return problem

def default_two_dimensional_geometry_parameters():
    
    two_dimensional_geometry = Parameters("two_dimensional_geometry")

    meshpoints_list = [(-1,0), (0,0), (1,0)]
    meshpoints_num = len(meshpoints_list)
    meshpoints_label_list = []
    meshpoints_color_list = ['black', 'blue', 'red']
    meshpoints_name_list = ['left_point', 'origin', 'right_point']
    for meshpoint in meshpoints_list:
        meshpoint_x_coord_string = "{:.4f}".format(meshpoint[0])
        meshpoint_y_coord_string = "{:.4f}".format(meshpoint[1])
        meshpoint_label = r'$'+'('+meshpoint_x_coord_string+','+meshpoint_y_coord_string+')'+'$'
        meshpoints_label_list.append(meshpoint_label)
    
    for meshpoint_indx in range(meshpoints_num):
        two_dimensional_geometry.add("meshpoint_indx_"+str(meshpoint_indx)+"_x_coord", meshpoints_list[meshpoint_indx][0])
        two_dimensional_geometry.add("meshpoint_indx_"+str(meshpoint_indx)+"_y_coord", meshpoints_list[meshpoint_indx][1])
        two_dimensional_geometry.add("meshpoint_indx_"+str(meshpoint_indx)+"_label", meshpoints_label_list[meshpoint_indx])
        two_dimensional_geometry.add("meshpoint_indx_"+str(meshpoint_indx)+"_color", meshpoints_color_list[meshpoint_indx])
        two_dimensional_geometry.add("meshpoint_indx_"+str(meshpoint_indx)+"_name", meshpoints_name_list[meshpoint_indx])
    
    two_dimensional_geometry.add("meshpoint_num", meshpoints_num)

    return two_dimensional_geometry

def default_three_dimensional_geometry_parameters():
    
    three_dimensional_geometry = Parameters("three_dimensional_geometry")

    meshpoints_list = [(-1,0,0), (0,0,0), (1,0,0)]
    meshpoints_num = len(meshpoints_list)
    meshpoints_label_list = []
    meshpoints_color_list = ['black', 'blue', 'red']
    meshpoints_name_list = ['left_point', 'origin', 'right_point']
    for meshpoint in meshpoints_list:
        meshpoint_x_coord_string = "{:.4f}".format(meshpoint[0])
        meshpoint_y_coord_string = "{:.4f}".format(meshpoint[1])
        meshpoint_z_coord_string = "{:.4f}".format(meshpoint[2])
        meshpoint_label = r'$'+'('+meshpoint_x_coord_string+','+meshpoint_y_coord_string+','+meshpoint_z_coord_string+')'+'$'
        meshpoints_label_list.append(meshpoint_label)
    
    for meshpoint_indx in range(meshpoints_num):
        three_dimensional_geometry.add("meshpoint_indx_"+str(meshpoint_indx)+"_x_coord", meshpoints_list[meshpoint_indx][0])
        three_dimensional_geometry.add("meshpoint_indx_"+str(meshpoint_indx)+"_y_coord", meshpoints_list[meshpoint_indx][1])
        three_dimensional_geometry.add("meshpoint_indx_"+str(meshpoint_indx)+"_z_coord", meshpoints_list[meshpoint_indx][2])
        three_dimensional_geometry.add("meshpoint_indx_"+str(meshpoint_indx)+"_label", meshpoints_label_list[meshpoint_indx])
        three_dimensional_geometry.add("meshpoint_indx_"+str(meshpoint_indx)+"_color", meshpoints_color_list[meshpoint_indx])
        three_dimensional_geometry.add("meshpoint_indx_"+str(meshpoint_indx)+"_name", meshpoints_name_list[meshpoint_indx])
    
    three_dimensional_geometry.add("meshpoint_num", meshpoints_num)

    return three_dimensional_geometry

def default_material_parameters():
    
    material = Parameters("material")

    # Fundamental material constants
    k_B          = constants.value(u"Boltzmann constant") # J/K
    N_A          = constants.value(u"Avogadro constant") # 1/mol
    h            = constants.value(u"Planck constant") # J/Hz
    hbar         = h/(2*np.pi) # J*sec
    T            = 298 # absolute room temperature, K
    beta         = 1./(k_B*T) # 1/J
    omega_0      = 1./(beta*hbar) # J/(J*sec) = 1/sec
    zeta_nu_char = 300
    kappa_nu     = 2300
    nu_b         = "None"
    zeta_b_char  = "None"
    kappa_b      = "None"

    material.add("k_B", k_B)
    material.add("N_A", N_A)
    material.add("h", h)
    material.add("hbar", hbar)
    material.add("T", T)
    material.add("beta", beta)
    material.add("omega_0", omega_0)
    material.add("zeta_nu_char", zeta_nu_char)
    material.add("kappa_nu", kappa_nu)
    material.add("nu_b", nu_b)
    material.add("zeta_b_char", zeta_b_char)
    material.add("kappa_b", kappa_b)

    # composite uFJC scission model
    material.add("scission_model", "analytical")

    lmbda_nu_crit_min = "None"
    lmbda_nu_crit_max = "None"
    tau            = "None"
    lmbda_nu_check = "None"

    material.add("lmbda_nu_crit_min", lmbda_nu_crit_min)
    material.add("lmbda_nu_crit_max", lmbda_nu_crit_max)
    material.add("tau", tau)
    material.add("lmbda_nu_check", lmbda_nu_check)

    # Network-level damage
    d_c_lmbda_nu_crit_min = 1.0001
    d_c_lmbda_nu_crit_max = 1.005

    material.add("d_c_lmbda_nu_crit_min", d_c_lmbda_nu_crit_min)
    material.add("d_c_lmbda_nu_crit_max", d_c_lmbda_nu_crit_max)

    # Non-local interaction length scale
    material.add("l_nl", 0.01)

    # Interaction intensity
    material.add("n", 1)

    # Define the chain segment number statistics in the network
    material.add("nu_distribution", "itskov")

    nu_list  = [nu for nu in range(5, 16)] # nu = 5 -> nu = 15
    nu_min   = min(nu_list)
    nu_max   = max(nu_list)
    nu_num   = len(nu_list)
    nu_bar   = 8
    Delta_nu = nu_bar-nu_min

    for nu_indx in range(nu_num):
        material.add("nu_indx_"+str(nu_indx)+"_nu_val", nu_list[nu_indx])
    material.add("nu_min", nu_min)
    material.add("nu_max", nu_max)
    material.add("nu_num", nu_num)
    material.add("nu_bar", nu_bar)
    material.add("Delta_nu", Delta_nu)

    # Define chain segment numbers to chunk during deformation
    nu_chunks_list = nu_list[::2] # nu = 5, nu = 7, ..., nu = 15
    nu_chunks_num = len(nu_chunks_list)
    nu_chunks_indx_in_nu_list = nu_chunks_list.copy()
    for nu_chunk_indx in range(nu_chunks_num):
        nu_chunks_indx_in_nu_list[nu_chunk_indx] = nu_list.index(nu_chunks_list[nu_chunk_indx])
    nu_chunks_label_list = [r'$\nu='+str(nu_list[nu_chunks_indx_in_nu_list[nu_chunk_indx]])+'$' for nu_chunk_indx in range(nu_chunks_num)]
    nu_chunks_color_list = ['orange', 'blue', 'green', 'red', 'purple', 'brown']

    for nu_chunk_indx in range(nu_chunks_num):
        material.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_nu_val", nu_chunks_list[nu_chunk_indx])
        material.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_indx_val_in_nu_list", nu_chunks_indx_in_nu_list[nu_chunk_indx])
        material.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_label", nu_chunks_label_list[nu_chunk_indx])
        material.add("nu_chunk_indx_"+str(nu_chunk_indx)+"_color", nu_chunks_color_list[nu_chunk_indx])

    material.add("nu_chunks_num", nu_chunks_num)

    composite_ufjc_nu_max = RateIndependentScissionCompositeuFJC(nu=nu_list[-1],
                                                          zeta_nu_char=zeta_nu_char,
                                                          kappa_nu=kappa_nu)
    
    lmbda_c_crit_max = composite_ufjc_nu_max.lmbda_c_eq_crit / composite_ufjc_nu_max.A_nu
    material.add("lmbda_c_tilde_lb", 0.0)
    material.add("lmbda_c_tilde_ub", lmbda_c_crit_max)
    
    # Define various characteristics of the deformation for the network
    material.add("network_model", "statistical_mechanics_model")
    material.add("phenomenological_model", "neo_hookean")
    material.add("physical_dimension", 2)
    material.add("physical_dimensionality", "two_dimensional")
    material.add("incompressibility_assumption", "nearly_incompressible")
    material.add("macro2micro_deformation_assumption", "nonaffine")
    material.add("micro2macro_homogenization_scheme", "eight_chain_model")
    material.add("chain_level_load_sharing", "equal_strain")
    material.add("rate_dependence", "rate_independent")
    material.add("two_dimensional_formulation", "plane_strain")
    material.add("microcircle_quadrature_order", 1)
    material.add("microsphere_quadrature_order", 1)

    return material

def default_fem_parameters():
    
    fem = Parameters("fem")

    fem.add("solver_algorithm", "monolithic") # "alternate_minimization"
    fem.add("solver_bounded", False) # True
    fem.add("u_degree", 1)
    fem.add("scalar_prmtr_degree", 1)
    fem.add("quadrature_degree", 4)
    
    return fem

def default_deformation_parameters():

    deformation = Parameters("deformation")

    deformation.add("deformation_type", "uniaxial")

    # general network deformation parameters
    deformation.add("K__G", 100)
    deformation.add("lmbda_damping_init", 1e3)
    deformation.add("min_lmbda_damping_val", 1e-8)
    deformation.add("iter_max_Gamma_val_NR", 1000)
    deformation.add("tol_Gamma_val_NR", 1e-2)
    deformation.add("iter_max_lmbda_c_val_NR", 1000)
    deformation.add("tol_lmbda_c_val_NR", 1e-4)
    deformation.add("iter_max_stag_NR", 2000)
    deformation.add("tol_lmbda_c_val_stag_NR", 1e-4)
    deformation.add("tol_Gamma_val_stag_NR", 1e-2)
    deformation.add("epsilon", 1e-12)
    deformation.add("max_J_val_cond", 1e12)
    deformation.add("itrtn_max_lmbda_c_tilde_val", 1000)
    deformation.add("tol_lmbda_c_tilde_val", 1e-4)
    deformation.add("k_cond_val", 1e-4)
    deformation.add("k_g_cond_val", 1e-4)

    # timing parameters
    deformation.add("strain_rate", 0.1) # 1/sec
    deformation.add("t_min", 0.) # sec
    deformation.add("t_max", 15.) # sec
    deformation.add("t_step", 0.1) # sec
    deformation.add("t_step_chunk_num", 1)

    return deformation

def default_post_processing_parameters():
    
    post_processing = Parameters("post_processing")

    ext = "xdmf"
    post_processing.add("ext", ext)
    post_processing.add("file_results", "results"+"."+ext)

    post_processing.add("rewrite_function_mesh", False)
    post_processing.add("flush_output", True)
    post_processing.add("functions_share_mesh", True)

    post_processing.add("save_lmbda_c_mesh", True)
    post_processing.add("save_lmbda_c_chunks", True)
    
    post_processing.add("save_lmbda_c_eq_mesh", False)
    post_processing.add("save_lmbda_c_eq_chunks", False)

    post_processing.add("save_lmbda_nu_mesh", False)
    post_processing.add("save_lmbda_nu_chunks", False)

    post_processing.add("save_lmbda_c_tilde_mesh", True)
    post_processing.add("save_lmbda_c_tilde_chunks", True)

    post_processing.add("save_lmbda_c_eq_tilde_mesh", False)
    post_processing.add("save_lmbda_c_eq_tilde_chunks", False)

    post_processing.add("save_lmbda_nu_tilde_mesh", False)
    post_processing.add("save_lmbda_nu_tilde_chunks", False)

    post_processing.add("save_lmbda_c_tilde_max_mesh", True)
    post_processing.add("save_lmbda_c_tilde_max_chunks", True)

    post_processing.add("save_lmbda_c_eq_tilde_max_mesh", False)
    post_processing.add("save_lmbda_c_eq_tilde_max_chunks", False)

    post_processing.add("save_lmbda_nu_tilde_max_mesh", False)
    post_processing.add("save_lmbda_nu_tilde_max_chunks", False)

    post_processing.add("save_g_mesh", False)
    post_processing.add("save_g_chunks", False)

    post_processing.add("save_upsilon_c_mesh", False)
    post_processing.add("save_upsilon_c_chunks", False)

    post_processing.add("save_Upsilon_c_mesh", True)
    post_processing.add("save_Upsilon_c_chunks", True)

    post_processing.add("save_d_c_mesh", False)
    post_processing.add("save_d_c_chunks", False)

    post_processing.add("save_D_c_mesh", True)
    post_processing.add("save_D_c_chunks", True)

    post_processing.add("save_epsilon_cnu_diss_hat_mesh", False)
    post_processing.add("save_epsilon_cnu_diss_hat_chunks", False)

    post_processing.add("save_Epsilon_cnu_diss_hat_mesh", False)
    post_processing.add("save_Epsilon_cnu_diss_hat_chunks", False)

    post_processing.add("save_epsilon_c_diss_hat_mesh", False)
    post_processing.add("save_epsilon_c_diss_hat_chunks", False)

    post_processing.add("save_Epsilon_c_diss_hat_mesh", False)
    post_processing.add("save_Epsilon_c_diss_hat_chunks", False)

    post_processing.add("save_overline_epsilon_cnu_diss_hat_mesh", False)
    post_processing.add("save_overline_epsilon_cnu_diss_hat_chunks", False)

    post_processing.add("save_overline_Epsilon_cnu_diss_hat_mesh", False)
    post_processing.add("save_overline_Epsilon_cnu_diss_hat_chunks", False)

    post_processing.add("save_overline_epsilon_c_diss_hat_mesh", False)
    post_processing.add("save_overline_epsilon_c_diss_hat_chunks", False)

    post_processing.add("save_overline_Epsilon_c_diss_hat_mesh", False)
    post_processing.add("save_overline_Epsilon_c_diss_hat_chunks", False)

    post_processing.add("save_u_mesh", True)
    post_processing.add("save_u_chunks", True)

    post_processing.add("save_F_mesh", True)
    post_processing.add("save_F_chunks", True)

    post_processing.add("save_sigma_mesh", True)
    post_processing.add("save_sigma_chunks", True)

    post_processing.add("axes_linewidth", 1.0)
    post_processing.add("font_family", "sans-serif")
    post_processing.add("text_usetex", True)
    post_processing.add("ytick_right", True)
    post_processing.add("ytick_direction", "in")
    post_processing.add("xtick_top", True)
    post_processing.add("xtick_direction", "in")
    post_processing.add("xtick_minor_visible", True)

    return post_processing
