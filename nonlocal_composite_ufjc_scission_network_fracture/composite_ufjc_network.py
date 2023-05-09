# Import necessary libraries
from __future__ import division
from dolfin import *
from composite_ufjc_scission_ufl_fenics import (
    RateIndependentScissionCompositeuFJCUFLFEniCS,
    RateDependentScissionCompositeuFJCUFLFEniCS,
    RateIndependentSmoothstepScissionCompositeuFJCUFLFEniCS,
    RateDependentSmoothstepScissionCompositeuFJCUFLFEniCS,
    RateIndependentSigmoidScissionCompositeuFJCUFLFEniCS,
    RateDependentSigmoidScissionCompositeuFJCUFLFEniCS
)
# from .microsphere_quadrature import MicrosphereQuadratureScheme
# from .microcircle_quadrature import MicrocircleQuadratureScheme
from .utility import none_str2nonetype
import sys
import numpy as np


class CompositeuFJCNetwork(object):
    
    def __init__(self, material_parameters):

        mp = material_parameters

        # Access specified parameters
        self.scission_model = mp["scission_model"]
        self.network_model = mp["network_model"]
        self.physical_dimension = mp["physical_dimension"]
        self.physical_dimensionality = mp["physical_dimensionality"]
        self.incompressibility_assumption = mp["incompressibility_assumption"]
        self.macro2micro_deformation_assumption = mp["macro2micro_deformation_assumption"]
        self.micro2macro_homogenization_scheme = mp["micro2macro_homogenization_scheme"]
        self.chain_level_load_sharing = mp["chain_level_load_sharing"]
        self.rate_dependence = mp["rate_dependence"]
        self.two_dimensional_formulation = mp["two_dimensional_formulation"]
        self.microcircle_quadrature_order = mp["microcircle_quadrature_order"]
        self.microsphere_quadrature_order = mp["microsphere_quadrature_order"]
        self.omega_0 = none_str2nonetype(mp["omega_0"])

        # Verify the correctness of the specified parameters
        if self.physical_dimension == 2 and self.physical_dimensionality != "two_dimensional":
            sys.exit("Error: There is a mismatch between the stated physical dimensionality of the problem and the physical dimension number.")
        
        if self.physical_dimension == 3 and self.physical_dimensionality != "three_dimensional":
            sys.exit("Error: There is a mismatch between the stated physical dimensionality of the problem and the physical dimension number.")
        
        if self.scission_model != "analytical" and self.scission_model != "smoothstep" and self.scission_model != "sigmoid":
            sys.exit("Error: Need to specify the scission model. Either the analytical, smoothstep, or sigmoid scission models can be used.")
        
        if self.network_model != "statistical_mechanics_model":
            sys.exit("Error: This composite uFJC material class strictly corresponds to a statistical mechanics model.")
        
        if self.physical_dimension != 2 and self.physical_dimension != 3:
            sys.exit("Error: Need to specify either a 2D or a 3D problem.")
        
        if self.incompressibility_assumption != "incompressible" and self.incompressibility_assumption != "nearly_incompressible" and self.incompressibility_assumption != "compressible":
            sys.exit("Error: Need to specify a proper incompressibility assumption for the material. The material is either incompressible, nearly incompressible, or compressible.")

        if self.macro2micro_deformation_assumption != 'affine' and self.macro2micro_deformation_assumption != 'nonaffine':
            sys.exit('Error: Need to specify the macro-to-micro deformation assumption in the network. Either affine deformation or non-affine deformation can be used.')
        
        if self.micro2macro_homogenization_scheme != 'eight_chain_model' and self.micro2macro_homogenization_scheme != 'full_network_microcircle_model' and self.micro2macro_homogenization_scheme != 'full_network_microsphere_model':
            sys.exit('Error: Need to specify the micro-to-macro homogenization scheme in the network. Either the eight chain model, the full network microcircle micro-to-macro homogenization scheme, or the full network microsphere micro-to-macro homogenization scheme can be used.')
        
        if self.chain_level_load_sharing != 'equal_strain' and self.chain_level_load_sharing != 'equal_force':
            sys.exit('Error: Need to specify the load sharing assumption that the chains in the composite uFJC network obey. Either the equal strain chain level load sharing assumption or the equal force chain level load sharing assumption can be used.')
        
        if self.rate_dependence != 'rate_dependent' and self.rate_dependence != 'rate_independent':
            sys.exit('Error: Need to specify the network/chain dependence on the rate of applied deformation. Either rate-dependent or rate-independent deformation can be used.')
        
        if self.rate_dependence == 'rate_dependent' and self.omega_0 is None:
            sys.exit('Error: Need to specify the microscopic frequency of segments in the network for rate-dependent network deformation.')
        
        if self.macro2micro_deformation_assumption == 'affine' and self.micro2macro_homogenization_scheme == 'eight_chain_model':
            sys.exit('Error: The eight chain micro-to-macro homogenization scheme technically exhibits the non-affine macro-to-micro deformation assumption.')
        
        if self.macro2micro_deformation_assumption == 'nonaffine' and (self.micro2macro_homogenization_scheme == 'full_network_microsphere_model' or self.micro2macro_homogenization_scheme == 'full_network_microcircle_model') and self.chain_level_load_sharing == 'equal_strain':
            sys.exit('Error: In the non-affine macro-to-micro deformation assumption utilizing either the full network microsphere micro-to-macro homogenization scheme or the full network microcircle micro-to-macro homogenization scheme, the composite uFJCs are required to obey the equal force load sharing assumption.')
        
        if self.physical_dimension == 2:
            if self.two_dimensional_formulation != "plane_strain" and self.two_dimensional_formulation != "generalized_plane_strain" and self.two_dimensional_formulation != "plane_stress":
                sys.exit("Error: Need to specify a proper two-dimensional formulation. Either plane strain, generalized plane strain, or plane stress can be used.")
            
            if self.micro2macro_homogenization_scheme == 'full_network_microsphere_model':
                sys.exit("Error: For a 2D problem, the full network microsphere micro-to-macro homogenization scheme cannot be used. Either the eight chain model or the full network microcircle micro-to-macro homogenization scheme can be used for 2D problems.")
            
            # # Specify full network microcircle quadrature scheme, if necessary
            # elif self.micro2macro_homogenization_scheme == 'full_network_microcircle_model':
            #     if self.microcircle_quadrature_order is None:
            #         sys.exit('Error: Need to specify microcircle quadrature order number in order to utilize the full network microcircle micro-to-macro homogenization scheme.')
            #     else:
            #         self.microcircle = MicrocircleQuadratureScheme(self.microcircle_quadrature_order)
        
        elif self.physical_dimension == 3:
            if self.micro2macro_homogenization_scheme == 'full_network_microcircle_model':
                sys.exit("Error: For a 2D problem, the full network microcircle micro-to-macro homogenization scheme cannot be used. Either the eight chain model or the full network microsphere micro-to-macro homogenization scheme can be used for 2D problems.")
            
            # # Specify full network microsphere quadrature scheme, if necessary
            # elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model':
            #     if self.microsphere_quadrature_order is None:
            #         sys.exit('Error: Need to specify microsphere quadrature order number in order to utilize the full network microsphere micro-to-macro homogenization scheme.')
            #     else:
            #         self.microsphere = MicrosphereQuadratureScheme(self.microsphere_quadrature_order)
        
        # Specify chain-level load sharing and chain composition
        if self.chain_level_load_sharing == 'equal_strain':
            self.equal_strain_composite_ufjc_network(mp)
        elif self.chain_level_load_sharing == 'equal_force':
            self.equal_force_composite_ufjc_network()
    
    def equal_strain_composite_ufjc_network(self, material_parameters):
        
        mp = material_parameters
        
        # Extract chain segment polydispersity information
        self.nu_num = mp["nu_num"]
        nu_list = []
        P_nu_list = []
        
        for nu_indx in range(self.nu_num):
            nu_val = mp["nu_indx_"+str(nu_indx)+"_nu_val"]
            P_nu_val = self.P_nu(mp, nu_val)
            nu_list.append(nu_val)
            P_nu_list.append(P_nu_val)
        
        P_nu_sum = np.sum(P_nu_list)
        
        # Extract fundamental chain composition parameters and chain 
        # scission parameters
        nu_b = none_str2nonetype(mp["nu_b"])
        zeta_b_char = none_str2nonetype(mp["zeta_b_char"])
        kappa_b = none_str2nonetype(mp["kappa_b"])
        zeta_nu_char = none_str2nonetype(mp["zeta_nu_char"])
        kappa_nu = none_str2nonetype(mp["kappa_nu"])
        
        lmbda_nu_crit_min = none_str2nonetype(mp["lmbda_nu_crit_min"])
        lmbda_nu_crit_max = none_str2nonetype(mp["lmbda_nu_crit_max"])
        tau = none_str2nonetype(mp["tau"])
        lmbda_nu_check = none_str2nonetype(mp["lmbda_nu_check"])
        
        if self.rate_dependence == 'rate_independent' and self.scission_model == 'analytical':
            composite_ufjc_ufl_fenics_list = [
                RateIndependentScissionCompositeuFJCUFLFEniCS(nu=nu_list[nu_indx],
                                                    nu_b=nu_b,
                                                    zeta_b_char=zeta_b_char,
                                                    kappa_b=kappa_b,
                                                    zeta_nu_char=zeta_nu_char,
                                                    kappa_nu=kappa_nu)
                for nu_indx in range(self.nu_num)
            ]
        elif self.rate_dependence == 'rate_dependent' and self.scission_model == 'analytical':
            composite_ufjc_ufl_fenics_list = [
                RateDependentScissionCompositeuFJCUFLFEniCS(omega_0=self.omega_0,
                                                    nu=nu_list[nu_indx],
                                                    nu_b=nu_b,
                                                    zeta_b_char=zeta_b_char,
                                                    kappa_b=kappa_b,
                                                    zeta_nu_char=zeta_nu_char,
                                                    kappa_nu=kappa_nu)
                for nu_indx in range(self.nu_num)
            ]
        elif self.rate_dependence == 'rate_independent' and self.scission_model == 'smoothstep':
            composite_ufjc_ufl_fenics_list = [
                RateIndependentSmoothstepScissionCompositeuFJCUFLFEniCS(nu=nu_list[nu_indx],
                                                    nu_b=nu_b,
                                                    zeta_b_char=zeta_b_char,
                                                    kappa_b=kappa_b,
                                                    zeta_nu_char=zeta_nu_char,
                                                    kappa_nu=kappa_nu,
                                                    lmbda_nu_crit_min=lmbda_nu_crit_min,
                                                    lmbda_nu_crit_max=lmbda_nu_crit_max)
                for nu_indx in range(self.nu_num)
            ]
        elif self.rate_dependence == 'rate_dependent' and self.scission_model == 'smoothstep':
            composite_ufjc_ufl_fenics_list = [
                RateDependentSmoothstepScissionCompositeuFJCUFLFEniCS(omega_0=self.omega_0,
                                                    nu=nu_list[nu_indx],
                                                    nu_b=nu_b,
                                                    zeta_b_char=zeta_b_char,
                                                    kappa_b=kappa_b,
                                                    zeta_nu_char=zeta_nu_char,
                                                    kappa_nu=kappa_nu,
                                                    lmbda_nu_crit_min=lmbda_nu_crit_min,
                                                    lmbda_nu_crit_max=lmbda_nu_crit_max)
                for nu_indx in range(self.nu_num)
            ]
        elif self.rate_dependence == 'rate_independent' and self.scission_model == 'sigmoid':
            composite_ufjc_ufl_fenics_list = [
                RateIndependentSigmoidScissionCompositeuFJCUFLFEniCS(nu=nu_list[nu_indx],
                                                    nu_b=nu_b,
                                                    zeta_b_char=zeta_b_char,
                                                    kappa_b=kappa_b,
                                                    zeta_nu_char=zeta_nu_char,
                                                    kappa_nu=kappa_nu,
                                                    tau=tau,
                                                    lmbda_nu_check=lmbda_nu_check)
                for nu_indx in range(self.nu_num)
            ]
        elif self.rate_dependence == 'rate_dependent' and self.scission_model == 'sigmoid':
            composite_ufjc_ufl_fenics_list = [
                RateDependentSigmoidScissionCompositeuFJCUFLFEniCS(omega_0=self.omega_0,
                                                    nu=nu_list[nu_indx],
                                                    nu_b=nu_b,
                                                    zeta_b_char=zeta_b_char,
                                                    kappa_b=kappa_b,
                                                    zeta_nu_char=zeta_nu_char,
                                                    kappa_nu=kappa_nu,
                                                    tau=tau,
                                                    lmbda_nu_check=lmbda_nu_check)
                for nu_indx in range(self.nu_num)
            ]

        # Reformat chain segment polydispersity information to FEniCS
        nu_list = [Constant(nu_list[nu_indx]) for nu_indx in range(self.nu_num)]
        A_nu_list = [composite_ufjc_ufl_fenics_list[nu_indx].A_nu for nu_indx in range(self.nu_num)]
        Lambda_nu_ref_list = [composite_ufjc_ufl_fenics_list[nu_indx].Lambda_nu_ref for nu_indx in range(self.nu_num)]
        P_nu_list = [Constant(P_nu_list[nu_indx]) for nu_indx in range(self.nu_num)]
        P_nu_sum = Constant(P_nu_sum)
        
        self.nu_list  = nu_list
        self.A_nu_list = A_nu_list
        self.Lambda_nu_ref_list = Lambda_nu_ref_list
        self.P_nu_list = P_nu_list
        self.P_nu_sum = P_nu_sum
        
        # Retain specified parameters
        self.composite_ufjc_ufl_fenics_list = composite_ufjc_ufl_fenics_list

        self.zeta_nu_char = composite_ufjc_ufl_fenics_list[0].zeta_nu_char
        self.kappa_nu = composite_ufjc_ufl_fenics_list[0].kappa_nu
        self.lmbda_nu_ref = composite_ufjc_ufl_fenics_list[0].lmbda_nu_ref
        self.lmbda_c_eq_ref = composite_ufjc_ufl_fenics_list[0].lmbda_c_eq_ref
        self.lmbda_nu_crit = composite_ufjc_ufl_fenics_list[0].lmbda_nu_crit
        self.lmbda_c_eq_crit = composite_ufjc_ufl_fenics_list[0].lmbda_c_eq_crit
        self.xi_c_crit = composite_ufjc_ufl_fenics_list[0].xi_c_crit
        self.lmbda_nu_pade2berg_crit = composite_ufjc_ufl_fenics_list[0].lmbda_nu_pade2berg_crit
        self.lmbda_c_eq_pade2berg_crit = composite_ufjc_ufl_fenics_list[0].lmbda_c_eq_pade2berg_crit
    
    def equal_force_composite_ufjc_network(self):
        sys.exit("The equal force chain-level load sharing implementation has not been finalized yet. For now, only equal strain chain-level load sharing is permitted.")

    def P_nu(self, material_parameters, nu):
        
        mp = material_parameters

        if mp["nu_distribution"] == "itskov":
            return (1/(mp["Delta_nu"]+1))*(1+(1/mp["Delta_nu"]))**(mp["nu_min"]-nu)
        if mp["nu_distribution"] == "uniform":
            return 1./self.nu_num