#################################################
nonlocal composite uFJC scission network fracture
#################################################

|build| |license|

A repository that incorporates the composite uFJC model within a localizing gradient-enhanced damage model (LGEDM) for elastomers. This repository is dependent upon the `composite-uFJC-scission-UFL-FEniCS <https://pypi.org/project/composite-ufjc-scission-ufl-fenics/>`_ Python package. It is also dependent upon the latest version of legacy FEniCS (2019.1.0).

*****
Setup
*****

Once the contents of the repository have been cloned or downloaded, the Python virtual environment associated with the project needs to be installed. The installation of this Python virtual environment and some essential packages is handled by the ``fenics-virtual-environment-install-master.sh`` Bash script. Before running the script, make sure to change the ``VENV_PATH`` parameter to comply with your personal filetree structure. Also, please refer to the general and most up-to-date Internet presence of FEniCS for assistance in the proper installation of legacy FEniCS (2019.1.0). After the Python virtual environment has been installed, the  ``nonlocal-composite-ufjc-scission-network-fracture`` package needs to be installed in the virtual environment by moving to the main project directory (where the setup and configuration files are located), and (with the virtual environment activated) executing ``pip install -e .`` for an editable installation or ``pip install .`` for a standard installation.

*****
Usage
*****

The ``nonlocal_composite_ufjc_scission_network_fracture`` directory contains the following files that constitute the ``nonlocal-composite-ufjc-scission-network-fracture`` package: ``composite_ufjc_network.py``, ``default_parameters.py``, ``problem.py``, and ``utility.py``. The ``notched-crack`` and ``refined-notched-crack`` directories respectively contain ``notched_crack.py`` and ``refined_notched_crack.py``, the main executible files for this repository. Each of these files creates the finite element problem for the elastomer LGEDM, solves the problem, and (to a certain extent) post-processes results. Before using these codes, it is highly recommended that you carefully examine each and every one of the aforementioned Python codes to understand how the code works as a whole, how certain parts of the code depend on other packages (especially ``nonlocal-composite-ufjc-scission-network-fracture``, FEniCS (2019.1.0), and ``composite-ufjc-scission-ufl-fenics``), how certain parts of the code relate to one another, and how the code is generally structured (in an object-oriented fashion). If necessary, feel free to modify any of these codes for your purposes.

In order to run either of the main executible files in serial, first activate the Python virtual environment, and then execute the following command in the terminal

::

    python3 {notched_crack, refined_notched_crack}.py

In order to run either of the main executible files in parallel (thanks to the parallel computing capabilities of FEniCS), first activate the Python virtual environment, and then execute the following command in the terminal

::

    mpirun -np number_of_cores python3 {notched_crack, refined_notched_crack}.py

Do note that the codes published as is in this repository are unable to be run in parallel (due to the ASCII-encoding of results in XDMF files, which is required for the most recent versions of ParaView to be able to open XDMF files produced via FEniCS).

***********
Information
***********

- `License <https://github.com/jasonmulderrig/nonlocal-composite-uFJC-scission-network-fracture/LICENSE>`__
- `Releases <https://github.com/jasonmulderrig/nonlocal-composite-uFJC-scission-network-fracture/releases>`__
- `Repository <https://github.com/jasonmulderrig/nonlocal-composite-uFJC-scission-network-fracture>`__

********
Citation
********

\Jason Mulderrig, Brandon Talamini, and Nikolaos Bouklas, ``composite-ufjc-scission-ufl-fenics``: the Python package for the composite uFJC model with scission implemented in the Unified Form Language (UFL) in FEniCS, `Zenodo (2023) <https://doi.org/10.5281/zenodo.7738019>`_.

\Jason Mulderrig, Brandon Talamini, and Nikolaos Bouklas, ``composite-ufjc-scission``: the Python package for the composite uFJC model with scission, `Zenodo (2022) <https://doi.org/10.5281/zenodo.7335564>`_.

\Jason Mulderrig, Brandon Talamini, and Nikolaos Bouklas, Statistical mechanics-based gradient-enhanced damage for elastomeric materials, In preparation.

\Jason Mulderrig, Brandon Talamini, and Nikolaos Bouklas, A statistical mechanics framework for polymer chain scission, based on the concepts of distorted bond potential and asymptotic matching, `Journal of the Mechanics and Physics of Solids 174, 105244 (2023) <https://www.sciencedirect.com/science/article/pii/S0022509623000480>`_.

..
    Badges ========================================================================

.. |build| image:: https://img.shields.io/github/checks-status/jasonmulderrig/nonlocal-composite-uFJC-scission-network-fracture/main?label=GitHub&logo=github
    :target: https://github.com/jasonmulderrig/nonlocal-composite-uFJC-scission-network-fracture

.. |license| image:: https://img.shields.io/github/license/jasonmulderrig/nonlocal-composite-uFJC-scission-network-fracture?label=License
    :target: https://github.com/jasonmulderrig/nonlocal-composite-uFJC-scission-network-fracture/LICENSE