#!/bin/bash

export sudo DEBIAN_FRONTEND=noninteractive && \
sudo apt-get -qq update && \
sudo apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
sudo apt-get -y install \
build-essential \
clang-10 \
g++ \
gfortran \
git \
dpkg \
dpkg-dev \
libgl1-mesa-dev \
libboost-dev \
libboost-all-dev \
libboost-filesystem-dev \
libboost-iostreams-dev \
libboost-math-dev \
libboost-program-options-dev \
libboost-system-dev \
libboost-thread-dev \
libboost-timer-dev \
libeigen3-dev \
libgmp-dev \
libhdf5-mpich-dev \
liblapack-dev \
libmpich-dev \
libmpfr-dev \
libopenblas-dev \
libopenmpi-dev \
libqt5help5 \
libqt5svg5-dev \
libqt5x11extras5-dev \
libssl-dev \
libtbb-dev \
libxt-dev \
ninja-build \
pkg-config \
python3-dev \
python3-matplotlib \
python3-numpy \
python3-pip \
python3-scipy \
python3-setuptools \
python3-venv \
qttools5-dev \
qtxmlpatterns5-dev-tools \
qt5-default \
wget \
zlib1g-dev

sudo apt-get install software-properties-common
sudo apt-add-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install --no-install-recommends fenics


# For the workstation AnalyticalEngine
VENV_PATH="/home/jason_mulderrig/research/projects/nonlocal-composite-uFJC-scission-network-fracture"
COMPUTATIONAL_PLATFORMS_PATH="/home/jason_mulderrig/computational-platforms"
FENICS_VERSION="2019.1.0"

# # For the laptop WSL2 running Ubuntu 20.04
# VENV_PATH="/home/jasonmulderrig/research/projects/temp"
# COMPUTATIONAL_PLATFORMS_PATH="/home/jasonmulderrig/computational-platforms"

if [ ! -d ${COMPUTATIONAL_PLATFORMS_PATH} ]
then
  mkdir -p ${COMPUTATIONAL_PLATFORMS_PATH}
fi

# # Install CMake --- THIS WORKS
# sudo apt remove --purge cmake
# CMAKE_VERSION=3.19.1
# CMAKE_DIR="${COMPUTATIONAL_PLATFORMS_PATH}/cmake-${CMAKE_VERSION}"
#
# # Uncomment the code below to remove and reinstall CMake
# if [ ! -d ${CMAKE_DIR} ]
# then
#   cd ${COMPUTATIONAL_PLATFORMS_PATH}
#   wget -nc --quiet https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz && \
#   tar -xf cmake-${CMAKE_VERSION}.tar.gz && \
#   cd ${CMAKE_DIR} && \
#   ./bootstrap && make && sudo make install
# # else
# #   sudo rm -rf ${CMAKE_DIR}
# #   cd ${COMPUTATIONAL_PLATFORMS_PATH}
# #   wget -nc --quiet https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz && \
# #   tar -xf cmake-${CMAKE_VERSION}.tar.gz && \
# #   cd ${CMAKE_DIR} && \
# #   ./bootstrap && make && sudo make install
# fi

# # # Install Paraview - Jason has currently not checked this
# PARAVIEW_VERSION=5.8.0 # this is what Jason has currently installed on his workstation from the procedure below
# PARAVIEW_DIR="${COMPUTATIONAL_PLATFORMS_PATH}/paraview"
#
# # Uncomment the code below to remove and reinstall Paraview
# if [ ! -d ${PARAVIEW_DIR} ]
# then
#   cd ${COMPUTATIONAL_PLATFORMS_PATH}
#   git clone https://gitlab.kitware.com/paraview/paraview.git
#   mkdir paraview-build
#   cd paraview
#   git checkout v${PARAVIEW_VERSION}
#   git submodule update --init --recursive
#   cd ../paraview-build
#   cmake -GNinja -DPARAVIEW_USE_PYTHON=ON -DPARAVIEW_USE_MPI=ON -DVTK_SMP_IMPLEMENTATION_TYPE=TBB -DCMAKE_BUILD_TYPE=Release ../paraview
#   ninja
# # else
# #   sudo rm -rf ${PARAVIEW_DIR}
# #   cd ${COMPUTATIONAL_PLATFORMS_PATH}
# #   git clone https://gitlab.kitware.com/paraview/paraview.git
# #   mkdir paraview-build
# #   cd paraview
# #   git checkout v${PARAVIEW_VERSION}
# #   git submodule update --init --recursive
# #   cd ../paraview-build
# #   cmake -GNinja -DPARAVIEW_USE_PYTHON=ON -DPARAVIEW_USE_MPI=ON -DVTK_SMP_IMPLEMENTATION_TYPE=TBB -DCMAKE_BUILD_TYPE=Release ../paraview
# #   ninja
# fi


# Set up Python virtual environment, Python packages, and FEniCS

if [ ! -d ${VENV_PATH} ]
then
  mkdir -p ${VENV_PATH}
  python3 -m venv ${VENV_PATH}
  cd ${VENV_PATH}
else
  cd ${VENV_PATH}
  if [ ! -f pyvenv.cfg ]
  then
    python3 -m venv ${VENV_PATH}
  else
    rm -rf bin include lib share && rm lib64 && rm pyvenv.cfg
    python3 -m venv ${VENV_PATH}
  fi
fi

source bin/activate

pip3 install wheel && pip3 install --upgrade setuptools && pip3 install --upgrade pip
pip3 install numpy==1.21.6 scipy mpmath sympy matplotlib pynverse
pip3 install mpi4py

pip3 uninstall gmsh-dev gmsh-sdk gmsh-sdk-git
pip3 install --upgrade gmsh
pip3 install meshio[all] pygmsh

pip3 install pkgconfig ply pybind11

pip3 uninstall fenics-fiat fenics-dijitso fenics-ufl fenics-ffc fenics-dolfin mshr

sudo rm -rf fiat dijitso ufl ffc dolfin mshr

git clone --branch=${FENICS_VERSION} https://bitbucket.org/fenics-project/fiat
git clone --branch=${FENICS_VERSION} https://bitbucket.org/fenics-project/dijitso
git clone --branch=${FENICS_VERSION} https://bitbucket.org/fenics-project/ufl
git clone --branch=${FENICS_VERSION} https://bitbucket.org/fenics-project/ffc
git clone --branch=${FENICS_VERSION} https://bitbucket.org/fenics-project/dolfin
git clone --branch=${FENICS_VERSION} https://bitbucket.org/fenics-project/mshr
cd fiat    && pip3 install . && cd ..
cd dijitso && pip3 install . && cd ..
cd ufl     && pip3 install . && cd ..
cd ffc     && pip3 install . && cd ..
mkdir dolfin/build && cd dolfin/build && cmake .. && sudo make install && cd ../..
mkdir mshr/build   && cd mshr/build   && cmake .. && sudo make install && cd ../..
cd dolfin/python && pip3 install . && cd ../..
cd mshr/python   && pip3 install . && cd ../..

pip3 install composite-ufjc-scission-ufl-fenics==1.4.0

deactivate
