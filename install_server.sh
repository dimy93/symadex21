#!/bin/bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 777 Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
rm ./Miniconda3-latest-Linux-x86_64.sh

source ~/miniconda3/bin/activate

ELINA_VER=62ccc06539d8103a436e219ed2cfb78e489048f6 
CONDA_ENV=eran_symadex
conda create -n ${CONDA_ENV} python=3.7
conda activate ${CONDA_ENV}

conda install -c anaconda tensorflow==1.15

GEOMETRIC_HOME="$(pwd)/ELINA_install/build"
GEOMETRIC_HOME_BIN="$(pwd)/ELINA_install/build/bin"
GEOMETRIC_HOME_LIB="$(pwd)/ELINA_install/build/lib"

mkdir -p ${GEOMETRIC_HOME}
mkdir -p ${GEOMETRIC_HOME_BIN}
mkdir -p ${GEOMETRIC_HOME_LIB}

export LD_LIBRARY_PATH=${GEOMETRIC_HOME_BIN}/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${GEOMETRIC_HOME_BIN}/lib:${LIBRARY_PATH}
export C_INCLUDE_PATH=${GEOMETRIC_HOME_BIN}/include:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=${GEOMETRIC_HOME_BIN}/include:${CPLUS_INCLUDE_PATH}
export PATH=${GEOMETRIC_HOME_BIN}/bin:${PATH}

set -e

has_cuda=0

while : ; do
    case "$1" in
        "")
            break;;
        -use-cuda|--use-cuda)
         has_cuda=1;;
        *)
            echo "unknown option $1, try -help"
            exit 2;;
    esac
    shift
done

wget ftp://ftp.gnu.org/gnu/m4/m4-1.4.1.tar.gz
tar -xvzf m4-1.4.1.tar.gz
cd m4-1.4.1
./configure --prefix "${GEOMETRIC_HOME_LIB}" --exec-prefix="${GEOMETRIC_HOME_BIN}"
make
make install
cd ..
rm m4-1.4.1.tar.gz

wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
tar -xvf gmp-6.1.2.tar.xz
cd gmp-6.1.2
./configure --enable-cxx --prefix "${GEOMETRIC_HOME_LIB}" --exec-prefix="${GEOMETRIC_HOME_BIN}"
make
make install
cd ..
rm gmp-6.1.2.tar.xz

export GMP_PREFIX="${GEOMETRIC_HOME_BIN}"

wget https://files.sri.inf.ethz.ch/eran/mpfr/mpfr-4.1.0.tar.xz
tar -xvf mpfr-4.1.0.tar.xz
cd mpfr-4.1.0
./configure --prefix "${GEOMETRIC_HOME_LIB}" --exec-prefix="${GEOMETRIC_HOME_BIN}" --with-gmp-include="${GMP_PREFIX}/include/" --with-gmp-lib="${GMP_PREFIX}/lib/"
make
make install
cd ..
rm mpfr-4.1.0.tar.xz


echo "$(pwd)"
cp -r "${GEOMETRIC_HOME_LIB}/include/"* "${GEOMETRIC_HOME_BIN}/include"
export MPFR_PREFIX="${GEOMETRIC_HOME_BIN}"

wget https://packages.gurobi.com/9.0/gurobi9.0.3_linux64.tar.gz
tar -xvf gurobi9.0.3_linux64.tar.gz
cd gurobi903/linux64/src/build
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
make
cp libgurobi_c++.a ../../lib/
cp ../../lib/libgurobi90.so ${GEOMETRIC_HOME_BIN}/lib
cd ../..
python3 setup.py install
cd ../..
rm gurobi9.0.3_linux64.tar.gz

export GUROBI_HOME="$(pwd)/gurobi903/linux64"
export PATH="${GUROBI_HOME}/bin:${PATH}"
export CPATH="${GUROBI_HOME}/include:${CPATH}"
export LD_LIBRARY_PATH=${GUROBI_HOME}/lib:${LD_LIBRARY_PATH}

git clone https://github.com/eth-sri/ELINA.git
cd ELINA
git checkout ${ELINA_VER}
if test "$has_cuda" -eq 1
then
    ./configure -use-cuda --prefix "${GEOMETRIC_HOME_BIN}"
else
    ./configure --prefix "${GEOMETRIC_HOME_BIN}"
fi

make
make install
cd ..

git clone https://github.com/eth-sri/deepg.git
cd deepg/code
mkdir build
make shared_object
cp ./build/libgeometric.so ${GEOMETRIC_HOME_BIN}/lib
cd ../..


pip install -r requirements.txt

mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d
touch ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
touch ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh

echo -e '#!/bin/sh\n\nif [[ -v OLD_LD_LIBRARY_PATH ]];\nthen\n\texport LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}\nfi\nif [[ -v OLD_LIBRARY_PATH ]];\nthen\n\texport LIBRARY_PATH=${OLD_LIBRARY_PATH}\nfi\nif [[ -v OLD_PATH ]];\nthen\n\texport PATH=${OLD_PATH}\nfi\nif [[ -v OLD_C_INCLUDE_PATH ]];\nthen\n\texport C_INCLUDE_PATH=${OLD_C_INCLUDE_PATH}\nfi\nif [[ -v OLD_CPATH ]];\nthen\n\texport CPATH=${OLD_CPATH}\nfi\nif [[ -v OLD_CPLUS_INCLUDE_PATH ]];\nthen\n\texport CPLUS_INCLUDE_PATH=${OLD_CPLUS_INCLUDE_PATH}\nfi\n\nunset OLD_LD_LIBRARY_PATH\nunset OLD_LIBRARY_PATH\nunset OLD_PATH\nunset OLD_C_INCLUDE_PATH\nunset OLD_CPATH\nunset OLD_CPLUS_INCLUDE_PATH\n' > ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh

echo -e '#!/bin/sh\n\nexport OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}\nexport LD_LIBRARY_PATH='"${GEOMETRIC_HOME_BIN}/lib:${GUROBI_HOME}/lib:"'${LD_LIBRARY_PATH}\n\nexport OLD_LIBRARY_PATH=${LIBRARY_PATH}\nexport LIBRARY_PATH='"${GEOMETRIC_HOME_BIN}/lib:${GUROBI_HOME}/lib:"'${LIBRARY_PATH}\n\nexport OLD_PATH=${PATH}\nexport PATH='"${GEOMETRIC_HOME_BIN}/bin:${GUROBI_HOME}/bin:"'${PATH}\n\nexport OLD_C_INCLUDE_PATH=${C_INCLUDE_PATH}\nexport C_INCLUDE_PATH='"${GEOMETRIC_HOME_BIN}/include:${GUROBI_HOME}/include:"'${C_INCLUDE_PATH}\n\nexport OLD_CPATH=${CPATH}\nexport CPATH='"${GEOMETRIC_HOME_BIN}/include:${GUROBI_HOME}/include:"'${CPATH}\n\nexport OLD_CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}\nexport CPLUS_INCLUDE_PATH='"${GEOMETRIC_HOME_BIN}/include:${GUROBI_HOME}/include:"'${CPLUS_INCLUDE_PATH}' > ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh

