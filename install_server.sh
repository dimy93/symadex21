#!/bin/bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 777 Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
rm ./Miniconda3-latest-Linux-x86_64.sh

git checkout 136b5ec5970e41036b109922f32525c53c1a4067 
cd tf_verify
cp ../patch_ERAN.txt ./
git apply patch_ERAN.txt
cd ../

install=0
has_cuda=0
CONDA_ENV=eran_symadex_geometric

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -c|--use-cuda)
    has_cuda=1
    shift # past argument
    ;;
    -n|--name)
    CONDA_ENV="$2"
    shift # past argument
    shift # past value
    ;;
    -i|--install-conda)
    install=1
    shift # past argument
    ;;
esac
done

if test "$install" -eq 1
then
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	chmod 777 Miniconda3-latest-Linux-x86_64.sh
	./Miniconda3-latest-Linux-x86_64.sh
	rm ./Miniconda3-latest-Linux-x86_64.sh
fi

source ~/miniconda3/bin/activate

conda create -n ${CONDA_ENV} python=3.7
conda activate ${CONDA_ENV}

conda install -c conda-forge tensorflow==1.15

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

if test "$has_cuda" -eq 1
then
    conda install -c conda-forge cudatoolkit-dev
fi


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

mkdir -p cddlib_tools
export CDD_PATH=$PATH
export PATH=$PWD/cddlib_tools/bin:$PATH


wget ftp://ftp.gnu.org/gnu/m4/m4-1.4.18.tar.gz
tar -xvzf m4-1.4.18.tar.gz
cp m4-1.4.18-glibc-change-work-around.patch ./m4-1.4.18/lib
cd m4-1.4.18
cd lib
export lddver=$(ldd --version | head -1 | awk '{print $NF}')
if [ "$lddver" \> "2.27" ];
then
	        patch < m4-1.4.18-glibc-change-work-around.patch
fi
cd ..
./configure --prefix "$PWD/../cddlib_tools/"
make
make install
cd ..
rm m4-1.4.18.tar.gz

wget http://ftp.gnu.org/gnu/autoconf/autoconf-2.69.tar.gz
tar xf autoconf*
cd autoconf-2.69
./configure --prefix "$PWD/../cddlib_tools/"
make install
cd ..

wget http://ftp.gnu.org/gnu/automake/automake-1.16.2.tar.gz
tar xf automake*
cd automake-1.16.2
./configure --prefix "$PWD/../cddlib_tools/"
make install
cd ..

wget http://ftp.gnu.org/gnu/libtool/libtool-2.4.6.tar.gz
tar xf libtool*
cd libtool-2.4.6
./configure --prefix "$PWD/../cddlib_tools/"
make install
cd ..

git clone https://github.com/cddlib/cddlib.git
cd cddlib
./bootstrap || true
libtoolize
./bootstrap
./configure --prefix "${GEOMETRIC_HOME_LIB}" --exec-prefix="${GEOMETRIC_HOME_BIN}" --with-gmp-include="${GMP_PREFIX}/include/" --with-gmp-lib="${GMP_PREFIX}/lib/"
sed -i 's/= doc lib-src src/= lib-src src/' Makefile.am
sed -i 's#CC = gcc#CC = gcc -I '"${GMP_PREFIX}/include/"'#' lib-src/Makefile
sed -i 's#CC = gcc#CC = gcc -I '"${GMP_PREFIX}/include/"'#' src/Makefile
make
make install
cd ..

cp -r "${GEOMETRIC_HOME_LIB}/include/cddlib/"* "${GEOMETRIC_HOME_BIN}/include"
export PATH=$CDD_PATH
export CDD_PREFIX="${GEOMETRIC_HOME_LIB}"

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

mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d
touch ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
touch ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh

echo -e '#!/bin/sh\n\nif [[ -v OLD_LD_LIBRARY_PATH ]];\nthen\n\texport LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}\nfi\nif [[ -v OLD_LIBRARY_PATH ]];\nthen\n\texport LIBRARY_PATH=${OLD_LIBRARY_PATH}\nfi\nif [[ -v OLD_PATH ]];\nthen\n\texport PATH=${OLD_PATH}\nfi\nif [[ -v OLD_C_INCLUDE_PATH ]];\nthen\n\texport C_INCLUDE_PATH=${OLD_C_INCLUDE_PATH}\nfi\nif [[ -v OLD_CPATH ]];\nthen\n\texport CPATH=${OLD_CPATH}\nfi\nif [[ -v OLD_CPLUS_INCLUDE_PATH ]];\nthen\n\texport CPLUS_INCLUDE_PATH=${OLD_CPLUS_INCLUDE_PATH}\nfi\n\nunset OLD_LD_LIBRARY_PATH\nunset OLD_LIBRARY_PATH\nunset OLD_PATH\nunset OLD_C_INCLUDE_PATH\nunset OLD_CPATH\nunset OLD_CPLUS_INCLUDE_PATH\n' > ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh

echo -e '#!/bin/sh\n\nexport OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}\nexport LD_LIBRARY_PATH='"${GEOMETRIC_HOME_BIN}/lib:${GUROBI_HOME}/lib:"'${LD_LIBRARY_PATH}\n\nexport OLD_LIBRARY_PATH=${LIBRARY_PATH}\nexport LIBRARY_PATH='"${GEOMETRIC_HOME_BIN}/lib:${GUROBI_HOME}/lib:"'${LIBRARY_PATH}\n\nexport OLD_PATH=${PATH}\nexport PATH='"${GEOMETRIC_HOME_BIN}/bin:${GUROBI_HOME}/bin:"'${PATH}\n\nexport OLD_C_INCLUDE_PATH=${C_INCLUDE_PATH}\nexport C_INCLUDE_PATH='"${GEOMETRIC_HOME_BIN}/include:${GUROBI_HOME}/include:"'${C_INCLUDE_PATH}\n\nexport OLD_CPATH=${CPATH}\nexport CPATH='"${GEOMETRIC_HOME_BIN}/include:${GUROBI_HOME}/include:"'${CPATH}\n\nexport OLD_CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}\nexport CPLUS_INCLUDE_PATH='"${GEOMETRIC_HOME_BIN}/include:${GUROBI_HOME}/include:"'${CPLUS_INCLUDE_PATH}' > ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh

git clone https://github.com/eth-sri/ELINA.git
cd ELINA
git checkout 946397108597cb3562fc530bc97d505e3f5babf2
cp ../patch_ELINA.txt ./
git apply patch_ELINA.txt
if test "$has_cuda" -eq 1
then
    ./configure -use-cuda -use-deeppoly -use-gurobi -use-fconv --prefix "${GEOMETRIC_HOME_BIN}"  -cdd-prefix "${CDD_PREFIX}"
else
    ./configure -use-deeppoly -use-gurobi -use-fconv --prefix "${GEOMETRIC_HOME_BIN}"  -cdd-prefix "${CDD_PREFIX}"
fi

make
make install
cd ..

git clone https://github.com/eth-sri/deepg.git
cd deepg
git checkout b59e86c46cbb2966d4066e55cb418f2165afd15d
cp ../patch_deepg.txt ./
git apply patch_deepg.txt
cd code
mkdir build
make shared_object
cp ./build/libgeometric.so ${GEOMETRIC_HOME_BIN}/lib
cd ../..


pip install -r requirements.txt
