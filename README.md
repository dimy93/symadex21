Provably Robust Adversarial Examples
======
This is supplementary code to the ICML 2021 submission of the 'Provably Robust Adversarial Examples' paper.

CIFAR10 and MNIST Pixel Intensity Experiments
------

To download and install the code for pixel intensity CIFAR10 and MNIST experiments execute the following commands:
```
git clone https://github.com/dimy93/symadex21
cd symadex21
./install_server.sh 
source ~/miniconda3/bin/activate
conda activate eran_symadex
```
To run our approach on MNIST convSmall call the ***run\_mnist.sh*** script as follows:
```
cd tf_verify/
./run_mnist.sh
```
To run the baseline on MNIST convSmall call the ***run\_mnist\_baseline.sh*** script as follows:
```
cd tf_verify/
./run_mnist_baseline.sh
```
To run our approach on CIFAR10 convSmall call the ***run\_cifar10.sh*** script as follows:
```
cd tf_verify/
./run_cifar10.sh
```
To run the baseline on CIFAR10 convSmall call the ***run\_cifar10\_baseline.sh*** script as follows:
```
cd tf_verify/
./run_cifar10_baseline.sh
```
To run our approach on MNIST convBig call the ***run\_mnist\_big.sh*** script as follows:
```
cd tf_verify/
./run_mnist_big.sh
```
To run the baseline on MNIST convBig call the ***run\_mnist\_big\_baseline.sh*** script as follows:
```
cd tf_verify/
./run_mnist_big_baseline.sh
```
To run our approach on MNIST 8x200 call the ***run\_mnist\_ffn.sh*** script as follows:
```
cd tf_verify/
./run_mnist_ffn.sh
```
To run the baseline on MNIST 8x200 call the ***run\_mnist\_ffn\_baseline.sh*** script as follows:
```
cd tf_verify/
./run_mnist_ffn_baseline.sh
```
The resulting robust adversarial examples from our approach will appear under ***./tf\_verify/NetworkName\_ImgNum\_class\_AdvClass\_it\_Iteration***. The output of our method will appear under ***./tf\_verify/NetworkName\_ImgNum.txt***.

The resulting robust adversarial examples from the baseline will appear under ***./tf\_verify/NetworkName\_ImgNum\_class\_AdvClass\_baseline\_it\_Iteration***. The output of the baseline experiment will appear under ***./tf\_verify/NetworkName\_baseline\_ImgNum.txt***


CIFAR10 and MNIST Geometric Experiments
------

To download and install the code for pixel intensity CIFAR10 and MNIST experiments execute the following commands:
```
git clone https://github.com/dimy93/symadex21
cd symadex21
git checkout geometric
git clone https://github.com/eth-sri/eran.git
cp * eran/
cd eran/
./install_server.sh 
```
To run the geometric 3D Rotation experiments on MNIST convSmall call the ***3d\_convSmall\_rotation.sh*** script as follows:
```
cd tf_verify/
./3d_convSmall_rotation.sh
```
To run the geometric 3D Rotation experiments on MNIST convBig call the ***3d\_convBig\_rotation.sh*** script as follows:
```
cd tf_verify/
./3d_convBig_rotation.sh
```
To run the geometric 3D Rotation experiments on CIFAR10 convSmall call the ***3d\_cifar\_rotation.sh*** script as follows:
```
cd tf_verify/
./3d_cifar_rotation.sh
```
To run the geometric 3D Translation experiments on MNIST convSmall call the ***3d\_convSmall\_translation.sh*** script as follows:
```
cd tf_verify/
./3d_convSmall_translation.sh
```
To run the geometric 3D Translation experiments on MNIST convBig call the ***3d\_convBig\_translation.sh*** script as follows:
```
cd tf_verify/
./3d_convBig_translation.sh
```
To run the geometric 3D Translation experiments on CIFAR10 convSmall call the ***3d\_cifar\_translation.sh*** script as follows:
```
cd tf_verify/
./3d_cifar_translation.sh
```
To run the geometric 4D Rotation experiments on MNIST convSmall call the ***3d\_convSmall\_translation.sh*** script as follows:
```
cd tf_verify/
./4d_convSmall_rotation.sh
```
To run the geometric 4D Rotation experiments on MNIST convBig call the ***3d\_convBig\_translation.sh*** script as follows:
```
cd tf_verify/
./4d_convBig_rotation.sh
```
To run the geometric 4D Rotation experiments on CIFAR10 convSmall call the ***3d\_cifar\_translation.sh*** script as follows:
```
cd tf_verify/
./4d_cifar_rotation.sh
```
The experiments' output will appear under ***tf\_verify/TransformationName\_NetworkName\_0\_100.txt***.


