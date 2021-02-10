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
The resulting symbolic adversarial examples will appear under ***./ERAN/tf\_verify/NetworkName\_ImgNum\_class\_AdvClass\_it\_Iteration***.



CIFAR10 and MNIST Geometric Experiments
------

To download and install the code for pixel intensity CIFAR10 and MNIST experiments execute the following commands:
```
git clone https://github.com/dimy93/symadex21
cd symadex21
./install_server.sh 
```
To run the our approach on MNIST convSmall call the ***run\_mnist.sh*** script as follows:
```
cd tf_verify/
./run_mnist.sh
```
To run the our approach on CIFAR10 convSmall call the ***run\_cifar10.sh*** script as follows:
```
cd ERAN/tf_verify/
./run_cifar10.sh
```
To run the our approach on MNIST convBig call the ***run\_mnist\_big.sh*** script as follows:
```
cd ERAN/tf_verify/
./run_mnist_big.sh
```
To run the our approach on MNIST 8x200 call the ***run\_mnist\_ffn.sh*** script as follows:
```
cd ERAN/tf_verify/
./run_mnist_ffn.sh
```
The resulting symbolic adversarial examples will appear under ***./ERAN/tf\_verify/NetworkName\_ImgNum\_class\_AdvClass\_it\_Iteration***.
