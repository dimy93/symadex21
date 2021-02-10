import os, argparse
import numpy as np
import dill
parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Name of the direcotry of the experiment result")
args = parser.parse_args()
f = os.path.join(args.dir, "npdata.npz")
f = np.load( f, allow_pickle=True)
lb = f['shrink_lb']
ub = f['shrink_ub']
img = f['image']

print( 'Vizualizing ', args.dir )

mean, std = f['norm_args']
dataset = f['dataset'].item()
is_conv = f['is_conv'].item()
norm = f['norm_func']
denorm = f['denorm_func']

norm = dill.loads( norm )
denorm = dill.loads( denorm )
denorm( lb, dataset, mean, std, is_conv ) 
denorm( ub, dataset, mean, std, is_conv )

if dataset == 'mnist':
    lb = lb.reshape( 28, 28 )
    ub = ub.reshape( 28, 28 )
    img = img.reshape( 28, 28 )
elif dataset == 'cifar10':
    lb = lb.reshape( 32, 32, 3 )
    ub = ub.reshape( 32, 32, 3 )
    img = img.reshape( 32, 32, 3 )
else:
    assert False



import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

lb = ( lb * 255 ).astype( np.uint8 )
ub = ( ub * 255 ).astype( np.uint8 )
img = ( img * 255 ).astype( np.uint8 )

if dataset == 'mnist':
    imt = (ub - lb) + 1
else:
    imt = np.prod( (ub - lb) + 1, axis=2 )
fig = plt.figure(figsize=(13, 6))
ax = plt.gca()
im1 = ax.imshow(imt)
ax.axis('off')

divider = make_axes_locatable(ax)
cax = divider.append_axes('left', size='100%', pad='4%')
im2 = cax.imshow(img, cmap='gray')
cax.axis('off')

cax = divider.append_axes('right', size='6%', pad='4%')
fig.colorbar(im1, cax=cax, orientation='vertical');

outpath = os.path.join(args.dir, "img.eps")
plt.savefig(outpath, format='eps', dpi=300)
