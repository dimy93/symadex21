from PIL import Image
import numpy as np
import png

import os, argparse
import numpy as np
import dill
import tensorflow as tf
from read_net_file import read_tensorflow_net

sess = tf.Session()
parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Name of the direcotry of the experiment result")
parser.add_argument("netname", help="Network")
args = parser.parse_args()
f = os.path.join(args.dir, "npdata.npz")
f = np.load( f, allow_pickle=True)

mean, std = f['norm_args']
dataset = f['dataset']
lb = f['shrink_lb']
ub = f['shrink_ub']
is_conv = f['is_conv'].item()
img = f['image']
norm = f['norm_func']
denorm = f['denorm_func']

filename, file_extension = os.path.splitext(args.netname)
is_trained_with_pytorch = file_extension==".pyt"

model, is_conv, means, stds, layers = read_tensorflow_net(args.netname, len(lb), is_trained_with_pytorch)

tf_out = tf.get_default_graph().get_tensor_by_name( model.name )
tf_in = tf.get_default_graph().get_tensor_by_name( 'x:0' )

label = np.argmax( sess.run( tf_out, feed_dict={tf_in : img} ) )
target = np.argmax( sess.run( tf_out, feed_dict={tf_in : ub} ) )

from time import sleep
sleep(0.05)

inside = np.logical_and( lb <= f['data'] , ub >= f['data'] )
inside = np.all( inside, axis=1 )
print( "\n\n\n\n\n\n\n\n\nInside:", np.sum( inside ) ) 
norm = dill.loads( norm )
denorm = dill.loads( denorm )
norm( img, dataset, mean, std, is_conv )

center = ( lb + ub ) / 2
'''for i in range( 784 ):
    vec = sess.run( tf_out, feed_dict={tf_in : center} )
    dmid = vec[0,target] - vec[0,label]
    center[i] = ub[i]
    vec = sess.run( tf_out, feed_dict={tf_in : center} )
    dub = vec[0,target] - vec[0,label]
    center[i] = lb[i]
    vec = sess.run( tf_out, feed_dict={tf_in : center} )
    dlb = vec[0,target] - vec[0,label]
    idx = np.argmax( [dmid, dlb, dub] )
    center[i] = [(ub[i] + lb[i])/2, lb[i], ub[i]][idx]'''

denorm( center, dataset, mean, std, is_conv ) 
center *= 255.0
center = center.astype(dtype=np.uint8)
center = center / 255.0
norm( center, dataset, mean, std, is_conv )
target_round = np.argmax( sess.run( tf_out, feed_dict={tf_in : center} ) )

denorm( center, dataset, mean, std, is_conv )
denorm( lb, dataset, mean, std, is_conv )
denorm( ub, dataset, mean, std, is_conv )

if dataset == 'mnist':
    lb = lb.reshape( 28, 28 )
    ub = ub.reshape( 28, 28 )
    center = center.reshape( 28, 28 )
    img = img.reshape( 28, 28 )
elif dataset == 'cifar10':
    lb = lb.reshape( 32, 32, 3 )
    ub = ub.reshape( 32, 32, 3 )
    center = center.reshape( 32, 32, 3 )
    img = img.reshape( 32, 32, 3 )
else:
    assert False
quality = 2
png.from_array((center*255.0).astype(dtype=np.uint8), 'L').save(os.path.join(args.dir, "center.png"))

im1 = Image.open( os.path.join(args.dir, "center.png") )
im1.save(os.path.join(args.dir, "center.jpg"),"JPEG", quality=quality, subsampling=0)
im1 = Image.open( os.path.join(args.dir, "center.jpg") )
im1 = np.array(im1)/255.0
im1 = im1.reshape(-1)
norm( im1, dataset, mean, std, is_conv )
target_jpeg = np.argmax( sess.run( tf_out, feed_dict={tf_in : im1} ) )

print( 'Label:', label, 'Target:', target, 'Target_round:', target_round, 'Target_jpeg', target_jpeg ) 

rounds = 0
jpegs = 0
for i, pt in enumerate( f['data'] ):
    denorm( pt, dataset, mean, std, is_conv ) 
    pt *= 255.0
    pt = pt.astype(dtype=np.uint8)
    pt = pt / 255.0
    norm( pt, dataset, mean, std, is_conv )
    if np.argmax( sess.run( tf_out, feed_dict={tf_in : pt} ) ) == target:
        rounds += 1
    denorm( pt, dataset, mean, std, is_conv )
    
    if dataset == 'mnist':
        pt = pt.reshape( 28, 28 )
    elif dataset == 'cifar10':
        pt = pt.reshape( 32, 32, 3 )

    png.from_array((pt*255.0).astype(dtype=np.uint8), 'L').save(os.path.join(args.dir, "%d.png" % i))
    im1 = Image.open( os.path.join(args.dir, "%d.png" % i) )
    im1.save(os.path.join(args.dir, "%d.jpg" % i),"JPEG", quality=quality, subsampling=0)
    im1 = Image.open( os.path.join(args.dir, "%d.jpg" % i) )
    im1 = np.array(im1)/255.0
    im1 = im1.reshape(-1)
    if np.argmax( sess.run( tf_out, feed_dict={tf_in : im1} ) ) == target:
        jpegs += 1
    if i % 100 == 0:
        print( 'It:', i+1, 'Rounds:', rounds, 'Jpegs', jpegs )

print( 'Rounds:', rounds, 'Jpegs', jpegs )
