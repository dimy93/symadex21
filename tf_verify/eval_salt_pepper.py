from PIL import Image
import numpy as np
import png

from pgd_div import create_pgd_graph, pgd
import os, argparse
import numpy as np
import dill
import tensorflow as tf
from read_net_file import read_tensorflow_net

sess = tf.Session()
parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Name of the direcotry of the experiment result")
parser.add_argument("netname", help="Network")
parser.add_argument("epsilon", type=float, help="Epsilon")
args = parser.parse_args()
epsilon = args.epsilon
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
over_lb = f['orig_over_lb']
over_ub = f['orig_over_ub']
norm = dill.loads( norm )
denorm = dill.loads( denorm )

filename, file_extension = os.path.splitext(args.netname)
is_trained_with_pytorch = file_extension==".pyt"

model, is_conv, means, stds, layers = read_tensorflow_net(args.netname, len(lb), is_trained_with_pytorch)

# Create specLB/UB
if dataset=='mnist':
    specLB = np.clip(img - epsilon,0,1)
    specUB = np.clip(img + epsilon,0,1)
elif dataset=='cifar10':
    if(is_trained_with_pytorch):
        specLB = np.clip(img - epsilon,0,1)
        specUB = np.clip(img + epsilon,0,1)
    else:
        specLB = np.clip(img-epsilon,-0.5,0.5)
        specUB = np.clip(img+epsilon,-0.5,0.5)
if is_trained_with_pytorch:
    norm( specLB, dataset, mean, std, is_conv )
    norm( specUB, dataset, mean, std, is_conv )

center = ( lb + ub ) / 2
modifications = []
for i in range( 100 ):
    noise = np.random.normal(0, 1, len(lb)) * (np.minimum( specUB - center, center-specLB)) * 10
    modifications.append( noise )

tf_out = tf.get_default_graph().get_tensor_by_name( model.name )
tf_in = tf.get_default_graph().get_tensor_by_name( 'x:0' )

label = np.argmax( sess.run( tf_out, feed_dict={tf_in : img} ) )
target = np.argmax( sess.run( tf_out, feed_dict={tf_in : ub} ) )

from time import sleep
sleep(0.1)

inside = np.logical_and( lb <= f['data'] , ub >= f['data'] )
inside = np.all( inside, axis=1 )
print( "\n\n\n\n\n\n\n\n\nInside:", np.sum( inside ) ) 
norm( img, dataset, mean, std, is_conv )

'''
pgd_obj = create_pgd_graph( lb, ub, sess, tf_in, tf_out, target )
best = -100
for i in range( 100 ):
    center_new = pgd(sess, lb, ub, *pgd_obj, np.ones( img.shape ) * epsilon * 0.1, np.ones( img.shape ) * epsilon * 0.01, 10, 700)
    vec = sess.run( tf_out, feed_dict={tf_in : center_new} )
    if ( vec[0,target] - vec[0,label] >= best ):
        best = vec[0,target] - vec[0,label]
        center = center_new

'''
center = ( lb + ub ) / 2
'''
for i in range( 784 ):
    vec = sess.run( tf_out, feed_dict={tf_in : center} )
    dmid = vec[0,target] - vec[0,label]
    center[i] = ub[i]
    vec = sess.run( tf_out, feed_dict={tf_in : center} )
    dub = vec[0,target] - vec[0,label]
    center[i] = lb[i]
    vec = sess.run( tf_out, feed_dict={tf_in : center} )
    dlb = vec[0,target] - vec[0,label]
    idx = np.argmax( [dmid, dlb, dub] )
    center[i] = [(ub[i] + lb[i])/2, lb[i], ub[i]][idx]
'''

successful = 0
center_copy = center.copy()
for noise in modifications:
    center = center_copy.copy()
    center += noise
    denorm( center, dataset, mean, std, is_conv )
    center = np.clip( center, 0, 1 )
    norm( center, dataset, mean, std, is_conv )
    '''denorm( center, dataset, mean, std, is_conv ) 
    center *= 255.0
    center = center.astype(dtype=np.uint8)
    center = center / 255.0
    norm( center, dataset, mean, std, is_conv )'''
    if np.argmax( sess.run( tf_out, feed_dict={tf_in : center} ) ) == target:
        successful += 1

print( 'Label:', label, 'Target:', target, 'Successful:', successful ) 

successful = [0] * f['data'].shape[0]
for i, pt in enumerate( f['data'] ):
    pt_copy = pt.copy()
    vec = sess.run( tf_out, feed_dict={tf_in : pt} )
    obj = vec[0,target] - vec[0,label]  
    for j, noise in enumerate( modifications ):
        pt = pt_copy.copy() 
        #pt[idx] = val 
        
        pt += noise
        denorm( pt, dataset, mean, std, is_conv )
        center = np.clip( pt, 0, 1 )
        norm( pt, dataset, mean, std, is_conv )
 
        '''denorm( pt, dataset, mean, std, is_conv ) 
        pt *= 255.0
        pt = pt.astype(dtype=np.uint8)  
        pt = pt / 255.0
        norm( pt, dataset, mean, std, is_conv )'''
        if np.argmax( sess.run( tf_out, feed_dict={tf_in : pt} ) ) == target:
            successful[i] += 1
    print( 'It:', i, 'Avg:', np.sum( successful[0:i+1] ) / (i+1), 'Max:', np.max( successful[0:i+1] ), 'Obj:', obj )
