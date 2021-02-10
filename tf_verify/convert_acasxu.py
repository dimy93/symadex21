import re, itertools
import onnx
import tensorflow as tf
import numpy as np
from onnx import numpy_helper
import argparse
import onnxruntime.backend as rt

def parse_input_box(text):
    intervals_list = []
    for line in text.split('\n'):
        if line!="":
            interval_strings = re.findall("\[-?\d*\.?\d+, *-?\d*\.?\d+\]", line)
            intervals = []
            for interval in interval_strings:
                interval = interval.replace('[', '')
                interval = interval.replace(']', '')
                [lb,ub] = interval.split(",")
                intervals.append((np.double(lb), np.double(ub)))
            intervals_list.append(intervals)

    # return every combination
    boxes = itertools.product(*intervals_list)
    return list(boxes)

dataset = 'acasxu'
def normalize(image, means, stds, is_conv):
    if len(means) == len(image):
        for i in range(len(image)):
            image[i] -= means[i]
            image[i] /= stds[i]
    elif(dataset=='mnist'):
        for i in range(len(image)):
            image[i] = (image[i] - means[0])/stds[0]
    elif(dataset=='mortgage'):
        image[ : ] = image[ : ] - means
        image[ : ] = image[ : ] / stds
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = (image[count] - means[0])/stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1])/stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2])/stds[2]
            count = count + 1

        if(is_conv):
            for i in range(3072):
                image[i] = tmp[i]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count+1
                image[i+1024] = tmp[count]
                count = count+1
                image[i+2048] = tmp[count]
                count = count+1
    else:
        assert False

parser = argparse.ArgumentParser(description='Convert acasxu network to pyt',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--netname', type=str, default=None, help='the network name, the extension can be only .onnx')
parser.add_argument('--output', type=str, default=None, help='the output network name, the extension can be only .pyt')
parser.add_argument('--spec', type=str, default=None, help='the spec file')
parser.add_argument('--invert', action='store_true', help='Invert last layer')
args = parser.parse_args()
assert args.netname.endswith( '.onnx' )
assert args.output.endswith( '.pyt' )
model = onnx.load(args.netname)
weights = model.graph.initializer
ws = []
bs = []
for i in range(len(weights)):
    if '_Add_B' in weights[i].name:
        en = weights[i].name.index('_Add_B')
        st = weights[i].name.rindex('_', 0, en) + 1
        layer = int( weights[i].name[st:en] )
        assert len( bs ) + 1 == layer
        bs.append( numpy_helper.to_array(weights[i]) )
    elif '_MatMul_W' in weights[i].name:
        en = weights[i].name.index('_MatMul_W')
        st = weights[i].name.rindex('_', 0, en) + 1
        layer = int( weights[i].name[st:en] )
        assert len( ws ) + 1 == layer
        ws.append( numpy_helper.to_array(weights[i]) )
    elif 'input_AvgImg' == weights[i].name:
        assert np.all( numpy_helper.to_array(weights[i]) == 0.0 )
    else:
        print( weights[i].name )
assert len(ws) == len(bs)

means = [19791.091, 0.0, 0.0, 650.0, 600.0]
stds = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]
with open(args.output, 'w') as f:
    f.write('Normalize mean=')
    f.write(str(means))
    f.write(' std=')
    f.write(str(stds) + '\n')
    for i in range(len(ws)):
        if i == len(ws) - 1:
            f.write('Affine\n') 
        else:
            f.write('ReLU\n')
        if args.invert and i == len(ws) - 1:
            f.write(str((-ws[i]).T.tolist()) + '\n')
            f.write(str((-bs[i]).tolist()) + '\n')
        else:
            f.write(str(ws[i].T.tolist()) + '\n')
            f.write(str(bs[i].tolist()) + '\n')

from read_net_file import read_tensorflow_net
spec = open(args.spec, 'r').read()
spec = parse_input_box(spec)
spec_lb = np.array(spec)[0,:,0]
spec_ub = np.array(spec)[0,:,1]


model_tf, is_conv, means_r, stds_r, layers = read_tensorflow_net(args.output, 5, True)

assert np.all( means_r - means == 0.0 )
assert np.all( stds_r - stds == 0.0 )

normalize(spec_lb, means, stds, is_conv)
normalize(spec_ub, means, stds, is_conv)

tf_out = tf.get_default_graph().get_tensor_by_name( model_tf.name )
tf_in = tf.get_default_graph().get_tensor_by_name( 'x:0' )
sess = tf.Session()

out_lb = sess.run( tf_out, feed_dict={tf_in: spec_lb} )
out_ub = sess.run( tf_out, feed_dict={tf_in: spec_ub} )

runnable = rt.prepare(model, 'CPU')
out_lb_orig = runnable.run(spec_lb.reshape(1,1,1,5).astype(np.float32))
out_ub_orig = runnable.run(spec_ub.reshape(1,1,1,5).astype(np.float32))

if args.invert:
    print( 'Inverted' )
    assert np.all( np.abs( out_lb_orig + out_lb ) < 1e-6 ) 
    assert np.all( np.abs( out_ub_orig + out_ub ) < 1e-6 ) 
else:
    assert np.all( np.abs( out_lb_orig - out_lb ) < 1e-6 ) 
    assert np.all( np.abs( out_ub_orig - out_ub ) < 1e-6 ) 
