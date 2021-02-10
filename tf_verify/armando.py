import tensorflow as tf
import numpy as np
import cdd
import os
import time
from gurobipy import *
from tensorflow.contrib import graph_editor as ge
from deepzono_milp import create_model
from fractions import Fraction as frac

def create_hact_region_graph( layers ):
    relu = []
    for i,l in enumerate( layers ): 
        if 'Relu' in l.name:
            assert 'Bias' in layers[i-1].name
            relu.append( (l, layers[i-1]) ) 

    tf_out = layers[-1]
    tf_in = layers[0]
    act_pls = []
    for i in range(len(relu)):
        l = relu[i][0]
        a = relu[i][1]
        tf_act_in = tf.placeholder( shape=(l.shape), dtype=tf.float64 )
        act_pls.append( tf_act_in )
        tf_cond = tf.cast( tf_act_in > 0, tf.float64 ) 
        tf_val = tf_cond * a
        relus = relu[i+1:]
        relus = np.array( list(zip( *relus )) ).T.reshape(-1)
        relus = relus.tolist()+[tf_out]
        relus = ge.graph_replace(relus+[tf_out], {l: tf_val})
        tf_out = relus[-1]
        relus = relus[:-1]
        relus = list( zip( relus[0::2], relus[1::2] ) )
        relu[i+1:] = relus
        relu[i] = (tf_val, a)

    layers_new = np.array( list(zip( *relu )) ).T[:,[1,0]].reshape(-1).tolist()
    layers_new = layers_new[0::2] + [tf_out]
    
    layers_new = [layers[0]] + layers_new
    return layers_new, act_pls

def extract_activations( sess, im, layers ):
    acts = []
    for l in layers:
        if 'Relu' in l.name:
            act = sess.run( l, feed_dict={layers[0]: im} )
            act = ( act > 0 ).astype(np.float64)
            acts.append(act)
    return acts

def create_hact_region( sess, acts, layers_new, act_pls, target ):
    fd = {}
    for i in range(len(act_pls)):
        pl = act_pls[i]
        fd[pl] = acts[i]
    layers_new[-1] = layers_new[-1][:,:] - layers_new[-1][:,target:target+1]
    in_shape = layers_new[0].shape
    np_in = np.zeros( shape=in_shape )
    fd[layers_new[0]] = np_in
    biases = sess.run( layers_new[1:], feed_dict=fd )
    
    coeffs = []
    for i in range( in_shape[-1] ):
        np_in = np.zeros( shape=in_shape )
        np_in[i] = 1
        fd[layers_new[0]] = np_in

        out_new = sess.run( layers_new[1:], feed_dict=fd )
        coeff = [out_new[i] - biases[i] for i in range(len( out_new ))]
        coeffs.append( coeff )
    Ws = []
    acts.append( np.zeros_like( biases[-1] ) ) 
    for i in range( len( biases ) ):
        biases[i][ acts[i] > 0 ] = -biases[i][ acts[i] > 0 ]
        biases[i] = biases[i][0]
        W = [ coeffs[j][i][0] for j in range(in_shape[-1]) ]
        W = np.array( W ).T
        W[ acts[i][0] > 0, : ] = -W[ acts[i][0] > 0, : ]
        Ws.append( W )
    bs = biases
    return Ws, bs

def simplify_region( Ws, bs ):
    Ws = np.concatenate( Ws, axis=0 )
    bs = np.concatenate( bs, axis=0 )
    model = Model()
    xs = model.addMVar(shape=Ws.shape[1], lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    model.addConstr( Ws@xs <= -bs )
    model.update()
    model.setParam('OutputFlag', 0)
    model.setParam( 'DualReductions', 0 )

    tried_names = []
    while True:
        c = None
        for cs in model.getConstrs():
            if not cs.ConstrName in tried_names:
                c = cs
                break
        if c is None:
            break
        mc = model.copy()
        c = mc.getConstrByName( c.ConstrName )
        le = mc.getRow(c)
        rhs_orig = c.RHS
        if c.Sense == '>':
            c.RHS -= 1
            mc.setObjective( le, GRB.MINIMIZE )
        elif c.Sense == '<':
            c.RHS += 1
            mc.setObjective( le, GRB.MAXIMIZE )
        else:
            assert False
        mc.update()
        mc.optimize()
        if c.Sense == '<':
            if not mc.Status == 2:
                import pdb; pdb.set_trace()
            if mc.objval <= rhs_orig:
                model.remove(model.getConstrByName(c.ConstrName))
                model.update()
            else:
                tried_names.append( c.ConstrName )
        else:
            if mc.objval >= rhs_orig:
                model.remove(model.getConstrByName(c.ConstrName))
                model.update()
            else:
                tried_names.append( c.ConstrName )
        del mc

    #model.printStats()
    c_idx = [ int(c.ConstrName[1:]) for c in model.getConstrs() ]
    r = np.random.uniform( -1,1, size=Ws.shape[1] )
    return model, Ws[c_idx, :], bs[c_idx]

def get_volume( Ws, bs, vect ):
    #bs += np.matmul( Ws, vect ) - 1e-7 
    bs = bs.reshape(-1,1)
    #import pdb; pdb.set_trace()
    W = np.concatenate( (-bs, -Ws), axis=1 )
    chunks = math.ceil(W.shape[0] / 50) 
    with open('hpoly.txt', 'w') as f:
        f.write('H-representation\nbegin\n')
        f.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' rational\n')
        for i in range( chunks ):
            mat = cdd.Matrix( W[i*50:min(W.shape[0], (i+1)*50),:], number_type='fraction')
            mat.rep_type = cdd.RepType.INEQUALITY
            s = mat.__str__()
            s = s[s.find('\n')+1:s.rfind('\n')+1]
            s = s[s.find('\n')+1:]
            s = s[s.find('\n')+1:]
            f.write(s)
        f.write('end\n')
    #import pdb; pdb.set_trace()
    os.remove('vpoly.txt')
    os.system('lrs hpoly.txt vpoly.txt >/dev/null 2>&1')
    with open('vpoly.txt', 'r') as f:
        s = f.read()
        vs = 'vertices='
        m = s.find(vs)
        l = s.find(' ',m)
        vs = int( s[m+len(vs):l] )
        s = s.replace('\n*****', '\n' + str(vs), 1)
        s += 'volume\n'
    with open('vpoly.txt', 'w') as f:
        f.write( s )
    os.remove('volume.txt')
    os.system('lrs vpoly.txt volume.txt >/dev/null 2>&1')
    with open('volume.txt', 'r') as f:
        s = f.read()
        vs = 'Volume='
        m = s.find( vs )
        l = s.find( '\n', m )
        vol = s[m+len(vs):l]
        vol = vol.strip()
        a,b = vol.split('/')
        a = int( a )
        b = int( b )
    return a/b

def get_activations( nn, img, cut_model, layers, y_true, y_tar, specLB, specUB ):
    lay = layers[ 2 :: 2 ]
    out = cut_model.sess.run( lay, feed_dict={ cut_model.tf_input: img } )
    activations = [ l.reshape( -1 ) > 0 for l in out ]
    return activations

def check_region( nn, layers, activations, specLB, specUB, y_tar, boundary_layer=None, boundary_neuron=None ):
    relu_needed = [0]*( nn.numlayer )
    for i in range( nn.numlayer - 1 ):
        relu_needed[i] = 1
    krelu_groups = [ None ]*(nn.numlayer)
    nlb = [ [] ]*(nn.numlayer)
    nub = [ [] ]*(nn.numlayer)
    for i in range( nn.numlayer-1 ):
        nub[i] = activations[i]
    if not boundary_layer is None:
        nub[ boundary_layer ][ boundary_neuron ] = None
    counter, var_list, model = create_model(nn, specLB, specUB, nlb, nub, krelu_groups, nn.numlayer, False, relu_needed, False)
    model.update()
    expr = LinExpr()
    
    nns = 0
    layer_st = []
    layer_en = []
    for l in range( len( layers ) ):
        num_nns = np.prod( layers[l].shape )
        if l > 1 and l % 2 == 0: #RELU
            layer_st.append( nns )
        nns += num_nns
        if l > 1 and l % 2 == 0: #RELU
            layer_en.append( nns )

    model.update()
    model.optimize()
    res = model.SolCount == 1
    model.reset()
    return res, model

def genExplanation( nn, img, cut_model, hact_ten, layers, y_true, y_tar, specLB, specUB, MAX_NUM_REGIONS=100 ):
    st = time.time()
    ori_act = get_activations( nn, img, cut_model, layers, y_true, y_tar, specLB, specUB )
    check, model = check_region( nn, layers, ori_act, specLB, specUB, y_tar )
    assert check

    isDone = False
    region_list = [ori_act] 
    work_list = [ori_act]
    model_list = [(model,ori_act)]
    visited = {tuple( np.concatenate( ori_act, axis=0 ) )}
    import sys
    v_total = 0
    while len(work_list) > 0 and not isDone:
        sys.stdout.flush()
        print('Going into the worklist algorithm.', 1)
        rrs = work_list[0]
        work_list = work_list[1:]
        region_time = time.time()
        # Flip one constraint at a time
        for (idx, rr) in enumerate(rrs):
            if isDone:
                break
            for idx1 in range(rr.shape[0]):
                rrs1 = [ r.copy() for r in rrs ]
                rrs1[ idx ][ idx1 ] = not rrs1[ idx ][ idx1 ]
                rrs1_key = tuple( np.concatenate( rrs1, axis=0 ) )
                if rrs1_key not in visited:
                    bound_time = time.time()
                    visited.add(rrs1_key)
                    check, model = check_region( nn, layers, rrs1, specLB, specUB, y_tar, boundary_layer=idx, boundary_neuron=idx1)
                    if check:
                        work_list.append(rrs1)
                        region_list.append(rrs1)
                        model_list.append( (model,rrs1) )
                        acts = [ r.astype(np.float64).reshape(1,-1) for r in rrs1 ]
                        Ws, bs = create_hact_region( cut_model.sess, acts, *hact_ten, y_tar )
                        ls = cut_model.sess.run( layers, feed_dict={layers[0]:img} )
                        model, Ws, bs = simplify_region( Ws, bs )
                        vol = get_volume( Ws, bs )
                        v_total += vol
                        print('Number of regions found: ' + str(len(region_list)), 'Time:', st - time.time() ,'Vol:', float(vol),'/',float(v_total) )
                        if len(region_list) >= MAX_NUM_REGIONS:
                            print('Find enough regions stop', 1)
                            isDone = True
                            break

    en = time.time()
    print( 'Time:', st - en )
    model_list = [ model_list[0], model_list[50], model_list[-1] ]
    v_total = 0
    sys.stdout.flush()
    return region_list
