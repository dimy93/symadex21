import sys
import os
sys.path.insert(0, '../../ELINA/python_interface/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import re, itertools
import gc
import time
import psutil
from multiprocessing import Process, Pipe
import numpy as np
import argparse
import csv
import dill

parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--netname', type=str, default=None, help='the network name, the extension can be only .pyt, .tf and .meta')
parser.add_argument('--domain', type=str, default='DeepPoly',choices=['LP', 'DeepPoly'], help='Domain to use in verification')
parser.add_argument('--dataset', type=str, default=None, help='the dataset, can be either mnist, cifar10, or acasxu')
parser.add_argument('--image_number', type=int, default=None, help='Whether to test a specific image.' )
parser.add_argument('--epsilon', type=float, default=0, help='the epsilon for L_infinity perturbation' )
parser.add_argument('--seed', type=int, default=None, help='Random seed for adex generation.' )
parser.add_argument('--model', type=str, default=None, help='Which model to load, if no model is specified a new one is trained.' )
parser.add_argument('--max_cuts', type=int, default=50, help='Maximum number of cuts before shrinking' )
parser.add_argument('--save_every', type=int, default=10, help='How often to save model' )
parser.add_argument('--nowolf', action='store_true', help='Do not use Frank-Wolfe')
parser.add_argument('--obox_approx', action='store_true', help='Do not calculate full overapprox_box')
parser.add_argument('--obox_init', type=float, help='Initial value for obox')
parser.add_argument('--specnumber', type=int, default=2, help='the property number for the acasxu networks')
parser.add_argument('--target', type=int, default=-1, help='Target' )
parser.add_argument('--baseline', action='store_true', default=False, help='Run baseline')
args = parser.parse_args()

if args.seed:
    seed = args.seed
    np.random.seed(seed)
else:
    seed = None
netname = args.netname
epsilon = args.epsilon
dataset = args.dataset

filename, file_extension = os.path.splitext(netname)
if file_extension not in [ '.pyt', '.tf' ]:
    raise argparse.ArgumentTypeError('only .pyt and .tf formats supported')
is_trained_with_pytorch = file_extension==".pyt"

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

def normalize(image, dataset, means, stds, is_conv):
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

def denormalize(image, dataset, means, stds, is_conv):
    if len(means) == len(image):
        for i in range(len(image)):
            image[i] *= stds[i]
            image[i] += means[i]
    elif(dataset=='mnist'):
        for i in range(len(image)):
            image[i] = image[i]*stds[0] + means[0]
    elif(dataset=='mortgage'):
        image[ : ] = image[ : ] * stds
        image[ : ] = image[ : ] + means
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = image[count]*stds[0] + means[0]
            count = count + 1
            tmp[count] = image[count]*stds[1] + means[1]
            count = count + 1
            tmp[count] = image[count]*stds[2] + means[2]
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

def create_pool( seed, netname, dataset, img, model ):
    ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
    conns = []
    procs = []
    parent_pid = os.getpid()
    for cpu in range( ncpus ):
        parent_conn, child_conn = Pipe()
        conns.append( parent_conn )
        threadseed = None
        if not seed is None:
            threadseed = seed + cpu
        p = Process(target=thread, args=( threadseed, netname, dataset, img, child_conn, cpu, parent_pid, model ))
        p.start()
        procs.append( p )
    return conns, procs

def thread( seed, netname, dataset, im, conn, proc_id, parent_pid, model ):
    import sys 
    try:
        import os
        os.sched_setaffinity(0,[proc_id])
        # Prevent printing from child processes
        sys.stdout = open( str( proc_id ) + '.out', 'w')
        sys.stderr = open( str( proc_id ) + '.err', 'w')
        sys.stdout.flush()

        import tensorflow as tf
        if not seed is None:
            tf.set_random_seed( seed )

        cut_model, is_conv, means, stds, im_norm, pgd_means, pgd_stds, _, zeros, ones = create_tf_model( netname, dataset, im, model )

        from pgd_div import create_pgd_graph, pgd
        from clever_wolf import wolf_attack
        pgd_obj = None
        specLB = specUB = None
        
        while True:
            if not conn.poll( 1 ):
                try:
                    process = psutil.Process(parent_pid)
                    print( 'Process status:', process.status() )
                    sys.stdout.flush()
                    continue
                except (psutil.ZombieProcess, psutil.AccessDenied, psutil.NoSuchProcess):
                    print( 'Parent dead' )
                    sys.stdout.flush()
                    return

            req, x0 = conn.recv()
            print( 'Req recieved:', req )
            if req == 'pgd':
                exs = []
                for i in range( x0[ 0 ] ):
                    ex = pgd(cut_model.sess, specLB, specUB, *pgd_obj, *x0[1:])
                    bound = cut_model.sess.run( cut_model.tf_output, feed_dict={cut_model.tf_input: ex} )
                    if np.argmax( bound ) == cut_model.y_tar:
                        boundidx = np.argpartition(bound[0], -2)[-2:] 
                        bound = np.abs( bound[:,boundidx[0]] - bound[:,boundidx[1]] )
                        exs.append( (ex, bound) )
                print( 'pgd finished' ) 
                conn.send( exs )
            elif req == 'kill':
                break
            elif req == 'reset_model':
                specLB, specUB = x0
                cut_model.reset_model( *x0 )
                conn.send( 'done' )
            elif req == 'change_target':
                cut_model.update_target( *x0 )
                pgd_obj = create_pgd_graph( specLB, specUB, cut_model.sess, cut_model.tf_input, cut_model.tf_output, cut_model.y_tar )
                conn.send( 'done' )
            elif req == 'update_model':
                cut_model.update_bounds( *x0 )
                conn.send( 'done' )
            elif req == 'add_hyper':
                cut_model.add_hyperplane( *x0 )
                conn.send( 'done' )
            elif req.startswith('neg_'):
                req = req[ 4: ]
                exs = []
                for init in x0:
                    ex = wolf_attack( cut_model.model, cut_model.xs, init, cut_model.tf_out_neg, cut_model.tf_grad_positive, cut_model.stopping_crit_negative, cut_model.tf_input, cut_model.sess, req )
                    exs.append( ex )
                print( 'wolf finished' )
                conn.send( exs )
            elif req.startswith('pos_'):
                req = req[ 4: ]
                exs = []
                for init in x0:
                    ex = wolf_attack( cut_model.model, cut_model.xs, init, cut_model.tf_out_pos, cut_model.tf_grad_negative, cut_model.stopping_crit_positive, cut_model.tf_input, cut_model.sess, req )
                    exs.append( ex )
                print( 'wolf finished' )
                conn.send( exs )
            else:
                assert False, 'Bad cmd'
            print( 'Done' )
            sys.stdout.flush()

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stdout)
        print( e )
        sys.stdout.flush()

def map( data, op, print_every=100 ):
    data_size = len( data )
    mapping = {}
    for conn in conns:
        mapping[ conn ] = None
    sample_id = 0
    examples = [ None ] * data_size
    examples_gen = 0
    st = time.time()
    j = 0 
    while True:
        for conn in conns:
            if not mapping[ conn ] is None:
                status = conn.poll()
                if status:
                    examples_gen += 1
                    examples[ mapping[ conn ] ] = conn.recv()
                    mapping[ conn ] = None
            elif sample_id < data_size:
                mapping[ conn ] = sample_id
                conn.send( ( op, data[ sample_id ] ) )
                sample_id += 1
        if examples_gen == data_size:
            break
        else:
            if j * print_every < examples_gen:
                print( examples_gen, '/', data_size )
                j = int( examples_gen / print_every ) + 1
            time.sleep( 0.1 )
    end = time.time()
    print( end - st, 'sec' )
    return examples

def create_tf_model( netname, dataset, im, model_name ):
    import tensorflow as tf
    from read_net_file import read_tensorflow_net
    from clever_wolf import CutModel
    sess = tf.Session()
    filename, file_extension = os.path.splitext(netname)
    is_trained_with_pytorch = file_extension==".pyt"

    if(dataset=='mnist'):
        num_pixels = 784
    elif (dataset=='cifar10'):
        num_pixels = 3072
    elif(dataset=='acasxu'):
        num_pixels = 5
    elif(dataset=='mortgage'):
        num_pixels = 172
    model, is_conv, means, stds, layers = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch)
    pixel_size = np.array( [ 1.0 / 256.0 ] * num_pixels )
    pgd_means = np.zeros( ( num_pixels, 1 ) ) 
    pgd_stds = np.ones( ( num_pixels, 1 ) ) 

    zeros = np.zeros((num_pixels))
    ones = np.ones((num_pixels))
    if is_trained_with_pytorch:
        normalize( zeros, dataset, means, stds, is_conv )
        normalize( ones, dataset, means, stds, is_conv )

    if is_trained_with_pytorch:
        im_copy = np.copy( im )
        normalize( im_copy, dataset, means, stds, is_conv )
        if dataset == 'mnist':
            pgd_means[ : ] = means[ 0 ]
            pgd_stds[ : ] = stds[ 0 ]
            pixel_size = pixel_size / stds[ 0 ]
        elif dataset == 'cifar10': 
            if is_conv:
                count = 0 
                for i in range( 1024 ):
                    pixel_size[ count ] = pixel_size[ count ] / stds[ 0 ]
                    pgd_means[ count ] = means[ 0 ]
                    pgd_stds[ count ] = stds[ 0 ]
                    count = count + 1
                    pixel_size[ count ] = pixel_size[ count ] / stds[ 1 ]
                    pgd_means[ count ] = means[ 1 ]
                    pgd_stds[ count ] = stds[ 1 ]
                    count = count + 1
                    pixel_size[ count ] = pixel_size[ count ] / stds[ 2 ]
                    pgd_means[ count ] = means[ 2 ]
                    pgd_stds[ count ] = stds[ 2 ]
                    count = count + 1
            else:
                for i in range( 1024 ):
                    pixel_size[ i ] = pixel_size[ i ] / stds[ 0 ]
                    pgd_means[ i ] = means[ 0 ]
                    pgd_stds[ i ] = stds[ 0 ]
                    pixel_size[ i + 1024 ] = pixel_size[ i + 1024 ] / stds[ 1 ]
                    pgd_means[ i + 1024 ] = means[ 1 ]
                    pgd_stds[ i +1024 ] = stds[ 1 ]
                    pixel_size[ i + 2048 ] = pixel_size[ i + 2048 ] / stds[ 2 ]
                    pgd_means[ i + 2048 ] = means[ 2 ]
                    pgd_stds[ i + 2048 ] = stds[ 2 ]
        elif dataset == 'mortgage' or dataset == 'acasxu':
            pgd_means[ : , 0 ] = means
            pgd_stds[ : , 0 ] = stds
            pixel_size =  np.array( [ 1.0 ] * num_pixels ) / stds 
        else:
            # TODO Hack - works only on MNIST and CIFAR10 and mortgage and ACAS Xu
            assert False
    else:
        assert dataset == 'mnist'
        im_copy = np.copy( im )

    print( 'Model created' )
    tf_out = tf.get_default_graph().get_tensor_by_name( model.name )
    tf_in = tf.get_default_graph().get_tensor_by_name( 'x:0' )
    print( 'Tensors created' )

    out = sess.run( tf_out, feed_dict={ tf_in: im_copy } )
    print( 'Tf out computed' )
    if model_name is None:
        cut_model = CutModel( sess, tf_in, tf_out, np.argmax( out ), pixel_size )
    else:
        cut_model = CutModel.load( model_name, sess, tf_in, tf_out, np.argmax( out ) )
    print( 'Cut model created' )
    return cut_model, is_conv, means, stds, im_copy, pgd_means, pgd_stds, layers, zeros, ones

def generate_initial_region_PGD( num_samples, eps_init, eps_pgd, init_it, pgd_it ):
    arrs = map( [ ( 100, eps_init, eps_pgd, init_it, pgd_it ) ] * int( num_samples / 100 ), 'pgd', print_every=5 )
    examples = []
    bounds = []
    for arr in arrs:
        examples += [ x[0] for x in filter( lambda x: not x is None, arr ) ]
        bounds += [ x[1] for x in filter( lambda x: not x is None, arr ) ]
    examples = np.array( examples )
    bounds = np.array( bounds )
    print( examples.shape[ 0 ], '/', num_samples )
    
    if examples.shape[ 0 ] == 0:
        lb = None
        ub = None
    else:
        lb = np.min( examples, axis=0 )
        ub = np.max( examples, axis=0 )
    return ( examples, bounds, lb, ub )

def sample_wolf_attacks( inits, step_choice ):
    arrs = map( np.array_split( inits, int( inits.shape[ 0 ] / 10 ) ), step_choice )
    examples = []
    for arr in arrs:
        examples += arr
    attacks = []
    num_binary = 0
    k_sum = 0
    for attack, succ, binary, k in examples:
        if succ:
            k_sum += k
            if binary:
                num_binary += 1
            attacks.append( attack ) 
    attacks = np.array( attacks )
    avg_it = k_sum
    if not attacks.shape[ 0 ] == 0:
        avg_it /= attacks.shape[ 0 ]
    print( attacks.shape[ 0 ], '(', num_binary ,')/', inits.shape[ 0 ], 'Avg it:', avg_it  )

    return attacks

def update_pool_barrier( cmd, params ):
    nthreads = len( conns )
    results = [ False ] * nthreads
    results = np.array( results ) 
    for conn in conns:
        conn.send( ( cmd, params ) )
    
    # Barrier
    while True:
        for i in range( nthreads ):
            msg = conns[ i ].poll()
            if msg:
                assert conns[ i ].recv() == 'done'
                results[ i ] = True
        if np.all( results ):
            break
        time.sleep( 0.1 )
        
def add_hyperplane_pool( params ):
    update_pool_barrier( 'add_hyper', params )

def reset_pool( params ):
    update_pool_barrier( 'reset_model', params )

def update_pool( params ):
    update_pool_barrier( 'update_model', params )

def update_target_pool( params ):
    update_pool_barrier( 'change_target', params )

def lp_cut( cut_model, nn, domain, y_tar, lp_ver_output=None, complete=False ):
    
    if lp_ver_output is None:
        output = cut_model.lp_verification( nn, domain,  y_tar, complete=complete )
    else:
        output = lp_ver_output
    if isinstance( output, bool ):
        return True, None

    if domain == 'DeepPoly':
        eq, example, attack_class, bound, hps = output
        print( 'Network output:', cut_model.eval_network( example ), 'LP output:', bound )
        example = ( eq, example )
    else:
        assert False

    ws = np.concatenate( [-hp[0].T for hp in hps], axis=0 )
    bs = np.concatenate( [-hp[1] for hp in hps], axis=0 )
    
    '''
    if cut_model.it_cut < 6 and ws.shape[0] > 1000:
        idx = np.random.randint( ws.shape[0] - hps[-1][0].shape[1], size=1000)
        idx = idx.tolist() + list( range( ws.shape[0] - hps[-1][0].shape[1], ws.shape[0] ) )
        ws = ws[idx]
        bs = bs[idx]
    '''

    hyper = ( ws, bs )

    cut_model.add_hyperplane( *hyper )
    add_hyperplane_pool( hyper )

    print( 'Hyperplanes:', cut_model.W.shape )
    return False, hyper

def clever_wolf( nn, cut_model, y_true, y_tar, specLB, specUB, domain, args, pgd_args1, pgd_args2 ):
    try:
        delattr( cut_model, 'shrink_box_best' )
    except:
        pass
    clever_start_time = time.time()
    if cut_model.y_tar == None:
        reset_pool( ( specLB, specUB ) )                                                                                                                                
        update_target_pool( ( y_true, y_tar ) ) 
        data, _, lb, ub = generate_initial_region_PGD( 250, *pgd_args1 )
        cut_model.update_target( y_true, y_tar )
        cut_model.reset_model( specLB, specUB )
        succ_attacks = data.shape[ 0 ]
        all_attacks = 250
        if not args.nowolf:
            samples = cut_model.sample_poly_under( 250 )                                                                                                                   
            pos_ex = sample_wolf_attacks( samples, 'pos_brent' )
            succ_attacks += pos_ex.shape[ 0 ]
            all_attacks += 250

        data2, _, lb2, ub2 = generate_initial_region_PGD( 250, *pgd_args2 )
        succ_attacks += data2.shape[ 0 ]
        all_attacks += 250
        print( 'Target', y_tar, succ_attacks, '/', all_attacks )
        if succ_attacks > 0:
            reset_pool( ( specLB, specUB ) )
            update_target_pool( ( y_true, y_tar ) )
            cut_model.update_target( y_true, y_tar )
            cut_model.reset_model( specLB, specUB )
                    
            data, bounds, lb, ub = generate_initial_region_PGD( 2500, *pgd_args1 )
            data2, bounds2, lb2, ub2 = generate_initial_region_PGD( 2500, *pgd_args2 )
            if lb is None and lb2 is None:
                data = np.zeros( ( 0, cut_model.input_size ) ) 
                bounds = np.zeros( 0 ) 
            elif lb is None:
                data = data2
                bounds = bounds2
            elif not lb2 is None:
                data = np.concatenate( ( data, data2 ) )
                bounds = np.concatenate( ( bounds, bounds2 ) )
 
            if not args.nowolf:
                samples = cut_model.sample_poly_under( 5000 )
                pos_ex = sample_wolf_attacks( samples, 'pos_brent' )
                if not pos_ex.shape[ 0 ] == 0:
                    data = np.concatenate( ( data, pos_ex ) )

            if data.shape[0] == 0:
                return False

            lb = np.min( data, axis=0 )
            ub = np.max( data, axis=0 )

            cut_model.update_bounds( lb, ub )
            cut_model.set_data( data )
            cut_model.bounds = bounds
            update_pool( ( lb, ub ) )
        else:
            return False
        s = time.time()
        config.dyn_krelu = False
        config.use_3relu = False
        config.use_2relu = False
        config.numproc_krelu = 24
        
        norm_ser = dill.dumps(normalize)
        denorm_ser = dill.dumps(denormalize)
        cut_model.norm_func = norm_ser
        cut_model.denorm_func = denorm_ser
        cut_model.dataset = dataset
        cut_model.is_conv = is_conv
        cut_model.norm_args = ( means, stds )
        lb, ub = cut_model.overapprox_box()
        cut_model.orig_over_lb = lb
        cut_model.orig_over_ub = ub

        eran = ERAN(cut_model.tf_output, is_onnx=False)
        label,nn,nlb,nub = eran.analyze_box(lb, ub, 'deeppoly', 1, 1, True)
        print( 'Label:', label, 'Time:', time.time() - s, 's' )
        if label == -1:
            cut_model.nlb = [ np.array( lb ) for lb in nlb ]
            cut_model.nub = [ np.array( ub ) for ub in nub ]
            lb, ub = cut_model.overapprox_box()
            cut_model.nlb.insert( 0, lb )
            cut_model.nub.insert( 0, ub )
        else:
            cut_model.shrink_box_best = cut_model.overapprox_box()
            cut_model.save( model_name )
            print( 'Verified, time:', int( time.time() - clever_start_time ) )
            print_vol( cut_model )
            return True   
    print( 'Init model' )
    if args.obox_approx:
        cut_model.approx_obox = True
    process = psutil.Process(os.getpid())
    start_lp_sampling = args.nowolf
    method = None
    res = None
    lp_params = ( nn, domain, y_tar )
    wolf_params = [ 1000 ]

    shrink = cut_model.copy()
    if 'shrink_box_best' in dir( shrink ):
        del shrink.shrink_box_best

    print_vol( cut_model )
    sys.stdout.flush()
    lbs, ubs, lb_max, ub_max = shrink.create_underapprox_box_lp( y_tar, shrink_to=args.obox_init, baseline=args.baseline )
    if args.baseline:
        print( '\nBaseline, Time:', int( time.time() - clever_start_time ), 'sec,', 'Target:', y_tar, '\n')
    if not lb_max is None:
        data = cut_model.data
        cut_model.obox = None
        cut_model.ubox = None
        cut_model.update_target( y_true, y_tar )
        cut_model.reset_model( lb_max, ub_max )
        cut_model.update_bounds( lb_max, ub_max )
        cut_model.nlb = [ np.array( lb ) for lb in nlb ]
        cut_model.nub = [ np.array( ub ) for ub in nub ]
        cut_model.nlb.insert( 0, lb_max )
        cut_model.nub.insert( 0, ub_max )
        cut_model.data = data
    if not lbs is None:
        cut_model.shrink_box_best = ( np.array(lbs), np.array(ubs) )
    else:
        print('Failed to converge')
        return False

    print_vol( cut_model )
    if args.baseline:
        cut_model.save( model_name, baseline=True )
        return True
    cut_model.p_c = 0.9
    for cut in range( 50 ):

        cut_model.it_cut = cut
        cut_model.p_c *= 0.95
        if cut >= args.max_cuts:
            cut_model.p_c = 0 
        sys.stdout.flush()
        
        '''
        if dataset == 'mortgage':
            project = np.array( [ False ] * cut_model.input_size )
            project[ bounds_keys ] = True
            W, lb, ub = cut_model.denorm_W( project, means, stds )
            lb_sh, ub_sh = cut_model.shrink_box_best[0].copy(), cut_model.shrink_box_best[1].copy()
            denormalize( lb_sh, dataset, means, stds, False )
            denormalize( ub_sh, dataset, means, stds, False )
            lb_sh = lb_sh[bounds_keys] 
            ub_sh = ub_sh[bounds_keys] 
            center = None
            pos_ex = None
            neg_ex = None
            v2 = draw2d_region( W, lb, ub, lb_sh, ub_sh, model_name + '_reg' + str(cut) + '.png', ( bounds_lb, bounds_ub ), center=center, pos_ex=pos_ex, neg_ex=neg_ex, draw_hp=W.shape[0] - 1 )
            print( 'Volume:', v2 )
        '''

        gc.collect()
        print( '\nCut:', cut, ', Time:', int( time.time() - clever_start_time ), 'sec,', 'Target:', y_tar,',Memory:', process.memory_info().rss / (1024*1024),'\n')
        
        verified, _ = lp_cut( cut_model, *lp_params )
           
        if verified:
            if dataset == 'mortgage':
                project = np.array( [ False ] * cut_model.input_size )
                project[ bounds_keys ] = True
                W, lb, ub = cut_model.denorm_W( project, means, stds )
                
                lb_sh, ub_sh = cut_model.shrink_box_best[0].copy(), cut_model.shrink_box_best[1].copy()
                denormalize( lb_sh, dataset, means, stds, False )
                denormalize( ub_sh, dataset, means, stds, False )
                lb_sh = lb_sh[bounds_keys] 
                ub_sh = ub_sh[bounds_keys] 
 
                center = None
                pos_ex = None
                neg_ex = None
                v2 = draw2d_region( W, lb, ub, lb_sh, ub_sh, model_name + '_reg' + str(cut+1) + '.png', ( bounds_lb, bounds_ub ), center=center, pos_ex=pos_ex, neg_ex=neg_ex, draw_hp=W.shape[0] - 1 )
                print( 'Volume:', v2 )
            cut_model.save( model_name )
            print( 'Verified, time:', int( time.time() - clever_start_time ) )
            print_vol( cut_model )
            return True
    print('Failed to converge')
    return False

def destroy_pool():
    for conn in conns:
        conn.send( ( 'kill', None ) )
    for proc in procs:
        proc.join()

def draw2d_region( W, lb, ub, lb_sh, ub_sh, name, bounds, pos_ex=None, neg_ex=None, center=None, draw_hp=-1 ):
    lbs = np.concatenate( ( -np.eye( lb.shape[ 0 ] ), lb[:,np.newaxis] ), axis=1 ) 
    ubs = np.concatenate( ( np.eye( ub.shape[ 0 ] ), -ub[:,np.newaxis] ), axis=1 )
    W_full = np.concatenate( ( lbs, ubs ), axis=0 )
    W_full = np.concatenate( ( W_full, W ), axis=0 )
    verts = []
    for i in range( W_full.shape[0]):
        for j in range( i + 1, W_full.shape[0]):
            if j == i + lbs.shape[0] and i < lbs.shape[0]:
                continue
            idx = np.array( [i,j] )
            try:
                x = np.linalg.solve( W_full[idx, :-1], -W_full[idx, -1] )
            except np.linalg.LinAlgError as e:
                continue
            m = np.matmul( W_full, x.tolist() + [1] )
            if np.any( m > 1e-6 ):
                continue
            #print( i ,j , m, x )
            verts.append( x )
    verts = np.array( verts )
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull, convex_hull_plot_2d
    try:
        hull = ConvexHull(verts)
        hs = hull.simplices
        vol = hull.volume
    except:

        dist_0 = np.max( verts[ :, 0 ] - np.average( verts[ :, 0 ] ) ) 
        dist_1 = np.max( verts[ :, 1 ] - np.average( verts[ :, 1 ] ) ) 
        if dist_0 < dist_1:
            hs = []
            hs.append( ( np.argmin( verts[ :, 1 ] ), np.argmax( verts[ :, 1 ] ) ) )
        else:
            hs = []
            hs.append( ( np.argmin( verts[ :, 0 ] ), np.argmax( verts[ :, 0 ] ) ) )
        vol = 0
    plt.figure()
    plt.plot(verts[:,0], verts[:,1], 'ko')
    plt.plot([lb_sh[0],lb_sh[0], ub_sh[0], ub_sh[0]], [lb_sh[1],ub_sh[1], ub_sh[1], lb_sh[1]], 'bo')
    if not pos_ex is None:
        plt.plot(pos_ex[:, 0], pos_ex[:,1], 'g+')
    if not neg_ex is None:
        plt.plot(neg_ex[:, 0], neg_ex[:,1], color='red', marker='$-$', linestyle=' ')
    if not center is None:
        plt.plot(center[0], center[1], 'bx', markersize=12)
    axes = plt.gca()
    bounds_lb, bounds_ub = bounds
    margin = ( np.array( bounds_ub ) - bounds_lb ) / 20.0
    axes.set_xlim([ bounds_lb[ 0 ] - margin[ 0 ], bounds_ub[ 0 ] + margin[ 0 ] ])
    axes.set_ylim([ bounds_lb[ 1 ] - margin[ 1 ], bounds_ub[ 1 ] + margin[ 1 ] ])
    for simplex in hs:
        plt.plot(verts[simplex, 0], verts[simplex, 1], 'k-')
    if draw_hp > -1:
        l = W[ draw_hp, : ]
        y_min = ( l[ 0 ] * ( bounds_lb[ 0 ] - margin[ 0 ] ) + l[ 2 ] ) / -l[ 1 ]
        y_max = ( l[ 0 ] * ( bounds_ub[ 0 ] + margin[ 0 ] ) + l[ 2 ] ) / -l[ 1 ]
        plt.plot([bounds_lb[ 0 ] - margin[ 0 ], bounds_ub[ 0 ] + margin[ 0 ]], [y_min,y_max], 'y--')
    plt.show()
    plt.title( 'Volume:' + str(vol) )
    plt.savefig( name )
    plt.close()
    return vol

if(dataset=='mnist'):
    csvfile = open('../data/mnist_test.csv', 'r')
    tests = csv.reader(csvfile, delimiter=',')
elif(dataset=='cifar10'):
    csvfile = open('../data/cifar10_test.csv', 'r')
    tests = csv.reader(csvfile, delimiter=',')
elif(dataset=='mortgage'):
    csvfile = open('../data/mortgage_test.csv', 'r')
    tests = csv.reader(csvfile, delimiter=',')
elif(dataset=='acasxu'):
    specfile = '../data/acasxu/specs/acasxu_prop' + str(args.specnumber) +'_spec.txt'
    tests = open(specfile, 'r').read()
    tests = parse_input_box(tests)
else:
    assert False

if(dataset != 'acasxu'):
    tests = list( tests )
    test = tests[ args.image_number ]
    corr_label = int( test[ 0 ] )

# Create img
if(dataset=='mnist'):
    image= np.float64(test[1:len(test)])/np.float64(255)
elif(dataset=='mortgage'):
    image= (np.float64(test[1:len(test)]))
elif(dataset=='cifar10'):
    if(is_trained_with_pytorch):
        image= (np.float64(test[1:len(test)])/np.float64(255))
    else:
        image= (np.float64(test[1:len(test)])/np.float64(255)) - 0.5
elif(dataset=='acasxu'):
    image = np.average( np.array( tests )[0],axis=1)
else:
    assert False

img = np.copy(image)

if dataset=='mortgage':
    import json
    f = open( '../data/mortgage/spec' + str( args.image_number ) + '.txt', 'r' ) 
    x = f.readlines()[ 0 ]
    x = '{' + x + '}'
    feat_bounds = json.loads(x)
    feat_bounds = { int(k): feat_bounds[k] for k in feat_bounds }
    spread = np.array( list( feat_bounds.values() ) )
    spread = spread[ : , 1 ] - spread[ : , 0 ]
    margin = 0.05 * spread
    spread = 1.05 * spread
    l = 0
    for k in feat_bounds:
        feat_bounds[k] = ( feat_bounds[ k ][ 0 ] - margin[ l ], feat_bounds[ k ][ 1 ] + margin[ l ] )
        l += 1
    print( feat_bounds )
    bounds_lb = [ feat_bounds[ key ][ 0 ] for key in feat_bounds ]
    bounds_ub = [ feat_bounds[ key ][ 1 ] for key in feat_bounds ]
    bounds_keys = list( feat_bounds.keys() )
    e = np.zeros( image.shape, dtype=np.float32 )
    e[ bounds_keys ] = spread
    clip_max = image.copy() 
    clip_max[ bounds_keys ] = bounds_ub
    clip_max += 1e-6
    clip_min = image.copy()
    clip_min[ bounds_keys ] = bounds_lb
    clip_min -= 1e-6
elif dataset=='acasxu':
    bounds_lb = np.array( tests )[0,:,0]
    e = image - bounds_lb
    bounds_ub = np.array( tests )[0,:,1]
else:
    e = epsilon

conns, procs = create_pool( seed, netname, dataset, img, args.model ) 

os.sched_setaffinity(0,list(range(8)))
cut_model, is_conv, means, stds, img, _, _, layers, _, _ = create_tf_model( netname, dataset, img, args.model )
cut_model.graph_layers = layers


import atexit
atexit.register(destroy_pool)
domain = args.domain
cut_model.create_tf_sampling_net( netname, is_trained_with_pytorch, domain )

'''
i = 0 
prec = 0
for test in tests:
    corr_label = int( test[ 0 ] )
    if(dataset=='mnist'):
        image= np.float64(test[1:len(test)])/np.float64(255)
    else:
        if(is_trained_with_pytorch):
            image= (np.float64(test[1:len(test)])/np.float64(255))
        else:
            image= (np.float64(test[1:len(test)])/np.float64(255)) - 0.5

    img = np.copy(image)
    if is_trained_with_pytorch:
        normalize( img, dataset, means, stds, is_conv )
    label = np.argmax( cut_model.sess.run( cut_model.tf_output, feed_dict={cut_model.tf_input: img } ) )
    i += 1
    if label == corr_label:
        prec += 1 
    if i % 100 == 0:
        print( prec , '/', i )
exit()
'''

from clever_wolf import *
import sys
sys.path.insert(0, '../ELINA/python_interface/')
from eran import ERAN
from config import config
config.dyn_krelu = False
config.numproc_krelu = 1
eran = ERAN(cut_model.tf_output, is_onnx=False)
imgLB = np.copy( img )
imgUB = np.copy( img )
label,nn,nlb,nub = eran.analyze_box(imgLB, imgUB, 'deepzono', 1, 1, True)
assert label == cut_model.y_true

cut_model.orig_image = image.copy()

if dataset == 'acasxu' or dataset == 'mortgage':
    corr_label = cut_model.y_true

# Create specLB/UB
if dataset=='mnist':
    specLB = np.clip(image - epsilon,0,1)
    specUB = np.clip(image + epsilon,0,1)
elif dataset=='mortgage':
    specLB = image.copy()
    specUB = image.copy()
    specLB[ bounds_keys ] = bounds_lb
    specUB[ bounds_keys ] = bounds_ub
elif dataset=='acasxu':
    specLB = image.copy()
    specUB = image.copy()
    specLB = np.clip( image - e, bounds_lb, bounds_ub )
    specUB = np.clip( image + e, bounds_lb, bounds_ub )
elif dataset=='cifar10':
    if(is_trained_with_pytorch):
        specLB = np.clip(image - epsilon,0,1)
        specUB = np.clip(image + epsilon,0,1)
    else:
        specLB = np.clip(image-epsilon,-0.5,0.5)
        specUB = np.clip(image+epsilon,-0.5,0.5)
if is_trained_with_pytorch:
    normalize( specLB, dataset, means, stds, is_conv )
    normalize( specUB, dataset, means, stds, is_conv )
else:
    means = [0.0]
    stds = [1.0]

if not corr_label == label:
    print('Bad classification.')
    exit()

classes = cut_model.tf_output.shape.as_list()[1]
targets = range( classes )
if args.target != -1:
    assert args.target != label
    targets = [args.target]

if 'ConvBig__Point_mnist' in filename:
    list_img = { 7: [4], 12: [4], 15: [3], 18: [5, 8], 20: [4], 25: [8,9], 29: [3, 7], 31: [7], 40: [7], 41: [2], 45: [8], 58: [4], 59: [4,9], 62:[5,7,8], 65: [9], 73: [8], 78: [4], 96: [9] }
elif 'mnist_relu_9_200' in filename:
    list_img = { 6:[8], 7:[3, 7, 8], 15:[3, 8], 24:[9], 26:[9], 31:[3, 4, 7, 8, 9], 33:[0, 2, 6, 8, 9], 36:[3], 40:[3, 9], 41:[3], 45:[3, 8], 46:[3, 8, 9], 52:[3], 53:[3, 6, 8], 62:[3, 4, 5, 8], 66:[2], 73:[3, 7, 8], 77:[3, 7, 8, 9], 78:[8], 92:[4, 8], 96:[3, 4, 5, 8, 9], 98:[0, 2] }
elif 'convSmallRELU__Point' in filename:
    list_img = { 6:[8], 8:[6], 9:[7], 11:[8], 15:[3], 18:[2,5,6,8], 20:[7], 24:[7,9], 33:[0], 38:[1,3], 45:[8], 48:[8], 61:[2], 62:[4,5,8], 65:[5,9], 66:[3], 73:[7,8], 78:[8], 92:[4,7,8], 95:[8], 96:[4]}
elif 'convSmallRELU__cifar10_Point' in filename:
    list_img =  {3:[2, 8],  6:[5], 7:[2],  8:[2, 4, 5, 7], 9:[9], 18:[9], 21:[3], 26:[7], 30:[2, 3, 4, 5, 7], 32:[0, 2, 3, 6, 8], 41:[0, 2, 4], 55:[0], 56:[3, 5], 63:[9], 65:[6], 66:[8], 70:[0, 3, 4, 5, 8], 74:[6], 84:[0, 5], 86:[7], 96:[2, 4], 98:[6], 99:[5] }
else:
    assert False

assert args.image_number in list_img
targets = list_img[ args.image_number ]

if dataset == 'acasxu':
    if args.specnumber == 2:
        targets = [0]
    elif args.specnumber == 8:
        targets = [3]
    else:
        assert False
    assert not corr_label in targets

zs = np.zeros_like(imgLB)
eps = zs + e
if is_trained_with_pytorch:
    normalize( zs, dataset, means, stds, is_conv )
    normalize( eps, dataset, means, stds, is_conv )
eps = eps - zs
pgd_args1 = (eps * 0.1, eps * 0.1, 5, 50)
pgd_args2 = (eps * 0.01, eps * 0.01, 5, 250)


for i in targets:
    if i == label:
        continue
    if not args.model is None:
        if not i == cut_model.y_tar:
            continue
    try:
        model_name = os.path.basename( filename ) + '_' + str( args.image_number ) + '_class_' + str( i )
        clever_wolf( nn, cut_model, corr_label, i, specLB, specUB, domain, args, pgd_args1, pgd_args2 )
        if args.model is None:
            cut_model.y_tar = None
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stdout)
        print(e)
        cut_model.save(model_name+'_error')

