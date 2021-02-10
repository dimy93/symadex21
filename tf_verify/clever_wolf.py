import os
import time
import errno
import sys
import multiprocessing
from multiprocessing import Process, Manager
from gurobipy import *
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize_scalar
from deepzono_milp import create_model
def brent( x_k, s_k, sess, tf_in, tf_out ):
    def f( y_k, x_k, s_k, sess, tf_inp, tf_out ):
        x_k_plus_one = x_k*( 1 - y_k ) + y_k*s_k
        out = sess.run( tf_out, feed_dict={ tf_in: x_k_plus_one } )
        return -out
    res = minimize_scalar( f, bounds=(0, 1), method='bounded', args=( x_k, s_k, sess, tf_in, tf_out ) )
    return res.x

def binary_search_step( x_k, s_k, max_yk, stopping_crit ):
    overall_yk = 0.0
    last = max_yk
    x_k_plus_1 = x_k*( 1 - ( last + overall_yk ) ) + ( last + overall_yk ) * s_k
    if not stopping_crit( x_k_plus_1 ):
        return max_yk
    while last >= 0.0000001:
        x_k_plus_1 = x_k*( 1 - ( last + overall_yk ) ) + ( last + overall_yk ) * s_k
        while last >= 0.0000001 and stopping_crit( x_k_plus_1 ):
            last /= 2.0
            x_k_plus_1 = x_k*( 1 - ( last + overall_yk ) ) + ( last + overall_yk ) * s_k
        overall_yk += last
        #import pdb; pdb.set_trace()
        if overall_yk > max_yk:
            assert False
    #print( overall_yk ) 
    return overall_yk + 0.00001

def wolf_attack( gurobi_model, gurobi_xs, x_0, tf_out, tf_grad, stopping_crit, tf_input, sess, step_choice ):
    x_k_printing = x_k = x_0
    print( step_choice )
    for k in range(1000):
        if k % 100 == 99:
            print( 'K', k + 1, 'val', sess.run( tf_out, feed_dict={tf_input: x_k} ), 'dist', np.sum( np.abs( x_k_printing - x_k ) ) )
            x_k_printing = x_k
        if stopping_crit( x_k ):
            #print( 'Found, K', k, 'val', sess.run( tf_out, feed_dict={tf_input: x_k} ) ) 
            return ( x_k, True, False, k )
        s_k = wolf_attack_step( gurobi_model, gurobi_xs, x_k, tf_grad, tf_input, sess )
        if step_choice == 'regular':
            y_k = 2.0 / (k + 2)
        elif step_choice == 'brent' or step_choice == 'binary_brent' or step_choice == 'binary_brent_before':
            y_k = brent( x_k, s_k, sess, tf_input, tf_out )
            if step_choice == 'binary_brent' or step_choice == 'binary_brent_before':
                x_k_plus_1 = x_k*( 1 - y_k ) + y_k*s_k
                if stopping_crit( x_k_plus_1 ):
                    y_k = binary_search_step( x_k, s_k, y_k, stopping_crit )
                    if step_choice == 'binary_brent_before':
                        y_k = y_k - 0.00001
                    x_k_plus_1 = x_k*( 1 - y_k ) + y_k*s_k
                    if step_choice == 'binary_brent_before':
                        return ( x_k_plus_1, True, True, k+1 )
                    if not stopping_crit( x_k_plus_1 ):
                        import pdb; pdb.set_trace()
        elif step_choice == 'binary':
            y_k = binary_search_step( x_k, s_k, 2.0 / (k + 2), stopping_crit )

        x_k_plus_1 = x_k*( 1 - y_k ) + y_k*s_k
        if ( np.sum( np.abs( x_k_plus_1 - x_k ) ) / x_k.shape[ 0 ] < 1e-6 ):
        #if ( np.sum( np.abs( x_k_plus_1 - x_k ) ) < 1e-6 ):
        #    print( 'Not found, K', k, 'val', sess.run( tf_out, feed_dict={tf_input: x_k} ) ) 
            return ( x_k, stopping_crit( x_k ), False, k+1 )
        x_k = x_k_plus_1
    #print( 'Not found, K', k, 'val', sess.run( tf_out, feed_dict={tf_input: x_k} ) ) 
    return ( x_k, False, False, k )

def wolf_attack_step( gurobi_model, gurobi_xs, x_k, tf_grad, tf_input, sess ):
    df = sess.run( tf_grad, feed_dict={ tf_input: x_k } )
    dim = len( gurobi_xs )
    obj = LinExpr()
    for i in range( dim ):
        obj += df[i] * gurobi_xs[i]
    gurobi_model.setObjective(obj,GRB.MINIMIZE)
    gurobi_model.optimize()
    if(gurobi_model.SolCount==0):
        assert False
    s_k = np.zeros( dim )
    for i in range( dim ):
        s_k[i] = gurobi_xs[i].x
    return s_k

def cut_plane( neg_ex, cut_model, force=False ):
    W, b = cut_model.fit_plane( cut_model.data, neg_ex, force )

    # Update dataset
    y_pred = np.matmul( W, cut_model.data.T ) + b > 0
    y_pred = y_pred.reshape( -1 )
    new_data = cut_model.data[ y_pred, : ]
    cut_model.set_data( new_data )
    
    y_pred = np.matmul( W, neg_ex.T ) + b < 0
    y_pred = y_pred.reshape( -1 )

    assert ( np.all( np.matmul( W, cut_model.data.T ) + b > 0 ) )

    bad_idx = np.where( np.logical_not( y_pred ) )[ 0 ]
    return ( bad_idx, ( W, b ) ) 

def print_vol( cut_model ):
    vol_under, vol_over, vol_under_best = cut_model.calc_vol()
    print( 'Over:10^', vol_over )
    print( 'Under:10^', vol_under )
    if vol_under_best is None:
        print( 'Under box:', None )
    else:
        print( 'Under box:10^', vol_under_best )

def verify_the_other_half( cut_model, model_rev, W, b, nn, y_tar ):

    input_size = cut_model.input_size
    constr_names = [ constr.ConstrName for constr in model_rev.model.getConstrs()]
    
    # Try verifying other half 
    model_rev.add_hyperplane( W, b, GRB.LESS_EQUAL )
    output = model_rev.lp_verification( nn, False,  y_tar )
    
    if isinstance( output, bool ):
        for constr in cut_model.model.getConstrs():
            if not constr.ConstrName in constr_names:
                return constr.ConstrName
    else:
        example, ver_model_rev, var_list_rev, bound = output
        in_example = example[ 0 : input_size ]
                
        del ver_model_rev
        del var_list_rev

        return in_example, bound

def pool_func_last_layer( var_name ):
    thread_model = global_model.copy()
    obj = LinExpr()
    obj += thread_model.getVarByName( global_target ) 
    obj -= thread_model.getVarByName( var_name )
    thread_model.reset()
    thread_model.setObjective(obj,GRB.MINIMIZE)
    thread_model.optimize()

    if thread_model.SolCount==0:
        assert False
    
    obj_val = thread_model.objbound

    bad_exam = []
    num_vars = len( global_var_list )
    for j in range( num_vars ) :
        var = thread_model.getVarByName( global_var_list[ j ] )
        bad_exam.append( var.x )
    bad_exam = np.array( bad_exam )

    del thread_model

    return obj_val, bad_exam

def pool_func_deeppoly( idx ):
    thread_model = global_model.copy()
    input_size = global_eq[ 0 ].shape[ 1 ]
    xs = [ thread_model.getVarByName( 'x' + str( i ) ) for i in range( input_size ) ]
    obj = global_eq[ 0 ][ idx, : ] @ xs
    #for p in range( input_size ):
    #    obj += global_eq[ 0 ][ idx, p ] * xs[ p ]
    thread_model.reset()
    thread_model.setObjective( obj, GRB.MINIMIZE )
    thread_model.optimize()
    assert thread_model.SolCount == 1
    lb = thread_model.objbound + global_eq[ 2 ][ idx, 0 ]
    
    '''bad_exam_lb = []
    for p in range( input_size ) :
        bad_exam_lb.append( xs[p].x )
    bad_exam_lb = np.array( bad_exam_lb )'''
    
    obj = global_eq[ 1 ][ idx, : ] @ xs
    #for p in range( input_size ):
    #    obj += global_eq[ 1 ][ idx, p ] * xs[ p ]
    thread_model.reset()
    thread_model.setObjective( obj, GRB.MAXIMIZE )
    thread_model.optimize()
    assert thread_model.SolCount == 1
    ub = thread_model.objbound + global_eq[ 3 ][ idx, 0 ]
    
    if not global_get_example:
        return lb, ub
    
    bad_exam_ub = []
    for p in range( input_size ) :
        bad_exam_ub.append( xs[p].x )
    bad_exam_ub = np.array( bad_exam_ub )
 
    '''if not global_get_example:
        return lb, ub, bad_exam_lb, bad_exam_ub'''
   
    return lb, ub, bad_exam_ub

def pool_func( var_name ):
    lb = -GRB.INFINITY
    ub = GRB.INFINITY
    thread_model = global_model.copy()
    obj = LinExpr()
    obj += thread_model.getVarByName( var_name )

    thread_model.setObjective(obj,GRB.MINIMIZE)
    thread_model.optimize()

    if not thread_model.SolCount==0:
        lb = thread_model.objbound
    else:
        assert False
    thread_model.reset()
    thread_model.setObjective(obj,GRB.MAXIMIZE)
    thread_model.optimize()
    if not thread_model.SolCount==0:
        ub = thread_model.objbound
    else:
        assert False
    
    del thread_model
    return lb, ub

class CutModel:
    def __init__( self, sess, tf_input, tf_output, y_true, pixel_size, y_tar=None, **kwargs ):
        self.approx_obox = False
        self.sess = sess
        self.tf_input = tf_input
        self.input_size = tf_input.shape[ 0 ].value
        self.output_size = tf_output.shape[ -1 ].value
        self.pixel_size = pixel_size
        self.tf_output = tf_output
        
        self.tf_nlb = None
        self.tf_nub = None
        self.tf_attack = None
        self.tf_sampling_layers = None
        self.tf_sampling_x = None
        
        if not y_tar is None:
            self.update_target( y_true, y_tar )
        else:
            self.y_true = y_true
            self.y_tar = None

        if 'lb' in kwargs:
            lb = kwargs[ 'lb' ]
            ub = kwargs[ 'ub' ]
            self.reset_model( lb, ub )

        elif 'model' in kwargs:
            self.model = kwargs[ 'model' ]
            self.xs = kwargs[ 'xs' ]
            npdata = kwargs[ 'npdata' ]
            self.data = npdata[ 'data' ]
            if np.any( np.equal( self.data, None ) ):
                self.data = None
                self.data_size = 0
            else:
                self.data_size = self.data.shape[ 0 ]

            self.obox = npdata[ 'obox' ]
            if np.any( np.equal( self.obox, None ) ):
                self.obox = None

            self.ubox = npdata[ 'ubox' ]
            if np.any( np.equal( self.ubox, None ) ):
                self.ubox = None

            self.x0 = npdata[ 'x0' ]
            if np.any( np.equal( self.x0, None ) ):
                self.x0 = None

            self.precision = npdata[ 'precision' ]
            if np.any( np.equal( self.precision, None ) ):
                self.precision = None

            self.nlb = npdata[ 'nlb' ]
            self.nub = npdata[ 'nub' ]

            if type( self.nlb ) is np.ndarray:
                if self.nlb.size == 1 and self.nlb == None:
                    self.nlb = None
                self.nlb = self.nlb.tolist()
            if type( self.nub ) is np.ndarray:
                if self.nub.size == 1 and self.nub == None:
                    self.nub = None
                self.nub = self.nub.tolist()

            self.W = npdata[ 'W' ]
            self.model_nlb = npdata[ 'model_nlb' ]
            self.model_nub = npdata[ 'model_nub' ]
            self.cuts = npdata[ 'cuts' ]

            self.model.update()

    def set_data( self, data ):
        self.data = data
        self.data_size = data.shape[ 0 ]
    
    def reset_model( self, lb, ub ):
        model = Model( 'LP' )
        model.setParam( 'OutputFlag', 0 )
        
        xs = []
        input_size = self.input_size
        for i in range( input_size ):
            x = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[i], ub=ub[i], name='x%i' % i)
            xs.append( x )

        self.model = model
        self.xs = xs
        self.cuts = 0
        self.invalidate_calc()
        self.nlb = self.nub = None
        self.W = np.zeros( ( 0, input_size + 1 ) )
        self.model_nlb = None
        self.model_nub = None
        self.update_W_bounds()

    def update_target( self, y_true, y_tar ):
        self.y_true = y_true
        self.y_tar = y_tar

        self.tf_out_pos = self.tf_output[ 0, y_tar ] - self.tf_output[ 0, y_true ]
        self.tf_grad_positive = tf.gradients( self.tf_out_pos, self.tf_input )[ 0 ]
        self.tf_out_neg = self.tf_output[ 0, self.y_true ] - self.tf_output[ 0, y_tar ]
        self.tf_grad_negative = tf.gradients( self.tf_out_neg, self.tf_input )[ 0 ]

        def stopping_crit_positive( x_k ):
            output = self.sess.run( self.tf_output, feed_dict={ self.tf_input: x_k } )
            return np.argmax( output ) == y_tar

        def stopping_crit_negative( x_k ):
            output = self.sess.run( self.tf_output, feed_dict={ self.tf_input: x_k } )
            return np.argmax( output ) == y_true

        self.stopping_crit_positive = stopping_crit_positive
        self.stopping_crit_negative = stopping_crit_negative
	
        self.data = None
        self.nlb = self.nub = None
        self.data_size = 0

    def invalidate_calc( self ):
        self.model.update()

        self.obox = None
        self.ubox = None
        self.x0 = None
        self.precision = None

    def update_bounds( self, lb, ub ):
        for i in range( self.input_size ):
            self.xs[ i ].setAttr( GRB.Attr.UB, ub[ i ] )
            self.xs[ i ].setAttr( GRB.Attr.LB, lb[ i ] )
        self.invalidate_calc()
        self.update_W_bounds()

    def update_W_bounds( self ):
        self.model_nlb = np.zeros( self.input_size )
        self.model_nub = np.zeros( self.input_size )
        for i in range( self.input_size ):
            self.model_nlb[ i ] = self.xs[ i ].LB
            self.model_nub[ i ] = self.xs[ i ].UB

    @staticmethod
    def load( name, sess, tf_input, tf_output, y_true ): 
        npdata = np.load( name + '/npdata.npz', allow_pickle=True )
        
        model = read( name + '/' + name + '.lp' )
        model.setParam( 'OutputFlag', 0 )
        model.update()

        num_xs = len( model.getVars() )
        xs = [ model.getVarByName( 'x%i' % i ) for i in range( num_xs ) ]
        st0 = npdata[ 'rand_state' ]
        pixel_size = npdata[ 'pixel_size' ]
        st0 = tuple( st0.tolist() )
        np.random.set_state( st0 )
        y_tar = npdata[ 'y_tar' ]

        return CutModel( sess, tf_input, tf_output, y_true, pixel_size, y_tar=y_tar, model=model, xs=xs, npdata=npdata )
    
    def save( self, name, baseline=False ):
        if baseline:
            name = name + '_baseline_it_' + str( self.cuts )
        else:
            name = name + '_it_' + str( self.cuts )
        try:
            os.makedirs( name )
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.model.write( name + '/' + name + '.lp' )
        rand_state = np.random.get_state()
        np.savez( name + '/npdata.npz', 
                  data=self.data, obox=self.obox, ubox=self.ubox, x0=self.x0, W=self.W, 
                  rand_state=rand_state, cuts=self.cuts, pixel_size=self.pixel_size, 
                  precision=self.precision, y_tar=self.y_tar, nlb=self.nlb, nub=self.nub, 
                  model_nlb=self.model_nlb, model_nub=self.model_nub, 
                  shrink_lb=self.shrink_box_best[0], shrink_ub=self.shrink_box_best[1],
                  denorm_func=self.denorm_func, norm_func=self.norm_func, 
                  norm_args=self.norm_args, is_conv=self.is_conv, dataset=self.dataset,
                  image=self.orig_image, orig_over_lb=self.orig_over_lb, orig_over_ub=self.orig_over_ub )
        
    def copy( self ):
        model = self.model.copy()
        xs = [ model.getVarByName( 'x%i' % i ) for i in range( self.input_size ) ]
        
        if not self.data is None:
            data = self.data.copy()
        else:
            data = None

        if not self.x0 is None:
            x0 = self.x0.copy()
        else:
            x0 = None

        if not self.ubox is None:
            ubox = ( self.ubox[ 0 ].copy(), self.ubox[ 1 ].copy() )
        else:
            ubox = None

        if not self.obox is None:
            obox = ( self.obox[ 0 ].copy(), self.obox[ 1 ].copy() )
        else:
            obox = None

        if not self.W is None:
            W = self.W.copy()
        else:
            W = None
        
        if not self.W is None:
            model_nlb = self.model_nlb.copy()
        else:
            model_nlb = None

        if not self.model_nub is None:
            model_nub = self.model_nub.copy()
        else:
            model_nub = None

        npdata = { 'data': data, 'x0': x0, 'obox': obox, 'ubox': ubox, 
                   'W': W, 'cuts': self.cuts, 'pixel_size': self.pixel_size, 
                   'precision': self.precision, 'nlb': self.nlb,'nub': self.nub,
                   'model_nlb': model_nlb, 'model_nub': model_nub }

        copy = CutModel( self.sess, self.tf_input, self.tf_output, self.y_true, self.pixel_size, self.y_tar, model=model, xs=xs, npdata=npdata )
        
        if 'shrink_box_best' in dir(self):
            copy.shrink_box_best = ( self.shrink_box_best[0].copy(), self.shrink_box_best[1].copy() )

        copy.tf_grad_deeppoly = self.tf_grad_deeppoly
        copy.activation_pattern_lin = self.activation_pattern_lin
        copy.activation_pattern_pt = self.activation_pattern_pt 
        copy.activation_pattern_in = self.activation_pattern_in
        copy.activation_pattern_extract = self.activation_pattern_extract
        copy.tf_nlb = self.tf_nlb
        copy.tf_nub = self.tf_nub
        copy.tf_attack = self.tf_attack
        copy.tf_sampling_layers = self.tf_sampling_layers
        copy.tf_sampling_x = self.tf_sampling_x
        copy.backsubstitute_tens = self.backsubstitute_tens
        copy.bounds = self.bounds

        return copy

    def add_hyperplane( self, W, b, Sense=GRB.GREATER_EQUAL ):
        norm = np.max( np.abs( W ), axis=1 )
        norm = norm.reshape(-1,1)
        W = W / norm
        b = b / norm
        constrs = W @ self.xs
        assert b.shape[1] == 1
        for i in range( len( constrs ) ):    
            self.model.addConstr( constrs[i], Sense, -b[i] )
            self.cuts += 1
        if Sense == GRB.GREATER_EQUAL:
            W = -W 
            b = -b
        new_hp = np.concatenate( (W,b.reshape(-1,1)), axis=1 )
        self.W = np.concatenate( (self.W, new_hp), axis=0 )
        self.invalidate_calc()

    def check_if_inside( self, samples, precision=None ):
        if precision == None:
            precision = self.precision
        
        if samples.ndim == 1:
            samples =samples.reshape( 1, -1 )
        num_samples = samples.shape[ 0 ]
        ones = np.ones( ( num_samples, 1 ) )
        samples_extended = np.concatenate( ( samples, ones ), axis=1 )
        matmul = np.matmul( self.W, samples_extended.T ) < precision
        full_idx = np.all( matmul, axis=0 )
        full_idx = np.logical_and( full_idx, np.all( ( samples - self.model_nlb ) > -precision, axis=1 ) )
        full_idx = np.logical_and( full_idx, np.all( ( samples - self.model_nub ) <  precision, axis=1 ) )
        good_idx = np.where( full_idx ) [ 0 ]
        return good_idx 

    def sample_poly_under( self, num_samples ):
        '''Underapprox box and sample from it'''
        lb, ub = self.underapprox_box()
        samples = np.random.uniform( low=lb, high=ub, size=( num_samples, lb.shape[ 0 ] ) )
        return samples

    def overapprox_box( self ):
        if not self.obox is None:
            return self.obox
        
        t = time.time()
        if self.approx_obox:
            lb = []
            ub = []
            for i in range( self.input_size ):
                lb.append( self.xs[ i ].LB )
                ub.append( self.xs[ i ].UB )
            lb = np.array( lb )
            ub = np.array( ub )
        else:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")

            global global_model
            global_model = self.model.copy()
            global_model.setParam(GRB.Param.Threads, 1)
            global_model.update()
        
            var_names = [ var.VarName for var in self.xs ]
            with multiprocessing.Pool(ncpus) as pool:
                solver_result = pool.map( pool_func, var_names )
            del globals()[ 'global_model' ]
            lb = np.array( [ 0 ] * self.input_size, dtype=np.float64 )
            ub = np.array( [ 0 ] * self.input_size, dtype=np.float64 )
            for i in range( self.input_size ):
                id = int( var_names[ i ][ 1 : ] )
                lb[ id ] = solver_result[ i ][ 0 ]
                ub[ id ] = solver_result[ i ][ 1 ]
        elapsed_time = time.time() - t
        print( 'Overapprox Time:', elapsed_time, 'secs' )

        self.obox = ( lb, ub )
        return self.obox
    def create_underapprox_box_lp( self, y_tar, shrink_to=-100, baseline=False ):
        st = time.time()
        data = self.data
        lb, ub = self.overapprox_box()
        lb_orig, ub_orig = lb, ub
        centre = self.data[ np.argmax( self.bounds ), : ]

        nlb_copy = [ lb.copy() for lb in self.nlb[1:] ]
        nub_copy = [ ub.copy() for ub in self.nub[1:] ]
        
        shrink = 1
        lb_st = None
        ub_st = None
        print( "LP underbox start:", shrink_to )
        def verify_eps( eps, lb, ub, centre, nlb_copy, nub_copy ):
            dev_r = ub - centre 
            dev_l = centre - lb
            ub = centre + dev_r*eps
            lb = centre - dev_l*eps

            self.obox = None
            self.ubox = None
            self.update_target( self.y_true, y_tar )
            self.reset_model( lb, ub )
            self.update_bounds( lb, ub )
            self.nlb = [ lb.copy() for lb in nlb_copy ]
            self.nub = [ ub.copy() for ub in nub_copy ]
            self.nlb.insert( 0, lb )
            self.nub.insert( 0, ub )

            output = self.extract_deeppoly_backsub( y_tar, box=True )
            if output is True:
                bound = 0
            else:
                bound = output[-2]
            return bound, lb, ub 

        '''
        min_bound = 1.0 / ( 2**7 )
        eps = min_bound
        sol = None
        print( 'Initial search' )
        while eps < 1:
            bound, lb_sol, ub_sol = verify_eps( eps, lb, ub, centre, nlb_copy, nub_copy )
            if bound < 0:
                print( 'Eps', eps, 'Bound', bound )
                break
            print( 'Solution: Eps', eps )
            sol = eps
            eps *= 2.0
        if eps == None:
            eps = 1.0
        start_eps = eps / 4.0
        print( 'Initial eps', sol )
        while start_eps >= min_bound:
            print( 'Outer eps', eps )
            while start_eps >= min_bound:
                bound, lb_new, ub_new = verify_eps( eps - start_eps, lb, ub, centre, nlb_copy, nub_copy )
                if bound < 0:
                    print( 'Inner eps', eps - start_eps, 'Bound', bound )
                    break
                else:
                    sol = eps - start_eps
                    lb_sol = lb_new
                    ub_sol = ub_new
                    print( 'Solution: Inner eps', eps - start_eps )
                start_eps /= 2.0
            eps -= start_eps
            start_eps /= 2.0

        return lb_sol, ub_sol, None, None
        '''
        if baseline:
            while True:
                output, lb_new, ub_new = verify_eps( shrink, lb, ub, centre, nlb_copy, nub_copy )
                shrink -= 0.025
                if shrink < 0:
                    break
                if output >= 0:
                    lb_st = lb_new
                    ub_st = ub_new
                    lb = lb_new
                    ub = ub_new
                    return lb, ub, lb_st, ub_st

            return centre, centre, centre, centre

        its = 0
        while True:
            output, lb_new, ub_new = verify_eps( shrink, lb, ub, centre, nlb_copy, nub_copy )
            shrink -= 0.025
            if output > shrink_to:
                lb_st = lb_new
                ub_st = ub_new
                lb = lb_new
                ub = ub_new
                break
            its += 1
        if its == 0:
            shrink_to = output

        model = Model()
        model.setParam( 'OutputFlag', 0 )
        lbs = model.addVars( self.input_size, lb=lb, ub=ub )
        lbs = [ lbs[k] for k in lbs ] 
        ubs = model.addVars( self.input_size, lb=lb, ub=ub )
        ubs = [ ubs[k] for k in ubs ] 
        for i in range( self.input_size ):
            model.addConstr( lbs[i], GRB.LESS_EQUAL, ubs[i] ) 
        ones = np.ones( self.input_size ) 
        model.setObjective( ubs @ ones - lbs @ ones, GRB.MAXIMIZE )
        model.update()
        
        c_balance = 0.99
        for it in range(50):
            output = self.extract_deeppoly_backsub( y_tar, box=True )
            if output is True:
                end = time.time()
                print( 'Verified underapprox box:', end - st, 's' )
                return lb, ub, lb_st, ub_st

            eq, example, attack_class, bound, hps = output
            best_bound = np.maximum( eq[0], 0 ) @ ub + np.minimum( eq[0], 0 ) @ lb + eq[1]
            tar_bound = bound * c_balance + best_bound * (1-c_balance)
            if np.abs( bound ) < 0.05 and best_bound > 1e-7:
                tar_bound = 1e-7
            print( 'It:', it, 'Bound:', output[-2], 'Target:', tar_bound )
            copy = model.copy()
            lin = LinExpr()
            for i in range( self.input_size ):
                if eq[0][i] > 0: 
                    lin += eq[0][i] * copy.getVarByName( lbs[i].VarName )
                else:
                    lin += eq[0][i] * copy.getVarByName( ubs[i].VarName )

            copy.addConstr( lin + eq[1], GRB.GREATER_EQUAL, tar_bound )
            copy.optimize()
            assert copy.Status == 2 
            lb = [ copy.getVarByName( lbs[i].VarName ).X for i in range( self.input_size ) ]
            ub = [ copy.getVarByName( ubs[i].VarName ).X for i in range( self.input_size ) ]

            self.obox = None
            self.ubox = None
            self.update_target( self.y_true, y_tar )
            self.reset_model( lb, ub )
            self.update_bounds( lb, ub )
            self.nlb = [ lb.copy() for lb in nlb_copy ]
            self.nub = [ ub.copy() for ub in nub_copy ]
            self.nlb.insert( 0, lb )
            self.nub.insert( 0, ub )
            for i in range( self.input_size ):
                lbs[i].LB = lb[i]
                ubs[i].LB = lb[i]
                lbs[i].UB = ub[i]
                ubs[i].UB = ub[i]
            model.update()
            c_balance *= 0.99

        self.obox = None
        self.ubox = None
        self.update_target( self.y_true, y_tar )
        self.reset_model( lb_orig, ub_orig )
        self.update_bounds( lb_orig, ub_orig )
        self.nlb = [ lb.copy() for lb in nlb_copy ]
        self.nub = [ ub.copy() for ub in nub_copy ]
        self.nlb.insert( 0, lb_orig )
        self.nub.insert( 0, ub_orig )
        self.data = data
        return self.create_underapprox_box_lp( y_tar, shrink_to=shrink_to/1.5 )

    def underapprox_box( self ):
        if not self.ubox is None:
            return self.ubox

        model_new = Model( 'Underbox' )
        model_new.setParam( 'OutputFlag', 0 )
        vars_new = {}

        lbo, ubo = self.overapprox_box()
        for i in range( self.input_size ):
            var_lo = model_new.addVar(vtype=GRB.CONTINUOUS, lb=lbo[ i ], ub=ubo[ i ], name='lo_' + self.xs[ i ].VarName)
            var_hi = model_new.addVar(vtype=GRB.CONTINUOUS, lb=lbo[ i ], ub=ubo[ i ], name='hi_' + self.xs[ i ].VarName)
            vars_new[ self.xs[ i ] ] = ( var_lo, var_hi )
            constr_new = LinExpr()
            constr_new += var_hi - var_lo
            model_new.addConstr( constr_new, GRB.GREATER_EQUAL, 0 )

        for constr in self.model.getConstrs():
            constr_new = LinExpr()
            for x in self.xs:
                coef = self.model.getCoeff( constr, x )
                if ( coef >= 0 and constr.Sense == '<' ) or ( coef < 0 and constr.Sense == '>' ) :
                    constr_new += coef*vars_new[ x ][ 1 ]
                if ( coef < 0 and constr.Sense == '<' ) or ( coef >= 0 and constr.Sense == '>' ) :
                    constr_new += coef*vars_new[ x ][ 0 ]
                if constr.Sense == '=':
                    assert False
            model_new.addConstr( constr_new, constr.Sense, constr.RHS )
        
        obj = LinExpr()
        for x in self.xs:
            obj += vars_new[ x ][ 1 ] - vars_new[ x ][ 0 ]
        model_new.setObjective( obj, GRB.MAXIMIZE )
        model_new.optimize()
        if model_new.SolCount == 0:
            assert False

        lb = np.zeros( self.input_size )
        ub = np.zeros( self.input_size )
        for i in range( self.input_size ):
            x = self.xs[ i ]
            lb[ i ] = vars_new[ x ][ 0 ].x
            ub[ i ] = vars_new[ x ][ 1 ].x

        del model_new
        
        self.ubox = ( lb, ub )
        return self.ubox

    def calc_vol( self ):
        lbo, ubo = self.overapprox_box()
        lbu, ubu = self.underapprox_box()
        if 'shrink_box_best' in dir(self) and not self.shrink_box_best is None:
            lbub, ubub = self.shrink_box_best

        sizes = ( ( ubo - lbo ) / self.pixel_size ).astype( np.int64 )
        sizes = sizes[ sizes > 0 ]
        vol_over = np.sum( np.log10( sizes ) )

        sizes = ( ( ubu - lbu ) / self.pixel_size ).astype( np.int64 )
        sizes = sizes[ sizes > 0 ]
        vol_under = np.sum( np.log10( sizes ) )
        
        if 'shrink_box_best' in dir(self) and not self.shrink_box_best is None:
            sizes = ( ( ubub - lbub ) / self.pixel_size ).astype( np.int64 )
            sizes = sizes[ sizes > 0 ]
            vol_under_best = np.sum( np.log10( sizes ) )
        else:
            vol_under_best = None

        return ( vol_under, vol_over, vol_under_best )

    def eval_network( self, input ):
        out = self.sess.run( self.tf_out_pos, feed_dict={ self.tf_input: input } )
        return out

    def lp_verification( self, nn, ver_type, target, complete=False ):
        if ver_type == 'MILP':
            use_milp = True
        else:
            use_milp = False
        if ver_type == 'DeepPoly':
            return self.extract_deeppoly_backsub( target )
            use_deeppoly = True
        else:
            use_deeppoly = False
        
        ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
        self.full_models=[]
        input_size = self.input_size
        
        if self.nlb is None:
            self.nlb = [ np.array( [] ) ] * ( nn.numlayer + 1 )
        if self.nub is None:
            self.nub = [ np.array( [] ) ] * ( nn.numlayer + 1 )

        if complete or len( self.nlb[ 0 ] ) == 0 or len( self.nub[ 0 ] ) == 0:
            LB_N0, UB_N0 = self.overapprox_box()
            self.nlb[ 0 ] = LB_N0
            self.nub[ 0 ] = UB_N0
        for layerno in range( nn.numlayer ):
            relu_needed = [0]*(layerno+1)
            for i in range(layerno):
                relu_needed[i] = 1
            krelu_groups = [ None ]*(layerno+1)
            if use_deeppoly:
                deeppoly = [1]*(layerno+1)
            else:
                deeppoly = False

            print( 'Layer:', layerno+1, 'MILP:', use_milp, 'DeepPoly:', deeppoly )
            counter, var_list, model = create_model(nn, self.nlb[ 0 ], self.nub[ 0 ], self.nlb[ 1 : layerno + 1 ] + [[]], self.nub[ 1 : layerno + 1 ] + [[]], krelu_groups, layerno + 1, use_milp, relu_needed, deeppoly)

            num_vars = len( var_list )
            output_size = num_vars - counter
            model.update()
            for constr in self.model.getConstrs():
                constr_new = LinExpr()
                for i in range( input_size ):
                    coef = self.model.getCoeff( constr, self.xs[ i ] )
                    constr_new += coef * var_list[ i ]
                    assert var_list[i].VarName == self.xs[i].VarName
                model.addConstr( constr_new, constr.Sense, constr.RHS )
            
            self.full_models.append(model)

            global global_model
            global_model = model

            if layerno == nn.numlayer - 1:
                model.setParam(GRB.Param.Threads, 1)
                model.update()

                t = time.time()

                global global_var_list
                global_var_list = [ var.VarName for var in var_list ]
                
                global global_target
                global_target = var_list[ counter + target ].VarName

                adv_targets = [ var.VarName for var in var_list[ counter : ] if not global_target == var.VarName ] 
                with multiprocessing.Pool(ncpus) as pool:
                    solver_result = pool.map( pool_func_last_layer, adv_targets )
                
                del globals()[ 'global_model' ]
                del globals()[ 'global_var_list' ]
                del globals()[ 'global_target' ]

                smallestobj = 0.0
                bad_exam = None
                bad_class = None
                j = 0
                for i in range(output_size):
                    if i == target:
                        continue
                    obj_val = solver_result[ j ][ 0 ]
                    if obj_val < smallestobj:
                        smallestobj = obj_val
                        bad_class = i
                        bad_exam = solver_result[ j ][ 1 ]
                    j += 1
                
                del var_list
                del model
                del counter

                elapsed_time = time.time() - t                                                                         
                print( 'Time:', elapsed_time, 'secs' )

                if bad_class == None:
                    return True

                return ( bad_exam, bad_class, smallestobj )
            else:
                bounds_are_computed = not len( self.nlb[ layerno + 1 ] ) == 0 and not len( self.nub[ layerno + 1 ] ) == 0
                if bounds_are_computed:
                    lbi = self.nlb[ layerno + 1 ]
                    ubi = self.nub[ layerno + 1 ]
                else:
                    lbi = np.array( [ -GRB.INFINITY ] * output_size )
                    ubi = np.array( [ GRB.INFINITY ] * output_size )
                
                model.setParam(GRB.Param.Threads, 1)
                model.update()

                t = time.time()
                if complete or not bounds_are_computed:
                    neurons = np.array( range( output_size ), np.int64 )
                else:
                    l = np.array( self.nlb[ layerno + 1 ] ) < 0
                    u = np.array( self.nub[ layerno+ 1 ] ) > 0
                    neurons = np.where( np.logical_and( l, u ) )[ 0 ]
                    neurons = np.int64( neurons )
                #neurons = np.array( [0], np.int64 ) 
                var_names = [ var_list[ var ].VarName for var in counter + neurons ]
                print( 'Recomputed Vars:', len( var_names ) )
                with multiprocessing.Pool(ncpus) as pool:
                    solver_result = pool.map( pool_func, var_names )
                del globals()[ 'global_model' ]
                for i in range( neurons.shape[ 0 ] ):
                    lbi[ neurons[ i ] ] = solver_result[ i ][ 0 ]
                    ubi[ neurons[ i ] ] = solver_result[ i ][ 1 ]
                elapsed_time = time.time() - t
                print( 'Time:', elapsed_time, 'secs' )
                if bounds_are_computed:
                    self.nub[ layerno + 1 ] = np.minimum( self.nub[ layerno + 1 ], ubi )
                    self.nlb[ layerno + 1 ] = np.maximum( self.nlb[ layerno + 1 ], lbi )
                else:
                    self.nub[ layerno + 1 ] = ubi
                    self.nlb[ layerno + 1 ] = lbi

            del var_list
            del model
            del counter

    def create_tf_sampling_net( self, net_file, is_trained_with_pytorch, ver_type ):
        from read_net_file import myConst, parseVec, extract_mean, extract_std, permutation, runRepl
        if ver_type == 'MILP':
            use_milp = True
        else:
            use_milp = False
        if ver_type == 'DeepPoly':
            use_deeppoly = True
        else:
            use_deeppoly = False
        
        mean = 0.0
        std = 0.0
        net = open(net_file,'r')
        x = tf.placeholder( tf.float64, [ None, self.input_size ], name='x_sampling' )
        num_images = tf.shape( x )[ 0 ]
        sampling_layers_size = tf.stack( ( num_images, -1 ), axis=0 )
        self.tf_sampling_x = x
        last_layer = None
        h,w,c = None, None, None
        is_conv = False

        tf_attack = tf.placeholder( x.dtype, shape=[None] )
        tf_attack_idx = self.input_size 
        relu_idx = 1 
        tf_nlb = []
        tf_nub = []
        tf_sampling_layers = []
       
        deeppoly_layer_args = []
        shapes = []
        while True:
            curr_line = net.readline()[:-1]
            if 'Normalize' in curr_line:
                mean = extract_mean(curr_line)
                std = extract_std(curr_line)
            elif curr_line in ["ReLU", "Affine"]:
                print(curr_line)
                W = None
                if (last_layer in ["Conv2D", "ParSumComplete", "ParSumReLU"]) and is_trained_with_pytorch:
                    W = myConst(permutation(parseVec(net), h, w, c).transpose())
                else:
                    W = myConst(parseVec(net).transpose())
                b = parseVec(net)
                if len( shapes ) == 0:
                    shapes.append( x.shape )
                #b = myConst(b.reshape([1, numel(b)]))
                b = myConst(b)
                if(curr_line=="Affine"):
                    x = tf.nn.bias_add(tf.matmul(tf.reshape(x, sampling_layers_size),W), b)
                    tf_sampling_layers.append( tf.reshape( x, shape=sampling_layers_size ) )
                    deeppoly_layer_args.append( ( 'Affine', W, b, None, None ) )
                    shapes.append( x.shape )
                elif(curr_line=="ReLU"):
                    x = tf.nn.bias_add(tf.matmul(tf.reshape(x, sampling_layers_size),W), b)
                    tf_sampling_layers.append( tf.reshape( x, shape=sampling_layers_size ) )
                    prev_layer_attack = tf_attack[ tf_attack_idx : tf_attack_idx + np.prod( x.shape[ 1 : ] ) ]
                    tf_attack_idx_st = tf_attack_idx + np.prod( x.shape[ 1 : ] )
                    tf_attack_idx = tf_attack_idx_en = tf_attack_idx_st + np.prod( x.shape[ 1 : ] )
                    tf_attack_layer = tf_attack[ tf_attack_idx_st : tf_attack_idx_en ]
                    x, x_bef_reshape, tf_lbi, tf_ubi = self.create_tf_layer( x, prev_layer_attack, tf_attack_layer, relu_idx, use_deeppoly )
                    tf_nlb.append( tf_lbi )
                    tf_nub.append( tf_ubi )
                    tf_sampling_layers.append( x_bef_reshape )
                    deeppoly_layer_args.append( ( 'Affine', W, b, tf_lbi, tf_ubi ) )
                    shapes.append( x.shape )
                    relu_idx += 1
                print("\tOutShape: ", x.shape)
                print("\tWShape: ", W.shape)
                print("\tBShape: ", b.shape)
            elif curr_line == "Conv2D":
                is_conv = True
                line = net.readline()
                args = None
                #print(line[-10:-3])
                start = 0
                if("ReLU" in line):
                    start = 5
                elif("Sigmoid" in line):
                    start = 8
                elif("Tanh" in line):
                    start = 5
                elif("Affine" in line):
                    start = 7
                if 'padding' in line:
                    args =  runRepl(line[start:-1], ["filters", "input_shape", "kernel_size", "stride", "padding"])
                else:
                    args = runRepl(line[start:-1], ["filters", "input_shape", "kernel_size"])

                W = myConst(parseVec(net))
                print("W shape", W.shape)
                #W = myConst(permutation(parseVec(net), h, w, c).transpose())
                b = None
                if("padding" in line):
                    if(args["padding"]==1):
                        padding_arg = "SAME"
                    else:
                        padding_arg = "VALID"
                else:
                    padding_arg = "VALID"

                if("stride" in line):
                    stride_arg = [1] + args["stride"] + [1]
                else:
                    stride_arg = [1,1,1,1]

                tf_out_shape = tf.stack( [ num_images ] + args["input_shape"], axis=0 )
                x = tf.reshape(x, tf_out_shape)
                if len( shapes ) == 0:
                    shapes.append( x.shape )

                x = tf.nn.conv2d(x, filter=W, strides=stride_arg, padding=padding_arg)
                b = myConst(parseVec(net))
                h, w, c = [ int( i ) for i in x.shape[ 1 : ] ]
                print("Conv2D", args, "W.shape:",W.shape, "b.shape:", b.shape)
                print("\tOutShape: ", x.shape)
                if("ReLU" in line):
                    x = tf.nn.bias_add(x, b)
                    tf_sampling_layers.append( tf.reshape( x, shape=sampling_layers_size ) )
                    prev_layer_attack = tf_attack[ tf_attack_idx : tf_attack_idx + np.prod( x.shape[ 1 : ] ) ]
                    tf_attack_idx_st = tf_attack_idx + np.prod( x.shape[ 1 : ] )
                    tf_attack_idx = tf_attack_idx_en = tf_attack_idx_st + np.prod( x.shape[ 1 : ] )
                    tf_attack_layer = tf_attack[ tf_attack_idx_st : tf_attack_idx_en ]
                    x, x_bef_reshape, tf_lbi, tf_ubi = self.create_tf_layer( x, prev_layer_attack, tf_attack_layer, relu_idx, use_deeppoly )
                    tf_sampling_layers.append( x_bef_reshape )
                    tf_nlb.append( tf_lbi )
                    tf_nub.append( tf_ubi )
                    deeppoly_layer_args.append( ( 'Conv2D', W, b, stride_arg, padding_arg, tf_lbi, tf_ubi ) )
                    shapes.append( x.shape )
                    relu_idx += 1
                elif("Affine" in line):
                    x = tf.nn.bias_add(x, b)
                    tf_sampling_layers.append( tf.reshape( x, shape=sampling_layers_size ) )
                    deeppoly_layer_args.append( ( 'Conv2D', W, b, stride_arg, padding_arg, None, None ) )
                    shapes.append( x.shape )
                else:
                    raise Exception("Unsupported activation: ", curr_line)
            elif curr_line == "":
                break
            else:
                raise Exception("Unsupported Operation: ", curr_line)
            last_layer = curr_line
        if 'ReLU' in last_layer:
            tf_nlb = tf_nlb[ : -1 ]
            tf_nub = tf_nub[ : -1 ]
        self.tf_nlb = tf_nlb
        self.tf_nub = tf_nub
        self.tf_attack = tf_attack
        self.tf_sampling_layers = tf_sampling_layers

        x_inf = tf.placeholder( tf.float64, shapes[ -1 ], name='x_backward_deeppoly_inf' )
        x_sup = tf.placeholder( tf.float64, shapes[ -1 ], name='x_backward_deeppoly_sup' )
        x_inf_cst = tf.placeholder( tf.float64, [ None, 1 ], name='x_backward_deeppoly_inf_cst' )
        x_sup_cst = tf.placeholder( tf.float64, [ None, 1 ], name='x_backward_deeppoly_sup_cst' )
        backsubstitute = [ ( x_inf, x_sup, x_inf_cst, x_sup_cst ) ]
        layer_types = [] 
        for i in range( len( deeppoly_layer_args )-1, -1, -1 ):
            if deeppoly_layer_args[i][0] == 'Affine':
                if i != 0:
                    nlb_nub = deeppoly_layer_args[ i - 1 ][ -2 : ]
                else:
                    nlb_nub = [ None, None ]
                out = self.create_deeppoly_backsubs_ffn( *deeppoly_layer_args[ i ][ 1 : -2 ], *nlb_nub, *backsubstitute[ 0 ] )
                out = list( out )
                out[ 0 ] = tf.reshape( out[ 0 ], [ -1 ] + shapes[ i ][ 1 : ].as_list() ) 
                out[ 1 ] = tf.reshape( out[ 1 ], [ -1 ] + shapes[ i ][ 1 : ].as_list() )
                backsubstitute.insert( 0, out )
            elif deeppoly_layer_args[i][0] == 'Conv2D':
                if i != 0:
                    nlb_nub = deeppoly_layer_args[ i - 1 ][ -2 : ]
                else:
                    nlb_nub = [ None, None ]
                out = self.create_deeppoly_backsubs_conv( shapes[ i ][ 1 : ].as_list(), *deeppoly_layer_args[ i ][ 1 : -2 ], *nlb_nub, *backsubstitute[ 0 ] )
                out = list( out )
                out[ 0 ] = tf.reshape( out[ 0 ], [ -1 ] + shapes[ i ][ 1 : ].as_list() ) 
                out[ 1 ] = tf.reshape( out[ 1 ], [ -1 ] + shapes[ i ][ 1 : ].as_list() ) 
                backsubstitute.insert( 0, out )
        self.backsubstitute_tens = backsubstitute
        
        grad_calc = tf.reduce_sum(self.backsubstitute_tens[ 0 ][ 1 ]) + tf.reduce_sum(self.backsubstitute_tens[ 0 ][ 3 ])
        self.tf_grad_deeppoly = tf.gradients( grad_calc, self.tf_nlb + self.tf_nub )
        
        self.activation_pattern_lin = []
        x_real = self.activation_pattern_pt = tf.placeholder( tf.float64, [ None, self.input_size ], name='x_app' )
        x = self.activation_pattern_in = tf.placeholder( tf.float64, [ None, self.input_size ], name='x_api' )
        self.activation_pattern_extract = [] 
        for i in range(len(deeppoly_layer_args)):
            args = deeppoly_layer_args[i]
            shape = shapes[i]
            shape = np.array( [ i.value for i in shape ] )
            assert np.sum( shape == None ) == 1 
            shape[ shape == None ] = -1
            if args[0] == 'Affine':
                t, W, b, RELU, _ = args
                shape = [-1] + [ np.prod( shape[1:] ) ]
                x = tf.nn.bias_add(tf.matmul(tf.reshape(x, shape), W), b)
                x_real = tf.nn.bias_add(tf.matmul(tf.reshape(x_real, shape), W), b)
            elif args[0] == 'Conv2D':
                t, W, b, stride_arg, padding_arg, RELU, _ = args
                x = tf.reshape( x, shape )
                x_real = tf.reshape( x_real, shape )
                x = tf.nn.conv2d(x, filter=W, strides=stride_arg, padding=padding_arg)
                x_real = tf.nn.conv2d(x_real, filter=W, strides=stride_arg, padding=padding_arg)
                x = tf.nn.bias_add(x, b)
                x_real = tf.nn.bias_add(x_real, b)
            else:
                assert False, 'Bad layer'
            RELU = not RELU is None
            if RELU:
                self.activation_pattern_extract.append( x_real )
                self.activation_pattern_lin.append( x )
                x *= tf.cast( x_real >= 0.0, x.dtype )  
                x_real = tf.nn.relu( x_real )
        
        gridx = np.array( range( self.output_size ) )
        gridy = np.array( range( self.output_size ) )
        gridx, gridy = np.meshgrid(gridx, gridy)
        gridx = gridx.reshape( -1 )
        gridy = gridy.reshape( -1 )

        x = tf.gather( x, gridx, axis=1 ) - tf.gather( x, gridy, axis=1 )
        x_real = tf.gather( x_real, gridx, axis=1 ) - tf.gather( x_real, gridy, axis=1 )
        self.activation_pattern_lin.append( x )
        self.activation_pattern_extract.append( x_real )

        self.diff_lp_A = tf.placeholder( tf.float64, (None, self.input_size), name='diff_A' )  
        self.diff_lp_b_var = tf.placeholder( tf.float64, (None), name='diff_b_var' )
        self.diff_lp_b_pl = tf.placeholder( tf.float64, (None), name='diff_b_pl' ) 
        self.diff_lp_b = tf.concat( (self.diff_lp_b_var, self.diff_lp_b_pl), axis=0 ) 
        self.diff_lp_lb = tf.placeholder( tf.float64, (self.input_size, 1), name='diff_lb' )
        self.diff_lp_ub = tf.placeholder( tf.float64, (self.input_size, 1), name='diff_ub' )
        
        self.mean = mean
        self.std = std

    def create_deeppoly_backsubs_conv( self, shape, W, b, stride_arg, padding_arg, tf_lbi, tf_ubi, inf, sup, inf_cst, sup_cst ):
        batch_size = tf.shape( inf )[ 0 ]
        deconv_shape = tf.stack( [ batch_size, *shape ] )

        deconv_inf = tf.nn.conv2d_transpose( inf, filter=W, output_shape=deconv_shape, strides=stride_arg, padding=padding_arg)
        deconv_sup = tf.nn.conv2d_transpose( sup, filter=W, output_shape=deconv_shape, strides=stride_arg, padding=padding_arg)

        mul = tf.tensordot( inf, b, axes=1 )
        reduce_dims =  list( range( 1, len ( mul.shape ) ) )
        deconv_inf_cst = inf_cst  + tf.reduce_sum( mul, reduce_dims )[ : , tf.newaxis ]
        mul = tf.tensordot( sup, b, axes=1 )
        deconv_sup_cst = sup_cst  + tf.reduce_sum( mul, reduce_dims )[ : , tf.newaxis ]

        if tf_lbi == None:
            return deconv_inf, deconv_sup, deconv_inf_cst, deconv_sup_cst
        else:
            deconv_inf_non_neg = tf.nn.relu( deconv_inf )
            deconv_inf_non_pos = -tf.nn.relu( -deconv_inf )

            deconv_sup_non_neg = tf.nn.relu( deconv_sup )
            deconv_sup_non_pos = -tf.nn.relu( -deconv_sup )

            relu_inf_non_neg, relu_inf_non_pos, _, relu_inf_cst = self.create_deeppoly_backsubs_relu( tf_lbi, tf_ubi, deconv_inf_non_neg, deconv_inf_non_pos, deconv_inf_cst, deconv_inf_cst )
            relu_inf = relu_inf_non_neg + relu_inf_non_pos
            relu_sup_non_pos, relu_sup_non_neg, _, relu_sup_cst = self.create_deeppoly_backsubs_relu( tf_lbi, tf_ubi, deconv_sup_non_pos, deconv_sup_non_neg, deconv_sup_cst, deconv_sup_cst )
            relu_sup = relu_sup_non_neg + relu_sup_non_pos
            return relu_inf, relu_sup, relu_inf_cst, relu_sup_cst

    def create_deeppoly_backsubs_ffn( self, W, b, tf_lbi, tf_ubi, inf, sup, inf_cst, sup_cst ):
        deconv_inf = tf.matmul( inf,  tf.transpose( W ) )
        deconv_sup = tf.matmul( sup, tf.transpose( W ) )

        mul = tf.tensordot( inf, b, axes=1 )
        reduce_dims =  list( range( 1, len ( mul.shape ) ) )
        deconv_inf_cst = inf_cst  + tf.reduce_sum( mul, reduce_dims )[ : , tf.newaxis ]
        mul = tf.tensordot( sup, b, axes=1 )
        deconv_sup_cst = sup_cst  + tf.reduce_sum( mul, reduce_dims )[ : , tf.newaxis ]

        if tf_lbi == None:
            return deconv_inf, deconv_sup, deconv_inf_cst, deconv_sup_cst
        else:
            deconv_inf_non_neg = tf.nn.relu( deconv_inf )
            deconv_inf_non_pos = -tf.nn.relu( -deconv_inf )

            deconv_sup_non_neg = tf.nn.relu( deconv_sup )
            deconv_sup_non_pos = -tf.nn.relu( -deconv_sup )

            relu_inf_non_neg, relu_inf_non_pos, _, relu_inf_cst = self.create_deeppoly_backsubs_relu( tf_lbi, tf_ubi, deconv_inf_non_neg, deconv_inf_non_pos, deconv_inf_cst, deconv_inf_cst )
            relu_inf = relu_inf_non_neg + relu_inf_non_pos
            relu_sup_non_pos, relu_sup_non_neg, _, relu_sup_cst = self.create_deeppoly_backsubs_relu( tf_lbi, tf_ubi, deconv_sup_non_pos, deconv_sup_non_neg, deconv_sup_cst, deconv_sup_cst )
            relu_sup = relu_sup_non_neg + relu_sup_non_pos
            return relu_inf, relu_sup, relu_inf_cst, relu_sup_cst

    def create_deeppoly_backsubs_relu( self, tf_lbi, tf_ubi, inf, sup, inf_cst, sup_cst ):
        affine_shape = tf.shape( inf )
        num_images = affine_shape[ 0 ]
        sampling_layers_size = tf.stack( ( num_images, -1 ), axis=0 )

        prev_layer_inf = tf.reshape( inf, shape=sampling_layers_size )
        prev_layer_sup = tf.reshape( sup, shape=sampling_layers_size )
        numel = tf.shape( prev_layer_inf )[ 1 ]

        # nub < 0 => y = 0
        tf_out_zeros_count = tf.math.count_nonzero( tf_ubi < 0, dtype=tf.int32 )
        tf_out_zeros_shape = tf.stack( ( num_images, tf_out_zeros_count ), axis=0 )
        tf_out_zeros = tf.zeros( shape=tf_out_zeros_shape, dtype=inf.dtype ) 

        # nlb >= 0 => y = x
        tf_out_x_inf = tf.gather( prev_layer_inf, tf.where( tf_lbi >= 0 )[ :, 0 ], axis=1 )
        tf_out_x_sup = tf.gather( prev_layer_sup, tf.where( tf_lbi >= 0 )[ :, 0 ], axis=1 )

        # remaining_idx calculations
        remaining_idx_full = tf.logical_and( tf_lbi < 0, tf_ubi >= 0 )
        remaining_idx = tf.where( remaining_idx_full )[ : , 0 ]
        tf_lbi_remaining = tf.gather( tf_lbi, remaining_idx )
        tf_ubi_remaining = tf.gather( tf_ubi, remaining_idx )
        prev_layer_inf = tf.gather( prev_layer_inf, remaining_idx, axis=1 )
        prev_layer_sup = tf.gather( prev_layer_sup, remaining_idx, axis=1 )

        slope = tf_ubi_remaining / ( tf_ubi_remaining - tf_lbi_remaining )
        intercept = -slope * tf_lbi_remaining
        
        reduce_dims = list( range( 1, len( prev_layer_sup.shape ) ) )
        sup_cst += tf.reduce_sum( prev_layer_sup * intercept[ tf.newaxis, : ], axis=reduce_dims )[ : , tf.newaxis ]

        # area decision:
        tf_ubi_abs = tf.abs( tf_ubi )
        tf_lbi_abs = tf.abs( tf_lbi )

        b1_eliminated_idx_full = tf_ubi_abs < tf_lbi_abs
        b1_eliminated_idx = tf.gather( b1_eliminated_idx_full, remaining_idx )
        b1_eliminated_idx = tf.where( b1_eliminated_idx )[ : , 0 ]

        b1_eliminated_output_inf = tf.zeros( shape=( tf.stack( ( num_images, numel ), axis=0 ) ), dtype=inf.dtype )
        b1_eliminated_output_inf = tf.gather( b1_eliminated_output_inf, b1_eliminated_idx, axis=1 )

        b3_eliminated_idx_full = tf_ubi_abs >= tf_lbi_abs
        b3_eliminated_idx = tf.gather( b3_eliminated_idx_full, remaining_idx )
        b3_eliminated_idx = tf.where( b3_eliminated_idx )[ : , 0 ]

        b3_eliminated_output_inf = prev_layer_inf
        b3_eliminated_output_inf = tf.gather( b3_eliminated_output_inf, b3_eliminated_idx, axis=1 )

        b2_output_sup = prev_layer_sup * slope 

        out_combined_inf = tf.concat( ( tf_out_zeros, tf_out_x_inf, b3_eliminated_output_inf, b1_eliminated_output_inf ), axis=1 )
        zero_idx = tf.where( tf_ubi < 0 ) [ : , 0 ]
        x_idx = tf.where( tf_lbi >= 0 ) [ : , 0 ]
        b3_elim_idx = tf.where( tf.logical_and( remaining_idx_full, b3_eliminated_idx_full ) ) [ : , 0 ]
        b1_elim_idx = tf.where( tf.logical_and( remaining_idx_full, b1_eliminated_idx_full ) ) [ : , 0 ]
        idx = tf.argsort( tf.concat( ( zero_idx, x_idx, b3_elim_idx, b1_elim_idx ), axis=0 ) )
        out_combined_reordered_inf = tf.gather( out_combined_inf, idx, axis=1 )
        out_combined_reordered_reshaped_inf = tf.reshape( out_combined_reordered_inf, shape=( affine_shape ) )

        out_combined_sup = tf.concat( ( tf_out_zeros, tf_out_x_sup, b2_output_sup ), axis=1 )
        idx = tf.argsort( tf.concat( ( zero_idx, x_idx, remaining_idx ), axis=0 ) )
        out_combined_reordered_sup = tf.gather( out_combined_sup, idx, axis=1 )
        out_combined_reordered_reshaped_sup = tf.reshape( out_combined_reordered_sup, shape=( affine_shape ) )

        return out_combined_reordered_reshaped_inf, out_combined_reordered_reshaped_sup, inf_cst, sup_cst

    def create_tf_layer( self, affine, prev_layer_attack, tf_attack_layer, relu_idx, use_deeppoly ):
        tf_lbi = tf.placeholder( affine.dtype, shape=[None], name='lbi_%i' % relu_idx )
        tf_ubi = tf.placeholder( affine.dtype, shape=[None], name='ubi_%i' % relu_idx )

        affine_shape = tf.shape( affine )
        num_images = affine_shape[ 0 ]
        sampling_layers_size = tf.stack( ( num_images, -1 ), axis=0 )

        prev_layer = tf.reshape( affine, shape=sampling_layers_size )
        # nub < 0 => y = 0
        tf_out_zeros_count = tf.math.count_nonzero( tf_ubi <= 0, dtype=tf.int32 )
        tf_out_zeros_shape = tf.stack( ( num_images, tf_out_zeros_count ), axis=0 )
        tf_out_zeros = tf.zeros( shape=tf_out_zeros_shape, dtype=affine.dtype )

        # nlb > 0 => y = x
        tf_out_x = tf.gather( prev_layer, tf.where( tf_lbi >= 0 )[ :, 0 ], axis=1 )

        # remaining_idx calculations
        remaining_idx_full = tf.logical_and( tf_lbi < 0, tf_ubi > 0 )
        remaining_idx = tf.where( remaining_idx_full )[ : , 0 ]
        tf_lbi_remaining = tf.gather( tf_lbi, remaining_idx )
        tf_ubi_remaining = tf.gather( tf_ubi, remaining_idx )
        prev_layer_attack = tf.gather( prev_layer_attack, remaining_idx )
        tf_attack_layer = tf.gather( tf_attack_layer, remaining_idx )
        prev_layer = tf.gather( prev_layer, remaining_idx, axis=1 )

        # remove y >= 0
        slope = tf_ubi_remaining / ( tf_ubi_remaining - tf_lbi_remaining )
        intercept = -slope * tf_lbi_remaining
        b2 = prev_layer_attack * slope + intercept
        b1 = prev_layer_attack
        b3_eliminated_output = tf.clip_by_value( tf_attack_layer, b1, b2 )

        region = b2 - b1
        low_dist = b3_eliminated_output - b1
        region = tf.clip_by_value( region, 1e-6, region )
        low_dist = tf.clip_by_value( low_dist, 5e-7, low_dist )
        high_dist = region - low_dist
        low_dist /= region
        high_dist /= region

        b2_output = prev_layer * slope + intercept
        b1_output = prev_layer
        b3_eliminated_output = low_dist * b2_output +  high_dist * b1_output

        # remove y >= x
        b1 = 0
        b1_eliminated_output = tf.clip_by_value( tf_attack_layer, b1, b2 )

        region = b2 - b1
        low_dist = b1_eliminated_output - b1
        region = tf.clip_by_value( region, 1e-6, region )
        low_dist = tf.clip_by_value( low_dist, 5e-7, low_dist )
        high_dist = region - low_dist
        low_dist /= region
        high_dist /= region

        b1_eliminated_output = low_dist * b2_output

        tf_ubi_abs = tf.abs( tf_ubi )
        tf_lbi_abs = tf.abs( tf_lbi )

        if use_deeppoly:
            # abs( ubi ) > abs( lbi ) => remove y >= 0 
            b3_eliminated_idx_full = tf_ubi_abs >= tf_lbi_abs
            b3_eliminated_idx = tf.gather( b3_eliminated_idx_full, remaining_idx )
        else:
            # b3 < b1 => remove y >= 0 
            b3_eliminated_idx = b1_output >= 0

        if use_deeppoly:
            # abs( ubi ) < abs( lbi ) => remove y >= x 
            b1_eliminated_idx_full = tf_ubi_abs < tf_lbi_abs
            b1_eliminated_idx = tf.gather( b1_eliminated_idx_full, remaining_idx )
        else:
            # b3 > b1 => remove y >= x
            b1_eliminated_idx = b1_output < 0

        remaining_output = tf.where( b1_eliminated_idx, b1_eliminated_output, b3_eliminated_output )

        out_combined = tf.concat( ( tf_out_zeros, tf_out_x, remaining_output ), axis=1 )
        zero_idx = tf.where( tf_ubi <= 0 ) [ : , 0 ]
        x_idx = tf.where( tf_lbi >= 0 ) [ : , 0 ]
        idx = tf.argsort( tf.concat( ( zero_idx, x_idx, remaining_idx ), axis=0 ) )
        out_combined_reordered = tf.gather( out_combined, idx, axis=1 )
        out_combined_reordered_reshaped = tf.reshape( out_combined_reordered, shape=( affine_shape ) )

        return out_combined_reordered_reshaped, out_combined_reordered, tf_lbi, tf_ubi

    def get_deeppoly_obj( self, target, nlb, nub, batch_size=300 ):
        feed_dict = {}

        np_layer_list = []
        np_layer_list_idx = []

        for i in range( len( self.tf_nlb ) ):
            feed_dict[ self.tf_nlb[ i ] ] = nlb[ i ]
            feed_dict[ self.tf_nub[ i ] ] = nub[ i ]
        
        fin_layer = len( self.tf_nlb )

        size = self.backsubstitute_tens[ fin_layer + 1 ][ 0 ].shape.as_list()[ 1 : ]
        layer_size_full = np.prod( size )
        recompute_idx = [ j for j in range( layer_size_full ) if j != target ]
        layer_size = len( recompute_idx )
        
        feed_dict[ self.backsubstitute_tens[ fin_layer + 1 ][ 2 ] ] = np.zeros( ( batch_size, 1 ) )
        feed_dict[ self.backsubstitute_tens[ fin_layer + 1 ][ 3 ] ] = np.zeros( ( batch_size, 1 ) )

        lb_layer = []
        ub_layer = []
        out = [ np.zeros( ( 0, self.input_size ) ),  np.zeros( ( 0, self.input_size ) ), np.zeros( ( 0, 1 ) ), np.zeros( ( 0, 1 ) ) ]
        j = -1
        for j in range( int( layer_size / batch_size ) ):
            eye_input = np.zeros( ( batch_size, layer_size_full ) )
            idx = recompute_idx[ j * batch_size : ( j + 1 ) * batch_size ]
            idx = [ tuple( np.arange(batch_size) ), tuple( idx ) ]
            eye_input[ idx ] = 1
            eye_input[ : , target ] = -1
            feed_dict[ self.backsubstitute_tens[ fin_layer + 1 ][ 0 ] ] = eye_input.reshape( [-1] + size )
            feed_dict[ self.backsubstitute_tens[ fin_layer + 1 ][ 1 ] ] = eye_input.reshape( [-1] + size )
            out_batch = self.sess.run( self.backsubstitute_tens[ 0 ], feed_dict=feed_dict )
            out_batch[ 0 ] = out_batch[ 0 ].reshape( out_batch[ 0 ].shape[ 0 ], -1 ) 
            out_batch[ 1 ] = out_batch[ 1 ].reshape( out_batch[ 1 ].shape[ 0 ], -1 )
            for k in range( 4 ):
                out[ k ] = np.concatenate( ( out[ k ], out_batch[ k ] ), axis=0 )
        j += 1
        if layer_size - j * batch_size > 0 :
            eye_input = np.zeros( ( layer_size - j * batch_size, layer_size_full ) )
            idx = recompute_idx[ j * batch_size : layer_size ]
            idx = [ tuple( np.arange( eye_input.shape[ 0 ] ) ), tuple( idx ) ]
            eye_input[ idx ] = 1
            eye_input[ : , target ] = -1
            eye_input = eye_input.reshape( [-1] + size )

            feed_dict[ self.backsubstitute_tens[ fin_layer + 1 ][ 0 ] ] = eye_input
            feed_dict[ self.backsubstitute_tens[ fin_layer + 1 ][ 1 ] ] = eye_input
            feed_dict[ self.backsubstitute_tens[ fin_layer + 1 ][ 2 ] ] = np.zeros( ( eye_input.shape[ 0 ], 1 ) )
            feed_dict[ self.backsubstitute_tens[ fin_layer + 1 ][ 3 ] ] = np.zeros( ( eye_input.shape[ 0 ], 1 ) )
            out_batch = self.sess.run( self.backsubstitute_tens[ 0 ], feed_dict=feed_dict )
            out_batch[ 0 ] = out_batch[ 0 ].reshape( out_batch[ 0 ].shape[ 0 ], -1 ) 
            out_batch[ 1 ] = out_batch[ 1 ].reshape( out_batch[ 1 ].shape[ 0 ], -1 )
            for k in range( 4 ):
                out[ k ] = np.concatenate( ( out[ k ], out_batch[ k ] ), axis=0 )
        
        obj = -out[ 1 ].T
        obj_cst = -out[ 3 ].T
        obj = np.concatenate( (obj_cst, obj) )

        return obj
 
    def extract_deeppoly_backsub( self, target, box=False, batch_size=300 ):
        feed_dict = {}
        ncpus = os.sysconf("SC_NPROCESSORS_ONLN")

        np_layer_list = []
        np_layer_list_idx = []
        
        #used_idxs_set = set()
        for i in range( len( self.tf_nlb ) + 1 ):
            is_final_layer = ( i == len( self.tf_nlb ) )
            s = time.time()
            if i != 0:
                feed_dict[ self.tf_nlb[ i - 1 ] ] = self.nlb[ i ]
                feed_dict[ self.tf_nub[ i - 1 ] ] = self.nub[ i ]

            size = self.backsubstitute_tens[ i + 1 ][ 0 ].shape.as_list()[ 1 : ]
            layer_size_full = np.prod( size )
            if is_final_layer:
                recompute_idx = [ j for j in range( layer_size_full ) if j != target ]
            else:
                recompute_idx = np.where( np.logical_and( self.nlb[ i + 1 ] < 0, self.nub[ i + 1 ] > 0 ) )[ 0 ]
            layer_size = len( recompute_idx )
            if layer_size == 0:
                np_layer_list.append( [np.zeros((0,self.input_size)), np.zeros((0,self.input_size)), np.zeros((0,1)),np.zeros((0,1))] )
                np_layer_list_idx.append( [] )
                continue
            
            feed_dict[ self.backsubstitute_tens[ i + 1 ][ 2 ] ] = np.zeros( ( batch_size, 1 ) )
            feed_dict[ self.backsubstitute_tens[ i + 1 ][ 3 ] ] = np.zeros( ( batch_size, 1 ) )

            lb_layer = []
            ub_layer = []
            out = [ np.zeros( ( 0, self.input_size ) ),  np.zeros( ( 0, self.input_size ) ), np.zeros( ( 0, 1 ) ), np.zeros( ( 0, 1 ) ) ]
            j = -1
            for j in range( int( layer_size / batch_size ) ):
                eye_input = np.zeros( ( batch_size, layer_size_full ) )
                idx = recompute_idx[ j * batch_size : ( j + 1 ) * batch_size ]
                idx = [ tuple( np.arange(batch_size) ), tuple( idx ) ]
                eye_input[ idx ] = 1
                if is_final_layer:
                    eye_input[ : , target ] = -1
                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 0 ] ] = eye_input.reshape( [-1] + size )
                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 1 ] ] = eye_input.reshape( [-1] + size )
                out_batch = self.sess.run( self.backsubstitute_tens[ 0 ], feed_dict=feed_dict )
                out_batch[ 0 ] = out_batch[ 0 ].reshape( out_batch[ 0 ].shape[ 0 ], -1 ) 
                out_batch[ 1 ] = out_batch[ 1 ].reshape( out_batch[ 1 ].shape[ 0 ], -1 )
                for k in range( 4 ):
                    out[ k ] = np.concatenate( ( out[ k ], out_batch[ k ] ), axis=0 )
            j += 1
            if layer_size - j * batch_size > 0 :
                eye_input = np.zeros( ( layer_size - j * batch_size, layer_size_full ) )
                idx = recompute_idx[ j * batch_size : layer_size ]
                idx = [ tuple( np.arange( eye_input.shape[ 0 ] ) ), tuple( idx ) ]
                eye_input[ idx ] = 1
                if is_final_layer:
                    eye_input[ : , target ] = -1
                eye_input = eye_input.reshape( [-1] + size )

                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 0 ] ] = eye_input
                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 1 ] ] = eye_input
                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 2 ] ] = np.zeros( ( eye_input.shape[ 0 ], 1 ) )
                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 3 ] ] = np.zeros( ( eye_input.shape[ 0 ], 1 ) )
                out_batch = self.sess.run( self.backsubstitute_tens[ 0 ], feed_dict=feed_dict )
                out_batch[ 0 ] = out_batch[ 0 ].reshape( out_batch[ 0 ].shape[ 0 ], -1 ) 
                out_batch[ 1 ] = out_batch[ 1 ].reshape( out_batch[ 1 ].shape[ 0 ], -1 )
                for k in range( 4 ):
                    out[ k ] = np.concatenate( ( out[ k ], out_batch[ k ] ), axis=0 )
            
            if not box:
                global global_model, global_eq, global_xs, global_get_example
                global_model = self.model
                global_eq = out
                
                np_layer_list.append( out )
                np_layer_list_idx.append( recompute_idx )

                global_get_example = is_final_layer
                with multiprocessing.Pool(ncpus) as pool:
                    solver_result = pool.map( pool_func_deeppoly, list( range ( layer_size ) ) )
                del globals()[ 'global_model' ]
                del globals()[ 'global_eq' ]
                del globals()[ 'global_get_example' ]

                if is_final_layer:
                    lbi, ubi, bad_exams = zip( *solver_result )
                    '''bad_exam_lb = np.array( bad_exams )[np.array(ubi) > 0]
                    bad_exam_lb = np.concatenate( ( bad_exam_lb, np.ones( ( bad_exam_lb.shape[0], 1 ) ) ), axis=1 )
                    out_lb = self.W @ bad_exam_lb.T
                    out_lb = np.where( np.abs( out_lb ) < 1e-6 )[ 0 ]
                    used_idxs = out_lb'''
                else:
                    lbi, ubi = zip( *solver_result )
                    '''lbi, ubi, bad_exam_lb, bad_exam_ub = zip( *solver_result )
                    bad_exam_lb = np.array( bad_exam_lb )
                    bad_exam_ub = np.array( bad_exam_ub )
                    bad_exam_lb = np.concatenate( ( bad_exam_lb, np.ones( ( bad_exam_lb.shape[0], 1 ) ) ), axis=1 )
                    bad_exam_ub = np.concatenate( ( bad_exam_ub, np.ones( ( bad_exam_ub.shape[0], 1 ) ) ), axis=1 )
                    out_lb = self.W @ bad_exam_lb.T
                    out_ub = self.W @ bad_exam_ub.T
                    out_lb = np.unique( np.where( np.abs( out_lb ) < 1e-6 )[ 0 ] )
                    out_ub = np.unique( np.where( np.abs( out_ub ) < 1e-6 )[ 0 ] )
                    used_idxs = np.concatenate( (out_lb, out_ub), axis=0 ) '''
            else:
                pos = np.maximum( out[0], 0 )
                neg = np.minimum( out[0], 0 )
                lbi = pos @ self.nlb[0] + neg @ self.nub[0] + out[ 2 ][ : , 0 ]
                
                pos = np.maximum( out[1], 0 )
                neg = np.minimum( out[1], 0 )
                ubi = pos @ self.nub[0] + neg @ self.nlb[0] + out[ 3 ][ : , 0 ]

                if is_final_layer:
                    pos[ pos > 0 ] = 1
                    neg[ neg < 0 ] = 1
                    bad_exams = pos * self.nub[0] + neg * self.nlb[0] 
 
            #used_idxs_set.update( used_idxs )

            lbi = np.array( lbi )
            ubi = np.array( ubi )
            
            if is_final_layer:

                #notused = set(range(self.W.shape[0])) - used_idxs_set
                #print( len(notused), notused )
 
                bad_class = np.argmax( ubi )
                smallestobj = -ubi[ bad_class ]
                bad_exam = bad_exams[ bad_class ]
                bad_exam_eq = -out[ 1 ][ bad_class ] 
                bad_exam_eq_cst = -out[ 3 ][ bad_class ]
                bad_exam_eq = ( bad_exam_eq, bad_exam_eq_cst )
                if bad_class >= target:
                    bad_class += 1
                if smallestobj >= 0:
                    print( 'Verified:', smallestobj )
                    return True

                in_vec = np.zeros( (1,self.output_size) )
                in_vec[0,bad_class] = 1
                in_vec[0,target] = -1
                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 0 ] ] = in_vec
                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 1 ] ] = in_vec
                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 2 ] ] = np.zeros( ( 1, 1 ) )
                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 3 ] ] = np.zeros( ( 1, 1 ) )
                grad_np = self.sess.run( self.tf_grad_deeppoly, feed_dict=feed_dict )

                hps = []
                if 'shrink_box_best' in dir(self):
                    for gs in range( len( grad_np ) ):
                        layer = gs % len( self.tf_nlb ) 
                        lb = gs // len( self.tf_nlb )
                        
                        idx = np.argsort(grad_np[gs][1])
                        grad_np_1 = grad_np[gs][1][idx]
                        grad_np_0 = grad_np[gs][0][idx]

                        mask = np.isin(np_layer_list_idx[ layer ], grad_np_1)
                        layer_ws = np_layer_list[ layer ][ lb ][ mask ]
                        layer_csts = np_layer_list[ layer ][ lb + 2 ][ mask ]
                        layers_combs = np.concatenate( (layer_csts.T, layer_ws.T) )
                        if lb == 0:
                            layers_combs *= -1
                        '''
                        idx = np.abs( self.nlb[layer+1] ) >= np.abs( self.nub[layer+1] )
                        if lb == 0:
                            idx = np.logical_not( idx )
                        idx = idx[ np_layer_list_idx[ layer ][ mask ] ]
                        hyper = self.get_hp_bias_under( layers_combs[:,idx] )
                        '''
                        hyper = self.get_hp_bias_under( layers_combs )
                        hps.append( (hyper[0], hyper[1]) )
                    obj = out[ 1 ].T
                    obj_cst = out[ 3 ].T + 1e-7
                    obj = np.concatenate( (obj_cst, obj) )
                    hyper = self.get_hp_bias_under( obj )
                    hps.append( (hyper[0], hyper[1]) )
                #return ( bad_exam_eq, bad_exam, bad_class, smallestobj, hps, notused )
                return ( bad_exam_eq, bad_exam, bad_class, smallestobj, hps )
            else:
                ubi[ ubi < lbi ] = self.nub[ i + 1 ][ recompute_idx ][ ubi < lbi ]
                lbi[ ubi < lbi ] = self.nlb[ i + 1 ][ recompute_idx ][ ubi < lbi ]
                self.nub[ i + 1 ][ recompute_idx ] = np.minimum( self.nub[ i + 1 ][ recompute_idx ], ubi )
                self.nlb[ i + 1 ][ recompute_idx ] = np.maximum( self.nlb[ i + 1 ][ recompute_idx ], lbi )

            for j in range( 4 ):
                del feed_dict[ self.backsubstitute_tens[ i + 1 ][ j ] ]
            print( 'Layer', i,':', layer_size,'/', layer_size_full, time.time() - s ,'secs' )
    
    def calc_tf_sampling_net( self, example, sample_batch, force=False ): 
        feed_dict = { self.tf_sampling_x: sample_batch, self.tf_attack: example }
        for i in range( len( self.tf_nlb ) ):
            feed_dict[ self.tf_nlb[ i ] ] = self.nlb[ i + 1 ]
            feed_dict[ self.tf_nub[ i ] ] = self.nub[ i + 1 ]
        out = self.sess.run( self.tf_sampling_layers, feed_dict=feed_dict )
        attack = np.concatenate( [ sample_batch ] + out, axis=1 )
        is_attack = np.logical_not( np.argmax( out[ -1 ], axis=1 ) == self.y_tar )
        if force:
            order = np.argsort( np.max( out[ -1 ], axis=1 )-out[ -1 ][:, self.y_tar ] )
            is_attack[:] = False
            is_attack[order[0:int(out[-1].shape[0]/10)]] = True
        #else:
        #    order = np.argsort( np.max( out[ -1 ], axis=1 )-out[ -1 ][:, self.y_tar ] )
        #    is_attack[:] = False
        #    is_attack[order[0:int(out[-1].shape[0]/5)]] = True
        return is_attack, attack

    def denorm_W( self, project, means, stds ):
        assert np.all( ( self.nub[ 0 ] - self.nlb[ 0 ] )[ project == 0 ] < 1e-3 )
        o = ( self.nub[ 0 ] + self.nlb[ 0 ] ) / 2.0 
        idx = np.where( project == 0 )[ 0 ].tolist() + [ self.input_size ]
        o = np.concatenate( ( o, np.ones( 1 ) ) )
        consts = np.matmul( self.W[:, idx],  o[ idx ] )
        idx = np.where( project == 1 )[ 0 ]
        W = self.W[ : , idx ] / stds[ idx ]
        consts -= np.matmul( self.W[ : , idx ], means[ idx ] / stds[ idx ] )
        consts = consts[ :, np.newaxis ]
        W = np.concatenate( ( W, consts ), axis=1 )
        lb = self.model_nlb[ idx ] * stds[ idx ] + means[ idx ]
        ub = self.model_nub[ idx ] * stds[ idx ] + means[ idx ]
        return W, lb, ub

    def tf_lp_sampling( self, attack, samples, vertype, batch_size=50, force=False ):
        ts = []
        i = -1
        for i in range( int( samples.shape[ 0 ] / batch_size ) ):
            t = samples[ i * batch_size : ( i + 1 ) * batch_size ]
            which,_ = self.calc_tf_sampling_net( attack, t, force )
            ts.append( t[ which ] )
        i = i + 1
        t = samples[ i * batch_size : ]
        which,_ = self.calc_tf_sampling_net( attack, t, force )
        ts.append( t[ which ] )
        ts = np.concatenate( ts, axis=0 )
        return ts

    def get_activation_pattern( self, pt ):
        eye = np.eye( self.input_size )
        eye = np.concatenate( ( np.zeros( ( 1, self.input_size ) ), eye ))
        act_lin = self.sess.run( self.activation_pattern_lin, feed_dict={self.activation_pattern_pt: pt.reshape(1,-1), self.activation_pattern_in: eye } )
        act_lin = [ a.reshape( self.input_size + 1, -1 ) for a in act_lin ]
        act_lin = np.concatenate( act_lin, axis=1 )
        act_lin[1:] -= act_lin[0:1]
        act = self.sess.run( self.activation_pattern_extract, feed_dict={self.activation_pattern_pt: pt.reshape(1,-1)} )
        act = [ a.reshape( 1, -1 ) for a in act ]
        act = np.concatenate( act, axis=1 )
        
        pt_ext = np.concatenate( ([1], pt ) )
        lin_out = np.matmul( pt_ext, act_lin )                                     
        assert np.max( np.abs( lin_out - act[0,:] ) ) < 1e-8
        
        last_valid = -self.output_size * self.output_size
        act_lin_pre = act_lin[ :, :last_valid ]
        act_lin_pre[ :, act[ 0, :last_valid ] >= 0.0 ] *= -1
        act_lin_post = act_lin[ :, last_valid + self.y_tar * self.output_size : last_valid + ( self.y_tar + 1 ) * self.output_size ]

        act_lin = np.concatenate( ( act_lin_pre, act_lin_post ), axis=1 ) 

        return act_lin

    def get_hp_bias_under( self, act_lin ):
        W = act_lin[ 1 : ].copy()
        b = act_lin[ 0 ].copy().reshape( -1, 1 )
        if self.shrink_box_best is None:
            return ( W, b )

        # contains underapprox 
        dev = ( self.shrink_box_best[1] - self.shrink_box_best[0] ) / 2.0
        mid = ( self.shrink_box_best[1] + self.shrink_box_best[0] ) / 2.0
        dev = dev.reshape(-1,1)
        mid = mid.reshape(-1,1)

        lp_cost_1 = np.sum( np.abs( W * dev ), axis=0, keepdims=True ).T
        lp_cost_2 = b + W.T @ mid
        lp_cost_under = ( lp_cost_1 + lp_cost_2 ).reshape( -1, 1 )
        b -= np.maximum( lp_cost_under + 1e-13, 0 )
        
        if self.p_c > 1e-6:
            out = [W.T.copy(), W.T.copy(), b.copy(), b.copy()]
            global global_model, global_eq, global_xs, global_get_example
            global_model = self.model
            global_eq = out
            global_get_example = False
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            with multiprocessing.Pool(ncpus) as pool:
                solver_result = pool.map( pool_func_deeppoly, list( range ( W.shape[1] ) ) )
            del globals()[ 'global_model' ]
            del globals()[ 'global_eq' ]
            del globals()[ 'global_get_example' ]
            bounds = np.array( [p[1] for p in solver_result] ).reshape(-1,1)
            b -= self.p_c*np.maximum( bounds + 1e-13, 0 )

        return ( W, b )

    def filter_activation_pattern( self, act_lin ):
        #act_lin = act_lin * 1000
        W = act_lin[ 1 : ]
        b = act_lin[ 0 ].reshape( -1, 1 )
        
        # contains underapprox 
        dev = ( self.shrink_box_best[1] - self.shrink_box_best[0] ) / 2.0
        mid = ( self.shrink_box_best[1] + self.shrink_box_best[0] ) / 2.0
        dev = dev.reshape(-1,1)
        mid = mid.reshape(-1,1)

        lp_cost_1 = np.sum( np.abs( W * dev ), axis=0, keepdims=True ).T
        lp_cost_2 = b + W.T @ mid
        lp_cost_under = ( lp_cost_1 + lp_cost_2 <= 0 ).reshape( -1 )
        
        '''
        m = Model('m')
        m.setParam( 'OutputFlag', 0 )
        vs = m.addVars( self.shrink_box_best[0].shape[ 0 ], lb=self.shrink_box_best[0], ub=self.shrink_box_best[1] )
        for i in range( W.shape[1] ):
            obj = sum( [vs[v] * W[ v, i ] for v in range(len(vs))] )
            m.setObjective( obj, GRB.MAXIMIZE )
            m.optimize()

            if lp_cost_under[i]:
                assert m.obj_bound + b[i] <= 0 
            else:
                assert m.obj_bound + b[i] > -1e-7
            if i % 500 == 0:
                print( i )
        '''

        # intersects overapprox
        dev = ( self.nub[0] - self.nlb[0] ) / 2.0
        mid = ( self.nub[0] + self.nlb[0] ) / 2.0
        dev = dev.reshape(-1,1)
        mid = mid.reshape(-1,1)

        lp_cost_1 = np.sum( np.abs( W * dev ), axis=0, keepdims=True ).T
        lp_cost_2 = b + W.T @ mid
        lp_cost_over = np.logical_and( -lp_cost_1 < lp_cost_2, lp_cost_2 < lp_cost_1 ).reshape( -1 )

        lp_cost = np.logical_and( lp_cost_over, lp_cost_under )
        act_lin = act_lin[ :, lp_cost ]
        '''
        m = Model('m')
        m.setParam( 'OutputFlag', 0 )
        vs = m.addVars( self.shrink_box_best[0].shape[ 0 ], lb=self.nlb[0], ub=self.nub[0] )
        for i in range( W.shape[1] ):
            obj = sum( [vs[v] * W[ v, i ] for v in range(len(vs))] )
            m.setObjective( obj, GRB.MAXIMIZE )
            m.optimize()
            max_obj = m.obj_bound + b[i]
            
            m.setObjective( obj, GRB.MINIMIZE )
            m.optimize()
            min_obj = m.obj_bound + b[i]
            if lp_cost_over[i]:
                if not ( ( min_obj < 1e-7 and max_obj > -1e-7 ) or ( min_obj > -1e-7 and max_obj < 1e-7 ) ):
                    import pdb; pdb.set_trace()
            else:
                if not ( ( min_obj < 1e-7 and max_obj < 1e-7 ) or ( min_obj > -1e-7 and max_obj > -1e-7 ) ):
                    import pdb; pdb.set_trace()

            if i % 500 == 0:
                print( i )

       
        import pdb; pdb.set_trace()
        '''
        return np.where(lp_cost)[0], act_lin

    def simplify_region( self ):
        Ws = self.W[:, :-1]
        bs = self.W[:, -1]
        model = Model()
        xs = []
        for i in range( self.input_size ):
            xs.append( model.addVar(lb=self.nlb[0][i], ub=self.nub[0][i], vtype=GRB.CONTINUOUS, name='x%i'%i) )
        
        constrs = Ws@xs
        for i in range( len( constrs ) ):
            model.addConstr( constrs[i] <= -bs[i] )
        model.update()
        model.setParam('OutputFlag', 0)
        model.setParam( 'DualReductions', 0 )

        tried_names = []
        j = 0
        dels = 0
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
            mc.reset()
            mc.optimize()
            if c.Sense == '<':
                if not mc.Status == 2:
                    import pdb; pdb.set_trace()
                if mc.objval <= rhs_orig:
                    model.remove(model.getConstrByName(c.ConstrName))
                    model.update()
                    dels += 1
                else:
                    tried_names.append( c.ConstrName )
            else:
                if mc.objval >= rhs_orig:
                    model.remove(model.getConstrByName(c.ConstrName))
                    model.update()
                    dels += 1
                else:
                    tried_names.append( c.ConstrName )
            del mc
            j += 1
            if j % 10 == 0:
                print( dels, '/', j )
    
        c_idx = [ int(c.ConstrName[1:]) for c in model.getConstrs() ]
        return xs, model, Ws[c_idx, :], bs[c_idx]

    def __del__( self ):
        del self.xs
        del self.model
