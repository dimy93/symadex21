import numpy as np
from gurobipy import *
import tensorflow as tf

'''
class NN_bias_optimizer:
    def __init__(self, layers):
        for l 
'''

class LP_Layer:
    
    def __init__(self, *args):

        if len( args ) == 8:
            sess, lb, ub, W, b, f, f_const, minimize = args
            self.canon = False
        else:
            if len( args ) == 5:
                sess, W, b, f, f_const = args
                minimize = False
            else:
                sess, W, b, f, f_const, minimize = args

            self.canon = True
            lb = tf.zeros( f.shape )
            ub = tf.fill(f.shape, np.inf)

        assert len( lb.shape ) == 2, 'LB is not 2D'
        assert len( b.shape ) == 2, 'b is not 2D'
        assert len( f.shape ) == 2, 'f is not 2D'
        assert len( f_const.shape ) == 0, 'f_const is not 0D'
        assert len( W.shape ) == 2, 'W is not 2D'
        assert lb.shape == ub.shape, 'LB and UB don\'t match'
        assert f.shape == ub.shape, 'f doesn\'t match X'
        assert f.shape[1] == 1, 'f second dimention must be 1'
        assert b.shape[0] == W.shape[0], 'b doesn\'t match W'

        self.lb_tf = lb
        self.ub_tf = ub
        self.W_tf = W
        self.b_tf = b
        self.f_tf = f
        self.f_const_tf = f_const
        self.sess = sess
        self.minimize = minimize
        
        # Idx pplaceholders - change each iteration
        self.lb_idx_tf = tf.placeholder(dtype=tf.int32, shape=(None))
        self.ub_idx_tf = tf.placeholder(dtype=tf.int32, shape=(None))
        self.W_idx_tf = tf.placeholder(dtype=tf.int32, shape=(None))
        
        # Create simplex matrices
        eye_tf = tf.eye(tf.shape(self.lb_tf)[0], dtype=tf.float32)
        eye_idx_tf = tf.concat( ( self.lb_idx_tf, self.ub_idx_tf ), axis=0 )
        W_simplex = tf.gather( eye_tf, eye_idx_tf, axis=0 )
        self.W_simplex_tf = tf.concat( ( W_simplex, tf.gather( self.W_tf, self.W_idx_tf, axis=0 )), axis=0 )
        
        lb_simplex = tf.gather( self.lb_tf, self.lb_idx_tf, axis=0 )
        ub_simplex = tf.gather( self.ub_tf, self.ub_idx_tf, axis=0 )
        b_simplex = tf.gather( self.b_tf, self.W_idx_tf, axis=0 )
        self.b_simplex_tf = tf.concat( ( lb_simplex, ub_simplex, b_simplex ), axis=0 )

        # Create result
        self.x_tf = tf.linalg.solve( self.W_simplex_tf, self.b_simplex_tf )
        self.obj_tf = tf.reduce_sum( self.f_tf * self.x_tf ) + self.f_const_tf

        if self.canon:
            self.create_dual_sol_ten()

    def get_canon(self):
        eye_tf = tf.eye(tf.shape(self.lb_tf)[0], dtype=tf.float32)
        W_canon_tf = tf.concat( (self.W_tf, eye_tf), axis=0 )
        b_canon_tf = tf.concat( (self.b_tf - self.W_tf @ self.lb_tf, self.ub_tf-self.lb_tf), axis=0 )

        f_const_canon_tf = self.f_const_tf + tf.reduce_sum( self.f_tf * self.lb_tf )
        if self.minimize:
            f_canon_tf = -self.f_tf
            f_const_canon_tf *= -1
        else:
            f_canon_tf = self.f_tf

        canon = LP_Layer( self.sess, W_canon_tf, b_canon_tf, f_canon_tf, f_const_canon_tf )

        if self.minimize:
            canon.obj_tf *= -1
            canon.obj_dual_tf *= -1

        return canon

    def get_dual(self):
        assert not self.canon 
        eye_tf = tf.eye(tf.shape(self.lb_tf)[0], dtype=tf.float32)
        W_canon_tf = tf.concat( (self.W_tf, eye_tf), axis=0 )
        b_canon_tf = tf.concat( (self.b_tf - self.W_tf @ self.lb_tf, self.ub_tf-self.lb_tf), axis=0 )

        f_const_canon_tf = self.f_const_tf + tf.reduce_sum( self.f_tf * self.lb_tf ) 
        if self.minimize:
            f_canon_tf = -self.f_tf
            f_const_canon_tf *= -1
        else:
            f_canon_tf = self.f_tf

        W_dual_tf = -tf.transpose( W_canon_tf )
        b_dual_tf = -f_canon_tf
        f_dual_tf = b_canon_tf
        
        dual = LP_Layer( self.sess, W_dual_tf, b_dual_tf, f_dual_tf, f_const_canon_tf, True )

        if self.minimize:
            dual.obj_tf *= -1
            dual.obj_dual_tf *= -1

        dual.y_tf *= -1
        return dual

    def create_dual_sol_ten(self):
        assert self.canon

        # Dual sat constraints
        idx = tf.concat( (self.x_tf, self.b_tf - self.W_tf @ self.x_tf), axis=0 )
        idx = tf.argsort( idx, direction='DESCENDING', axis=0 )
        idx = idx[ : tf.shape(self.W_tf)[0] ]
        idx1 = tf.where( idx < tf.shape(self.x_tf)[0] )[:,0]
        idx2 = tf.where( idx >= tf.shape(self.x_tf)[0] )[:,0]
        idx1 = tf.gather( idx, idx1, axis=0 )
        idx2 = tf.gather( idx, idx2, axis=0 ) - tf.shape(self.x_tf)[0]
        idx1 = idx1[:,0]
        idx2 = idx2[:,0]

        W_dual_simplex_tf = tf.gather( self.W_tf, idx1, axis=1 )
        W_dual_simplex_tf = tf.transpose( W_dual_simplex_tf )
        eye_tf = tf.eye( tf.shape(self.W_tf)[0], dtype=tf.float32 )
        W_dual_simplex_2_tf = tf.gather( eye_tf, idx2, axis=0 )  
        self.W_dual_simplex_tf = tf.concat( (W_dual_simplex_tf, W_dual_simplex_2_tf), axis=0 )
        
        b_dual_simplex_tf = tf.gather( self.f_tf, idx1, axis=0 )
        b_dual_simplex_2_tf = tf.zeros( ( tf.shape(W_dual_simplex_2_tf)[0], 1 ) ) 
        self.b_dual_simplex_tf = tf.concat( (b_dual_simplex_tf, b_dual_simplex_2_tf), axis=0 )
        
        # Dual solution
        self.y_tf = tf.linalg.solve( self.W_dual_simplex_tf, self.b_dual_simplex_tf )
        self.obj_dual_tf = tf.reduce_sum( self.b_tf * self.y_tf ) + self.f_const_tf
        
        self.gap_tf = self.obj_dual_tf - self.obj_tf

    def get_new_feed_dict(self, lb_idx, ub_idx, W_idx, feed_dict={}):
        new_dict = { self.lb_idx_tf: lb_idx, self.ub_idx_tf: ub_idx, self.W_idx_tf: W_idx }
        new_dict = { k: new_dict.get(k) if k in new_dict else feed_dict.get( k ) for k in set( new_dict ) | set( feed_dict ) }
        return new_dict

    def get_constr_idx(self, feed_dict={}):
        tens = [self.lb_tf, self.ub_tf, self.W_tf, self.b_tf, self.f_tf]
        out = self.sess.run( tens, feed_dict=feed_dict )
        [ lb_np, ub_np, W_np, b_np, f_np ] = out
        lb_np = lb_np[:, 0]
        ub_np = ub_np[:, 0]
        f_np = f_np[:, 0]
        b_np = b_np[:, 0]

        m = Model()
        m.setParam('OutputFlag', 0)
        xs = []
        for i in range( lb_np.shape[0] ):
            x = m.addVar( lb= -GRB.INFINITY if np.isinf( lb_np[i] ) else lb_np[i], \
                          ub=  GRB.INFINITY if np.isinf( ub_np[i] ) else ub_np[i], \
                          name='x%d' % i )
            xs.append( x )
        m.update()
        
        constrs = W_np @ xs
        for i in range( len ( constrs ) ):
            m.addConstr( constrs[i] <= b_np[i], name='w%d' % i )
        m.setObjective( f_np @ xs, GRB.MINIMIZE if self.minimize else GRB.MAXIMIZE )
        # TODO: Load .bas
        m.optimize()
        assert m.Status == 2

        x_vals = np.array( [ x.x for x in xs ] )
    
        # Find simplex satisfied constraints
        constr_eval = []
        constr_eval.append( lb_np - x_vals )
        constr_eval.append( ub_np - x_vals )
        for i in range( len( constrs ) ): 
            constr_eval.append( [constrs[i].getValue() - b_np[i]] )
        constr_eval = np.concatenate( constr_eval, axis=0 )
        constr_eval = np.abs( constr_eval )
        idx = np.argsort( constr_eval )[ : lb_np.shape[0] ]
        lb_idx = idx[ idx < lb_np.shape[0] ]
        ub_idx = idx[ np.logical_and( idx < lb_np.shape[0] + ub_np.shape[0], idx >= lb_np.shape[0] ) ] - lb_np.shape[0]
        W_idx = idx[ idx >= lb_np.shape[0] + ub_np.shape[0] ] - lb_np.shape[0] - ub_np.shape[0]
        
        #x_np, obj_np = self.sess.run( (self.x_tf, self.obj_tf), feed_dict=self.get_new_feed_dict(lb_idx, ub_idx, W_idx, feed_dict) ) 
   
        #assert np.all( np.abs( x_vals - x_np[:,0] ) < 1e-5 )
        #assert np.abs( obj_np - m.objval ) < 1e-5

        return lb_idx, ub_idx, W_idx
