""" A Class for defining the ResNet model in Tensorflow. """
import tensorflow as tf

import numpy as np
import cv2
import matplotlib.pyplot as plt
import TerminalColors

class PlutoFlow:
    def __init__(self):
        """ Constructor basically defines the SGD and the variables """
        print 'PlutoFlow constructor'
        self.tcol = TerminalColors.bcolors

        # Define SGD on CPU
        with tf.device( '/cpu:0' ):
            self.opt = tf.train.AdamOptimizer(0.001)


        # Define all required variables (nightmare :'( )
        with tf.device( '/cpu:0' ):
            with tf.variable_scope( 'trainable_vars', reuse=None ):

                #top-level conv
                wc_top = tf.get_variable( 'wc_top', [7,7,3,64], initializer=tf.contrib.layers.xavier_initializer_conv2d() )
                bc_top = tf.get_variable( 'bc_top', [64], initializer=tf.constant_initializer(0.01) )



                ## RES2
                with tf.variable_scope( 'res2a', reuse=None ):
                    self._define_resnet_unit_var( 64, [64,64,256], [1,3,1], False )

                with tf.variable_scope( 'res2b', reuse=None ):
                    self._define_resnet_unit_var( 256, [64,64,256], [1,3,1], True )

                with tf.variable_scope( 'res2c', reuse=None ):
                    self._define_resnet_unit_var( 256, [64,64,256], [1,3,1], True )


                ## RES3
                with tf.variable_scope( 'res3a', reuse=None ):
                    self._define_resnet_unit_var( 256, [128,128,512], [1,3,1], False )

                with tf.variable_scope( 'res3b', reuse=None ):
                    self._define_resnet_unit_var( 512, [128,128,512], [1,3,1], True )

                with tf.variable_scope( 'res3c', reuse=None ):
                    self._define_resnet_unit_var( 512, [128,128,512], [1,3,1], True )

                with tf.variable_scope( 'res3d', reuse=None ):
                    self._define_resnet_unit_var( 512, [128,128,512], [1,3,1], True )

                ## RES4
                with tf.variable_scope( 'res4a', reuse=None ):
                    self._define_resnet_unit_var( 512, [256,256,1024], [1,3,1], False )

                with tf.variable_scope( 'res4b', reuse=None ):
                    self._define_resnet_unit_var( 1024, [256,256,1024], [1,3,1], True )

                with tf.variable_scope( 'res4c', reuse=None ):
                    self._define_resnet_unit_var( 1024, [256,256,1024], [1,3,1], True )

                with tf.variable_scope( 'res4d', reuse=None ):
                    self._define_resnet_unit_var( 1024, [256,256,1024], [1,3,1], True )

                with tf.variable_scope( 'res4e', reuse=None ):
                    self._define_resnet_unit_var( 1024, [256,256,1024], [1,3,1], True )

                ## RES5
                with tf.variable_scope( 'res5a', reuse=None ):
                    self._define_resnet_unit_var( 1024, [512,512,2048], [1,3,1], False )

                with tf.variable_scope( 'res5b', reuse=None ):
                    self._define_resnet_unit_var( 2048, [512,512,2048], [1,3,1], True )

                with tf.variable_scope( 'res5c', reuse=None ):
                    self._define_resnet_unit_var( 2048, [512,512,2048], [1,3,1], True )



                ## Fully Connected Layer
                # NOTE: This will make each layer identical, if need be in the
                # future this can be changed to have different structures for
                # prediction variables.
                with tf.variable_scope( 'fully_connected', reuse=None ):
                    for scope_str in ['x', 'y', 'z', 'yaw']:
                        with tf.variable_scope( scope_str, reuse=None ):
                            w_fc1 = tf.get_variable( 'w_fc1', [40960, 2000])
                            w_fc2 = tf.get_variable( 'w_fc2', [2000, 200])
                            w_fc3 = tf.get_variable( 'w_fc3', [200, 20])
                            w_fc4 = tf.get_variable( 'w_fc4', [20, 1])

                            #bias terms
                            b_fc1 = tf.get_variable( 'b_fc1', [ 2000])
                            b_fc2 = tf.get_variable( 'b_fc2', [ 200])
                            b_fc3 = tf.get_variable( 'b_fc3', [ 20])
                            b_fc4 = tf.get_variable( 'b_fc4', [ 1])


        # Place the towers on each of the GPUs and compute ops for
        # fwd_flow, avg_gradient and update_variables

        print 'Exit successfully, from PlutoFlow constructor'



    def resnet50_inference(self, x):
        """ This function creates the computational graph and returns the op which give a
            prediction given an input batch x
        """

        print 'Define ResNet50'

        with tf.variable_scope( 'trainable_vars', reuse=True ):
            wc_top = tf.get_variable( 'wc_top', [7,7,3,64] )
            bc_top = tf.get_variable( 'bc_top', [64] )



            conv1 = self._conv2d( x, wc_top, bc_top, strides=2 )
            conv1 = self._maxpool2d( conv1, k=2 )

            input_var = conv1

            ## RES2
            with tf.variable_scope( 'res2a', reuse=True ):
                conv_out = self.resnet_unit( input_var, 64, [64,64,256], [1,3,1], short_circuit=False )

            with tf.variable_scope( 'res2b', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 256, [64,64,256], [1,3,1], short_circuit=True )

            with tf.variable_scope( 'res2c', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 256, [64,64,256], [1,3,1], short_circuit=True )

                ## MAXPOOL
                conv_out = self._maxpool2d( conv_out, k=2 )


            ## RES3
            with tf.variable_scope( 'res3a', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 256, [128,128,512], [1,3,1], short_circuit=False )

            with tf.variable_scope( 'res3b', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 512, [128,128,512], [1,3,1], short_circuit=True )

            with tf.variable_scope( 'res3c', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 512, [128,128,512], [1,3,1], short_circuit=True )

            with tf.variable_scope( 'res3d', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 512, [128,128,512], [1,3,1], short_circuit=True )

                ## MAXPOOL
                conv_out = self._maxpool2d( conv_out, k=2 )


            ## RES4
            with tf.variable_scope( 'res4a', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 512, [256,256,1024], [1,3,1], short_circuit=False )

            with tf.variable_scope( 'res4b', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 1024, [256,256,1024], [1,3,1], short_circuit=True )

            with tf.variable_scope( 'res4c', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 1024, [256,256,1024], [1,3,1], short_circuit=True )

            with tf.variable_scope( 'res4d', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 1024, [256,256,1024], [1,3,1], short_circuit=True )

            with tf.variable_scope( 'res4e', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 1024, [256,256,1024], [1,3,1], short_circuit=True )

                ## MAXPOOL
                conv_out = self._maxpool2d( conv_out, k=2 )


            ## RES5
            with tf.variable_scope( 'res5a', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 1024, [512,512,2048], [1,3,1], short_circuit=False )

            with tf.variable_scope( 'res5b', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 2048, [512,512,2048], [1,3,1], short_circuit=True )

            with tf.variable_scope( 'res5c', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 2048, [512,512,2048], [1,3,1], short_circuit=True )

                ## MAXPOOL
                conv_out = self._maxpool2d( conv_out, k=2 )


            # Reshape Activations
            print conv_out.get_shape().as_list()[1:]
            n_activations = np.prod( conv_out.get_shape().as_list()[1:] )
            fc = tf.reshape( conv_out, [-1, n_activations] )

            # Fully Connected Layers
            predictions = []
            with tf.variable_scope( 'fully_connected', reuse=True):
                for scope_str in ['x', 'y', 'z', 'yaw']:
                    with tf.variable_scope( scope_str, reuse=True ):
                        pred = self._define_fc( fc, n_activations, [2000, 200, 20, 1] )
                        print pred
                        predictions.append( pred )



            # Weighted additions of the predictions
            resnet_out = tf.mul( predictions[0], 1. ) + tf.mul( predictions[1], 1. ) + tf.mul( predictions[2], 1. ) + tf.mul( predictions[3], .5 )

            return resnet_out





    def _print_tensor_info( self, display_str, T ):
        print self.tcol.WARNING, display_str, T.name, T.get_shape().as_list(), self.tcol.ENDC


    def resnet_unit( self, input_tensor, n_inputs, n_intermediates, intermediate_filter_size, short_circuit=True ):
        """ Defines the net structure of resnet unit
                input_tensor : Input of the unit
                n_inputs : Number of input channels
                n_intermediates : An array of intermediate filter outputs (usually len of this array is 3)
                intermediate_filter_size : Same size as `n_intermediates`, gives kernel sizes (sually 1,3,1)
                short_circuit : True will directly connect input to the (+-elementwise). False will add in a convolution before adding

                returns : output of the unit. after addition and relu

        """
        print '<--->'
        a = n_inputs
        b = n_intermediates #note, b[2] will be # of output filters
        c = intermediate_filter_size

        self._print_tensor_info( 'Input Tensor', input_tensor)

        wc1 = tf.get_variable( 'wc1', [c[0],c[0],a,b[0] ] )
        wc2 = tf.get_variable( 'wc2', [c[1],c[1],b[0],b[1]] )
        wc3 = tf.get_variable( 'wc3', [c[2],c[2],b[1],b[2]] )

        self._print_tensor_info( 'Request Var', wc1 )
        self._print_tensor_info( 'Request Var', wc2 )
        self._print_tensor_info( 'Request Var', wc3 )


        conv_1 = self._conv2d_nobias( input_tensor, wc1 )
        conv_2 = self._conv2d_nobias( conv_1, wc2 )
        conv_3 = self._conv2d_nobias( conv_2, wc3 )
        self._print_tensor_info( 'conv_1', conv_1 )
        self._print_tensor_info( 'conv_2', conv_2 )
        self._print_tensor_info( 'conv_3', conv_3 )

        if short_circuit==True: #direct skip connection (no conv on side)
            conv_out = tf.nn.relu( tf.add( conv_3, input_tensor ) )
        else: #side connection has convolution
            wc_side = tf.get_variable( 'wc1_side', [1,1,a,b[2]] )
            self._print_tensor_info( 'Request Var', wc_side )
            conv_side = self._conv2d_nobias( input_tensor, wc_side, relu_unit=False )
            conv_out = tf.nn.relu( tf.add( conv_3, conv_side ) )

        self._print_tensor_info( 'conv_out', conv_out )
        return conv_out


    def _define_resnet_unit_var( self, n_inputs, n_intermediates, intermediate_filter_size, short_circuit=True ):
        """ Defines variables in a resnet unit
                n_inputs : Number of input channels
                n_intermediates : An array of intermediate filter outputs (usually len of this array is 3)
                intermediate_filter_size : Same size as `n_intermediates`, gives kernel sizes (sually 1,3,1)
                short_circuit : True will directly connect input to the (+-elementwise). False will add in a convolution before adding

        """
        a = n_inputs
        b = n_intermediates #note, b[2] will be # of output filters
        c = intermediate_filter_size
        wc1 = tf.get_variable( 'wc1', [c[0],c[0],a,b[0]], initializer=tf.contrib.layers.xavier_initializer_conv2d() )
        wc2 = tf.get_variable( 'wc2', [c[1],c[1],b[0],b[1]], initializer=tf.contrib.layers.xavier_initializer_conv2d() )
        wc3 = tf.get_variable( 'wc3', [c[2],c[2],b[1],b[2]], initializer=tf.contrib.layers.xavier_initializer_conv2d() )

        if short_circuit == False:
            wc_side = tf.get_variable( 'wc1_side', [1,1,a,b[2]], initializer=tf.contrib.layers.xavier_initializer_conv2d() )


    # Create some wrappers for simplicity
    def _conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation

        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)

        # NORMPROP
        # return (tf.nn.relu(x) - 0.039894228) / 0.58381937
        return tf.nn.relu(x)

    def _define_fc( self, fc, n_input, interim_input_dim ):
        """ Define a fully connected layer
                fc : the reshaped array
                n_input : number of inputs
                interim_input_dim : array of intermediate data dims

                Note that this assume, the context is already in correct scope
        """

        a = n_input
        b = interim_input_dim
        w_fc1 = tf.get_variable( 'w_fc1', [a, b[0]])
        w_fc2 = tf.get_variable( 'w_fc2', [b[0], b[1]])
        w_fc3 = tf.get_variable( 'w_fc3', [b[1], b[2]])
        w_fc4 = tf.get_variable( 'w_fc4', [b[2], 1])

        b_fc1 = tf.get_variable( 'b_fc1', [ b[0]])
        b_fc2 = tf.get_variable( 'b_fc2', [ b[1]])
        b_fc3 = tf.get_variable( 'b_fc3', [ b[2]])
        b_fc4 = tf.get_variable( 'b_fc4', [ 1])

        fc1 = tf.nn.relu( tf.add( tf.matmul( fc, w_fc1 ), b_fc1 ) )
        fc2 = tf.nn.relu( tf.add( tf.matmul( fc1, w_fc2 ), b_fc2 ) )
        fc3 = tf.nn.relu( tf.add( tf.matmul( fc2, w_fc3 ), b_fc3 ) )
        fc4 = tf.add( tf.matmul( fc3, w_fc4 ), b_fc4, name='stacked_fc_out' )
        return fc4



    # Create some wrappers for simplicity
    def _conv2d_nobias(self, x, W, strides=1, relu_unit=True):
        # Conv2D wrapper, with bias and relu activation

        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')

        # NORMPROP
        # return (tf.nn.relu(x) - 0.039894228) / 0.58381937

        if relu_unit == True:
            return tf.nn.relu(x)
        else:
            # NOTE : No RELU
            return x


    def _maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        pool_out = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')
        # NORMPROP
        # return (pool_out - 1.4850) / 0.7010
        return pool_out
