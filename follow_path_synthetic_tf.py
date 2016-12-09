""" The synthetic test-render file. (class TestRenderer)
        This script defines a class that renders images along a set
        trajectory. This trajectory is defined as a spline with fixed
        nodes. The rendered images are used for prediction from a previously
        trained model. Further this stores a file with GT and predicted
        positions at every rendered image.
"""

# Panda3d
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from direct.stdpy import thread

# Usual Math and Image Processing
import numpy as np
import cv2
# import caffe
import tensorflow as tf
from scipy import interpolate


# Other System Libs
import os
import argparse
import Queue
import copy
import time
import code


# Misc
import TerminalColors
import CubeMaker
import PathMaker
import PlutoFlow as puf



class TestRenderer(ShowBase):
    renderIndx=0


    # Given a square matrix, substract mean and divide std dev
    def zNormalized(self, M ):
        M_mean = np.mean(M) # scalar
        M_std = np.std(M)
        M_statSquash = (M - M_mean)/M_std
        return M_statSquash

    # Basic Mesh & Camera Setup
    def loadAllTextures(self, mesh, basePath, silent=True):
        """ Loads texture files for a mesh """
        c = 0
        for child in mesh.getChildren():
            submesh_name = child.get_name()
            submesh_texture = basePath + submesh_name[:-5] + 'tex0.jpg'
            child.setTexture( self.loader.loadTexture(submesh_texture) )

            if silent == False:
                print 'Loading texture file : ', submesh_texture
            c = c + 1

        print self.tcolor.OKGREEN, "Loaded ", c, "textures", self.tcolor.ENDC
    def setupMesh(self):
        """ Loads the .obj files. Will load mesh sub-divisions separately """

        print 'Attempt Loading Mesh VErtices, FAces'
        self.cyt = self.loader.loadModel( 'model_l/l6/level_6_0_0.obj' )
        self.cyt2 = self.loader.loadModel( 'model_l/l6/level_6_128_0.obj' )

        self.low_res = self.loader.loadModel( 'model_l/l0/level_0_0_0.obj' )
        print self.tcolor.OKGREEN, 'Done Loading Vertices', self.tcolor.ENDC

        print 'Attempt Loading Textures'
        self.loadAllTextures( self.cyt, 'model_l/l6/')
        self.loadAllTextures( self.cyt2, 'model_l/l6/')
        self.loadAllTextures( self.low_res, 'model_l/l0/')
        print self.tcolor.OKGREEN, 'Done Loading Textures', self.tcolor.ENDC

    def positionMesh(self):
        """ WIll have to manually adjust this for ur mesh. I position the
        center where I fly my drone and oriented in ENU (East-north-up)
        cords for easy alignment of GPS and my cordinates. If your model
        is not metric scale will have to adjust for that too"""

        self.cyt.setPos( 140,-450, 150 )
        self.cyt2.setPos( 140,-450, 150 )
        self.low_res.setPos( 140,-450, 150 )
        self.cyt.setHpr( 198, -90, 0 )
        self.cyt2.setHpr( 198, -90, 0 )
        self.low_res.setHpr( 198, -90, 0 )
        self.cyt.setScale(172)
        self.cyt2.setScale(172)
        self.low_res.setScale(172)


    def customCamera(self, nameIndx):
        lens = self.camLens
        lens.setFov(83)
        print 'self.customCamera : Set FOV at 83'
        my_cam = Camera("cam"+nameIndx, lens)
        my_camera = self.scene0.attachNewNode(my_cam)
        # my_camera = self.render.attachNewNode(my_cam)
        my_camera.setName("camera"+nameIndx)
        return my_camera


    def customDisplayRegion(self, rows, cols):
        rSize = 1.0 / rows
        cSize = 1.0 / cols

        dr_list = []
        for i in range(0,rows):
            for j in range(0,cols):
                # print i*rSize, (i+1)*rSize, j*cSize, (j+1)*cSize
                dr_i = self.win2.makeDisplayRegion(i*rSize, (i+1)*rSize, j*cSize, (j+1)*cSize)
                dr_i.setSort(-5)
                dr_list.append( dr_i )
        return dr_list

    def test_iteration(self):
        """ Retrive an image from the queue and feed it into the testing network.
            Returns [GT, Pred, Loss]
        """
        # print 'DoCaffeTrainng'
        startTime = time.time()
        # print 'q_imStack.qsize() : ', self.q_imStack.qsize()
        # print 'q_labelStack.qsize() : ', self.q_labelStack.qsize()


        # if too few items in queue do not proceed with iterations
        if self.q_imStack.qsize() < 2:
            return None, None, None


        # Retrive from  the queue
        im = self.q_imStack.get() #320x240x3
        y = self.q_labelStack.get()

        im_gry = np.mean( im, axis=2)
        im_statSquash = self.zNormalized( im.astype('float32') )


        # cv2.imwrite( 'tf.logs/'+str(TestRenderer.renderIndx)+'.png', im.astype('uint8') )

        # tensorflow infer
        im_batch = np.zeros((1,240,320,3))
        im_batch[0,:,:,0] = self.zNormalized( im[:,:,0] )
        im_batch[0,:,:,1] = self.zNormalized( im[:,:,1] )
        im_batch[0,:,:,2] = self.zNormalized( im[:,:,2] )


        # session.run() for inference
        aa_out = self.tensorflow_session.run( [self.tf_infer_op], feed_dict={ self.tf_x:im_batch } ) #1x4x12x1
        aa_out = aa_out[0]
        pr_x = aa_out[0][0][0]
        pr_y = aa_out[1][0][0]
        pr_z = aa_out[2][0][0]
        pr_yaw = aa_out[3][0][0]


        # code.interact(local=locals())


        GT = [ y[0], y[1], y[2], y[3] ]
        # PRED = [0,0,0,0]
        PRED = [pr_x,pr_y,pr_z,pr_yaw]
        LOSS = [0,0,0,0]
        print 'GT   : %2.4f %2.4f %2.4f %2.4f' %(GT[0], GT[1], GT[2], GT[3])
        print 'Pr   : %2.4f %2.4f %2.4f %2.4f' %(PRED[0], PRED[1], PRED[2], PRED[3])
        # print 'loss : ', LOSS
        print 'time = %2.4f mili-sec' %((time.time() - startTime)*1000.)


        return PRED,GT, LOSS



    def monte_carlo_sample(self):
        """ Gives a random 6-dof pose. Need to set params manually here.
                X,Y,Z,  Yaw(abt Z-axis), Pitch(abt X-axis), Roll(abt Y-axis) """
        X = np.random.uniform(-50,50)
        Y = np.random.uniform(-100,100)
        Z = np.random.uniform(50,100)

        yaw = np.random.uniform(-60,60)
        roll = np.random.uniform(-5,5)
        pitch = np.random.uniform(-5,5)

        return X,Y,Z, yaw,roll,pitch

    # Annotation-helpers for self.render
    def putBoxes(self,X,Y,Z,r=1.,g=0.,b=0., scale=1.0):
        cube_x = CubeMaker.CubeMaker().generate()
        cube_x.setColor(r,g,b)
        cube_x.setScale(scale)
        cube_x.reparentTo(self.render)
        cube_x.setPos(X,Y,Z)
    def putAxesTask(self,task):


        cube_x = CubeMaker.CubeMaker().generate()
        cube_x.setColor(1.0,0.0,0.0)
        cube_x.setScale(1)
        cube_x.reparentTo(self.render)
        cube_x.setPos(task.frame,0,0)

        cube_y = CubeMaker.CubeMaker().generate()
        cube_y.setColor(0.0,1.0,0.0)
        cube_y.setScale(1)
        cube_y.reparentTo(self.render)
        cube_y.setPos(0,task.frame,0)

        cube_z = CubeMaker.CubeMaker().generate()
        cube_z.setColor(0.0,0.0,1.0)
        cube_z.setScale(1)
        cube_z.reparentTo(self.render)
        cube_z.setPos(0,0,task.frame)
        if task.time > 25:
            return None
        return task.cont


    # Render-n-Learn task
    def renderNtestTask(self, task):
        if task.frame < 50: #do not do anything for 50 ticks, as spline's 1st node is at t=50
            return task.cont


        # print randX, randY, randZ
        t = task.frame
        if t > self.spl_u.max():
            print 'End of Spline, End task'
            fName = 'trace__' + self.pathGen.__name__ + '.npz'
            np.savez( fName, loss=self.loss_ary, gt=self.gt_ary, pred=self.pred_ary )
            print 'PathData File Written : ', fName
            print 'Visualize result : `python tools/analyse_path_trace_subplot.py', fName, '`'
            return None


        #
        # set pose in each camera
        # Note: The texture is grided images in a col-major format
        # TODO : since it is going to be only 1 camera eliminate this loop to simply code
        poses = np.zeros( (len(self.cameraList), 4), dtype='float32' )
        for i in range(len(self.cameraList)): #here usually # of cams will be 1 (for TestRenderer)
            indx = TestRenderer.renderIndx
            pt = interpolate.splev( t, self.spl_tck)
            #randX,randY, randZ, randYaw, randPitch, randRoll = self.monte_carlo_sample()

            randX = pt[0]
            randY = pt[1]
            randZ = pt[2]
            randYaw = pt[3]
            randPitch = 0
            randRoll = 0


            self.cameraList[i].setPos(randX,randY,randZ)
            self.cameraList[i].setHpr(randYaw,-90+randPitch,0+randRoll)

            poses[i,0] = randX
            poses[i,1] = randY
            poses[i,2] = randZ
            poses[i,3] = randYaw




        # make note of the poses just set as this will take effect next
        if TestRenderer.renderIndx == 0:
            TestRenderer.renderIndx = TestRenderer.renderIndx + 1
            # self.putBoxes(0,0,0, scale=100)
            self.prevPoses = poses
            return task.cont




        #
        # Retrive Rendered Data
        tex = self.win2.getScreenshot()
        # A = np.array(tex.getRamImageAs("RGB")).reshape(960,1280,3) #@#
        A = np.array(tex.getRamImageAs("RGB")).reshape(240,320,3)
        # A = np.zeros((960,1280,3))
        # A_bgr =  cv2.cvtColor(A.astype('uint8'),cv2.COLOR_RGB2BGR)
        # cv2.imwrite( str(TestRenderer.renderIndx)+'.png', A_bgr.astype('uint8') )
        # myTexture = self.win2.getTexture()
        # print myTexture

        # retrive poses from prev render
        texPoses = self.prevPoses

        #
        # Cut rendered data into individual image. Note rendered data will be 4X4 grid of images
        #960 rows and 1280 cols (4x4 image-grid)
        nRows = 240
        nCols = 320
        # Iterate over the rendered texture in a col-major format
        c=0
        # TODO : Eliminate this 2-loop as we know there is only 1 display region
        if self.q_imStack.qsize() < 150:
            for j in range(1): #j is for cols-indx
                for i in range(1): #i is for rows-indx
                    #print i*nRows, j*nCols, (i+1)*nRows, (j+1)*nCols
                    im = A[i*nRows:(i+1)*nRows,j*nCols:(j+1)*nCols,:]
                    #imX = im.astype('float32')/255. - .5 # TODO: have a mean image
                    #imX = (im.astype('float32') - 128.0) /128.
                    imX = im.astype('float32')  #- self.meanImage

                    # Put imX into the queue
                    # do not queue up if queue size begin to exceed 150


                    self.q_imStack.put( imX )
                    self.q_labelStack.put( texPoses[c,:] )


                    # fname = '__'+str(poses[c,0]) + '_' + str(poses[c,1]) + '_' + str(poses[c,2]) + '_' + str(poses[c,3]) + '_'
                    # cv2.imwrite( str(TestRenderer.renderIndx)+'__'+str(i)+str(j)+fname+'.png', imX.astype('uint8') )

                    c = c + 1



        # AAZ
        # Call caffe iteration (reads from q_imStack and q_labelStack)
        #       Possibly upgrade to TensorFlow
        PRED, GT, LOSS = self.test_iteration()
        if PRED is not None:
            self.putBoxes(PRED[0],PRED[1],PRED[2], r=0, g=1, b=0, scale=0.5) # GT in green
            self.putBoxes(GT[0],GT[1],GT[2], r=1, g=0, b=0, scale=0.5) # GT in green

            self.loss_ary.append( LOSS )
            self.gt_ary.append( GT )
            self.pred_ary.append( PRED )



        #
        # Prep for Next Iteration
        TestRenderer.renderIndx = TestRenderer.renderIndx + 1
        self.prevPoses = poses

        # if( TestRenderer.renderIndx > 5 ):
            # return None

        return task.cont



    def __init__(self, tr_model, arch_proto):
        ShowBase.__init__(self)
        self.taskMgr.add( self.renderNtestTask, "renderNtestTask" ) #changing camera poses
        # self.taskMgr.add( self.putAxesTask, "putAxesTask" ) #draw co-ordinate axis


        # Misc Setup
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.setFrameRateMeter(True)

        self.tcolor = TerminalColors.bcolors()




        #
        # Set up Mesh (including load, position, orient, scale)
        self.setupMesh()
        self.positionMesh()


        # Custom Render
        #   Important Note: self.render displays the low_res and self.scene0 is the images to retrive
        self.scene0 = NodePath("scene0")
        # cytX = copy.deepcopy( cyt )
        self.low_res.reparentTo(self.render)

        self.cyt.reparentTo(self.scene0)
        self.cyt2.reparentTo(self.scene0)





        #
        # Make Buffering Window
        bufferProp = FrameBufferProperties().getDefault()
        props = WindowProperties()
        # props.setSize(1280, 960)
        props.setSize(320, 240) #@#
        win2 = self.graphicsEngine.makeOutput( pipe=self.pipe, name='wine1',
        sort=-1, fb_prop=bufferProp , win_prop=props,
        flags=GraphicsPipe.BFRequireWindow)
        #flags=GraphicsPipe.BFRefuseWindow)
        # self.window = win2#self.win #dr.getWindow()
        self.win2 = win2
        # self.win2.setupCopyTexture()



        # Adopted from : https://www.panda3d.org/forums/viewtopic.php?t=3880
        #
        # Set Multiple Cameras
        self.cameraList = []
        # for i in range(4*4):
        for i in range(1*1): #@#
            print 'Create camera#', i
            self.cameraList.append( self.customCamera( str(i) ) )


        # Disable default camera
        # dr = self.camNode.getDisplayRegion(0)
        # dr.setActive(0)




        #
        # Set Display Regions (4x4)
        dr_list = self.customDisplayRegion(1,1)


        #
        # Setup each camera
        for i in  range(len(dr_list)):
            dr_list[i].setCamera( self.cameraList[i] )


        #
        # Set buffered Queues (to hold rendered images and their positions)
        # each queue element will be an RGB image of size 240x320x3
        self.q_imStack = Queue.Queue()
        self.q_labelStack = Queue.Queue()


        # Caffe
        # Caffe init was here, now removed

        # Setup TensorFLow
        #TODO Get tensorflow model file info from command line (need to edit constructor)
        puf_obj = puf.PlutoFlow(trainable_on_device='/gpu:0')


        # Setup placeholders (need just 1 placeholder, ie. input image)
        #TODO Try `1` instead of `None`
        self.tf_x = tf.placeholder( 'float', [None,240,320,3], name='x' )


        # Set the ResNet inference_op
        with tf.device( '/gpu:0'):
            self.tf_infer_op = puf_obj.resnet50_inference(self.tf_x, is_training=False)


        # Print all Trainable Variables
        var_list = tf.trainable_variables()
        print '--Trainable Variables--', 'length= ', len(var_list)
        total_n_nums = []
        for vr in var_list:
            shape = vr.get_shape().as_list()
            n_nums = np.prod(shape)
            total_n_nums.append( n_nums )
            print self.tcolor.OKGREEN, vr.name, shape, n_nums, self.tcolor.ENDC

        print self.tcolor.OKGREEN, 'Total Trainable Params (floats): ', sum( total_n_nums )
        print 'Not counting the pop_mean and pop_varn as these were set to be non trainable', self.tcolor.ENDC


        # Fire up the TensorFlow-Session
        self.tensorflow_session = tf.Session( config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True) )


        # Load trainables' values from file
        self.tensorflow_saver = tf.train.Saver()
        restore_file_name = 'tf.models/model-165000'
        print 'Loading Model File : ', restore_file_name
        self.tensorflow_saver.restore( self.tensorflow_session, restore_file_name )
        print self.tcolor.OKGREEN, 'Loaded file : ', restore_file_name, self.tcolor.ENDC



        # store loss at each frame in the trajectory
        self.loss_ary = []
        self.gt_ary = []
        self.pred_ary = []

        #
        # Setting up Splines
        # Note: Start interpolation at 50,

        # self.pathGen = PathMaker.PathMaker().path_flat_h
        # self.pathGen = PathMaker.PathMaker().path_smallM
        # self.pathGen = PathMaker.PathMaker().path_yaw_only
        # self.pathGen = PathMaker.PathMaker().path_bigM
        # self.pathGen = PathMaker.PathMaker().path_flat_spiral
        # self.pathGen = PathMaker.PathMaker().path_helix
        # self.pathGen = PathMaker.PathMaker().path_like_real
        self.pathGen = PathMaker.PathMaker().path_like_real2


        t,X = self.pathGen()

        self.spl_tck, self.spl_u = interpolate.splprep(X.T, u=t.T, s=0.0, per=1)







#
# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--tr_model", help="Trained Model .caffemodel.h5 file")
parser.add_argument("-a", "--arch_proto", help="Net archi prototxt file")
args = parser.parse_args()

tcolor = TerminalColors.bcolors()

if args.tr_model:
	tr_model = args.tr_model
else:
    tr_model = 'caffe_snaps/x_ResNet50_b_iter_200000.caffemodel.h5'


if args.arch_proto:
	arch_proto = args.arch_proto
else:
    arch_proto = 'net_workshop/ResNet50_b_test.prototxt'


print tcolor.HEADER, 'tr_model : ', tr_model, tcolor.ENDC
print tcolor.HEADER, 'arch_proto   : ', arch_proto, tcolor.ENDC



app = TestRenderer(tr_model, arch_proto)
app.run()
