""" The training-render file. (class TrainRenderer)
        Defines a rendering class. Defines a spinTask (panda3d) which basicalyl
        renders 16-cameras at a time and sets them into a CPU-queue.
        After setting it into queue, a class function call_caffe() is called
        which transfroms the data and sets it into caffe, does 1 iteration
        and returns back to spinTask. Caffe prototxt and solver files loaded in
        class constructor.

        Training with TensorFlow
"""

# Panda3d
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from direct.stdpy import thread

# Usual Math and Image Processing
import numpy as np
import cv2
# import caffe


# Other System Libs
import os
import argparse
import Queue
import copy
import time


# Misc
import TerminalColors
import CubeMaker
import censusTransform as ct
import Noise


class TrainRenderer(ShowBase):
    renderIndx=0


    # Given a square matrix, substract mean and divide std dev
    def zNormalized(self, M ):
        M_mean = np.mean(M) # scalar
        M_std = np.std(M)
        if M_std < 0.0001 :
            return M

        M_statSquash = (M - M_mean)/(M_std+.0001)
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

        self.low_res = self.loader.loadModel( 'model_l/l3/level_3_0_0.obj' )
        print self.tcolor.OKGREEN, 'Done Loading Vertices', self.tcolor.ENDC

        print 'Attempt Loading Textures'
        self.loadAllTextures( self.cyt, 'model_l/l6/')
        self.loadAllTextures( self.cyt2, 'model_l/l6/')
        self.loadAllTextures( self.low_res, 'model_l/l3/')
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

    def learning_iteration(self):
        """Does 1 iteration.
            Basically retrives the rendered image and pose from queue. Preprocess/Purturb it
            and feed it into caffe for training.
        """
        # print 'DoCaffeTrainng'
        startTime = time.time()
        # print 'q_imStack.qsize() : ', self.q_imStack.qsize()
        # print 'q_labelStack.qsize() : ', self.q_labelStack.qsize()


        # if too few items in queue do not proceed with iterations
        if self.q_imStack.qsize() < 16*5:
            return None



        batchsize = self.solver.net.blobs['data'].data.shape[0]
        # print 'batchsize', batchsize
        # print "self.solver.net.blobs['data'].data", self.solver.net.blobs['data'].data.shape
        # print "self.solver.net.blobs['label_x'].data",self.solver.net.blobs['label_x'].data.shape
        for i in range(batchsize):
            im = self.q_imStack.get() #320x240x3
            y = self.q_labelStack.get()

            im_noisy = Noise.noisy( 'gauss', im )
            im_gry = np.mean( im_noisy, axis=2)


            # cv2.imwrite( str(i)+'__.png', x )

            cencusTR = ct.censusTransform( im_gry.astype('uint8') )
            edges_out = cv2.Canny(cv2.blur(im_gry.astype('uint8'),(3,3)),100,200)


            self.solver.net.blobs['data'].data[i,0,:,:] = self.zNormalized( im_gry.astype('float32') )
            self.solver.net.blobs['data'].data[i,1,:,:] = self.zNormalized( cencusTR.astype('float32') )
            self.solver.net.blobs['data'].data[i,1,:,:] = self.zNormalized( edges_out.astype('float32') )
            self.solver.net.blobs['label_x'].data[i,0] = y[0]
            self.solver.net.blobs['label_y'].data[i,0] = y[1]
            self.solver.net.blobs['label_z'].data[i,0] = y[2]
            self.solver.net.blobs['label_yaw'].data[i,0] = y[3]
            # print y[0], y[1], y[2], y[3]

        self.solver.step(1)
        self.caffeTrainingLossX[self.caffeIter] = self.solver.net.blobs['loss_x'].data
        self.caffeTrainingLossY[self.caffeIter] = self.solver.net.blobs['loss_y'].data
        self.caffeTrainingLossZ[self.caffeIter] = self.solver.net.blobs['loss_z'].data
        self.caffeTrainingLossYaw[self.caffeIter] = self.solver.net.blobs['loss_yaw'].data
        if self.caffeIter % 50 == 0 and self.caffeIter>0:
            print 'Writing File : train_loss.npy'
            np.save('train_loss_x.npy', self.caffeTrainingLossX[0:self.caffeIter])
            np.save('train_loss_y.npy', self.caffeTrainingLossY[0:self.caffeIter])
            np.save('train_loss_z.npy', self.caffeTrainingLossZ[0:self.caffeIter])
            np.save('train_loss_yaw.npy', self.caffeTrainingLossYaw[0:self.caffeIter])

        #time.sleep(.3)
        print 'my_iter=%d, solver_iter=%d, time=%f, loss_x=%f, lossYaw=%f' % (self.caffeIter, self.solver.iter, time.time() - startTime, self.caffeTrainingLossX[self.caffeIter], self.caffeTrainingLossYaw[self.caffeIter])
        self.caffeIter = self.caffeIter + 1





    def monte_carlo_sample(self):
        """ Gives a random 6-dof pose. Need to set params manually here.
                X,Y,Z,  Yaw(abt Z-axis), Pitch(abt X-axis), Roll(abt Y-axis) """
        X = np.random.uniform(-150,150)
        Y = np.random.uniform(-300,300)
        Z = np.random.uniform(70,120)

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

    def putTrainingBox(self,task):
        cube = CubeMaker.CubeMaker().generate()

        cube.setTransparency(TransparencyAttrib.MAlpha)
        cube.setAlphaScale(0.5)

        # cube.setScale(10)
        cube.setSx(150)
        cube.setSy(300)
        cube.setSz(25)
        cube.reparentTo(self.render)
        cube.setPos(0,0,95)



    def putAxesTask(self,task):
        if (task.frame / 10) % 2 == 0:
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
    def renderNlearnTask(self, task):
        if task.time < 2: #do not do anything for 1st 2 sec
            return task.cont


        # print randX, randY, randZ

        #
        # set pose in each camera
        # Note: The texture is grided images in a col-major format
        poses = np.zeros( (len(self.cameraList), 4), dtype='float32' )
        for i in range(len(self.cameraList)):
            randX,randY, randZ, randYaw, randPitch, randRoll = self.monte_carlo_sample()
            # if i<4 :
            #     randX = (i) * 30
            # else:
            #     randX = 0
            #
            # randY = 0#task.frame
            # randZ = 80
            # randYaw = 0
            # randPitch = 0
            # randRoll = 0


            self.cameraList[i].setPos(randX,randY,randZ)
            self.cameraList[i].setHpr(randYaw,-90+randPitch,0+randRoll)

            poses[i,0] = randX
            poses[i,1] = randY
            poses[i,2] = randZ
            poses[i,3] = randYaw

        #     self.putBoxes(randX,randY,randZ, scale=0.3)
        #
        # if task.frame < 100:
        #     return task.cont
        # else:
        #     return None



        # make note of the poses just set as this will take effect next
        if TrainRenderer.renderIndx == 0:
            TrainRenderer.renderIndx = TrainRenderer.renderIndx + 1
            self.prevPoses = poses
            return task.cont



        #
        # Retrive Rendered Data
        tex = self.win2.getScreenshot()
        A = np.array(tex.getRamImageAs("RGB")).reshape(960,1280,3)
        # A = np.zeros((960,1280,3))
        # A_bgr =  cv2.cvtColor(A.astype('uint8'),cv2.COLOR_RGB2BGR)
        # cv2.imwrite( str(TrainRenderer.renderIndx)+'.png', A_bgr.astype('uint8') )
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
        if self.q_imStack.qsize() < 150:
            for j in range(4): #j is for cols-indx
                for i in range(4): #i is for rows-indx
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
                    # cv2.imwrite( str(TrainRenderer.renderIndx)+'__'+str(i)+str(j)+fname+'.png', imX.astype('uint8') )

                    c = c + 1
        else:
            print 'q_imStack.qsize() > 150. Queue is filled, not retriving the rendered data'



        #
        # Call caffe iteration (reads from q_imStack and q_labelStack)
        #       Possibly upgrade to TensorFlow
        # self.learning_iteration()



        # if( TrainRenderer.renderIndx > 50 ):
        #     return None

        #
        # Prep for Next Iteration
        TrainRenderer.renderIndx = TrainRenderer.renderIndx + 1
        self.prevPoses = poses



        return task.cont



    def __init__(self, solver_proto, arch_proto):
        ShowBase.__init__(self)
        self.taskMgr.add( self.renderNlearnTask, "renderNlearnTask" ) #changing camera poses
        self.taskMgr.add( self.putAxesTask, "putAxesTask" ) #draw co-ordinate axis
        self.taskMgr.add( self.putTrainingBox, "putTrainingBox" )


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
        props.setSize(1280, 960)
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
        for i in range(4*4):
            print 'Create camera#', i
            self.cameraList.append( self.customCamera( str(i) ) )


        # Disable default camera
        # dr = self.camNode.getDisplayRegion(0)
        # dr.setActive(0)




        #
        # Set Display Regions (4x4)
        dr_list = self.customDisplayRegion(4,4)


        #
        # Setup each camera
        for i in  range(len(dr_list)):
            dr_list[i].setCamera( self.cameraList[i] )


        #
        # Set buffered Queues (to hold rendered images and their positions)
        # each queue element will be an RGB image of size 240x320x3
        self.q_imStack = Queue.Queue()
        self.q_labelStack = Queue.Queue()


        #
        # Set up Caffe (possibly in future TensorFLow)
        # caffe.set_device(0)
        # caffe.set_mode_gpu()
        # self.solver = None
        # self.solver = caffe.SGDSolver(solver_proto)
        # self.caffeIter = 0
        # self.caffeTrainingLossX = np.zeros(300000)
        # self.caffeTrainingLossY = np.zeros(300000)
        # self.caffeTrainingLossZ = np.zeros(300000)
        # self.caffeTrainingLossYaw = np.zeros(300000)








#
# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--solver_proto", help="Solver Prototxt file")
parser.add_argument("-a", "--arch_proto", help="Net archi prototxt file")
args = parser.parse_args()

tcolor = TerminalColors.bcolors()

if args.solver_proto:
	solver_proto = args.solver_proto
else:
    solver_proto = 'solver.prototxt'


if args.arch_proto:
	arch_proto = args.arch_proto
else:
    arch_proto = 'net_workshop/ResNet50_b.prototxt'


print tcolor.HEADER, 'solver_proto : ', solver_proto, tcolor.ENDC
print 'arch_proto   : ', arch_proto



app = TrainRenderer(solver_proto, arch_proto)
app.run()
