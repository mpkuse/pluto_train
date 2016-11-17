from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from direct.stdpy import thread
import numpy as np

import CubeMaker

class MyApp(ShowBase):

    def spinMesh( self, task ):

        t = task.time
        f = task.frame
        # print f/10

        return task.cont

    def spinTask( self, task ):

        if task.frame % 10 != 0:
            return task.cont

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

    def loadAllTextures(self, cyt, basePath):
        for child in cyt.getChildren():
            submesh_name = child.get_name()
            submesh_texture = basePath + submesh_name[:-5] + 'tex0.jpg'
            child.setTexture( self.loader.loadTexture(submesh_texture) )
            print 'Loading texture file : ', submesh_texture

    def __init__(self):
        ShowBase.__init__(self)
        self.taskMgr.add( self.spinTask, "spinCam")
        self.taskMgr.add( self.spinMesh, "spinMesh")

        self.cyt = self.loader.loadModel( 'model_l/l6/level_6_0_0.obj' )
        self.loadAllTextures( self.cyt, 'model_l/l6/')

        self.cyt2 = self.loader.loadModel( 'model_l/l6/level_6_128_0.obj' )
        self.loadAllTextures( self.cyt2, 'model_l/l6/')

        # self.cyt = self.loader.loadModel( 'l/l7/level_7_128_0.obj' )
        # self.loadAllTextures( self.cyt, 'l/l7/')
        #
        # self.cyt2 = self.loader.loadModel( 'l/l7/level_7_128_64.obj' )
        # self.loadAllTextures( self.cyt2, 'l/l7/')


        self.cyt.setPos( 140,-450, 170 )
        self.cyt2.setPos( 140,-450, 170 )
        self.cyt.setHpr( 198, -90, 0 )
        self.cyt2.setHpr( 198, -90, 0 )
        self.cyt.setScale(200)
        self.cyt2.setScale(200)

        self.cyt.reparentTo(self.render)
        self.cyt2.reparentTo(self.render)

        # cyt2 = self.loader.loadModel( 'l/l6/level_6_128_0.obj' )
        # cyt2.setPos( 0, 0, 0 )
        # cyt2.reparentTo(self.render)

        self.useTrackball()
        self.cam.setPos( 0,0,160)
        self.cam.setHpr( 0,-90,0)






app = MyApp()
app.run()
