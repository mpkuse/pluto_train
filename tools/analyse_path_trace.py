# Takes the *_trace.npz file as input. This .npz files have 3 arrays
# viz. 'pred', 'loss', 'gt'.
# This script plots :
# a) pred_x, gt_x vs time (frameIndx)
# b) pred_y, gt_y vs time (frameIndx)
# c) pred_z, gt_z vs time (frameIndx)
# d) pred_yaw, gt_yaw vs time (frameIndx)

import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D


parser = argparse.ArgumentParser()
parser.add_argument( "file", help="file name of the path-trace file" )
args = parser.parse_args()

if args.file == None:
    parser.print_help()
    quit()

print 'Opening File : ', args.file
X = np.load( args.file )



plt.plot( X['pred'][:,0], 'r-', label='pred' )
plt.plot( X['gt'][:,0], 'g-', label='gt' )
plt.xlabel( 'Frame#')
plt.ylabel( 'x co-ordinate (m)')
plt.title( 'comparing x-cord prediction')
plt.legend()

plt.figure()
plt.plot( X['pred'][:,1], 'r-', label='pred' )
plt.plot( X['gt'][:,1], 'g-', label='gt' )
plt.xlabel( 'Frame#')
plt.ylabel( 'y co-ordinate (m)')
plt.title( 'comparing y-cord prediction')
plt.legend()

plt.figure()
plt.plot( X['pred'][:,2], 'r-', label='pred' )
plt.plot( X['gt'][:,2], 'g-', label='gt' )
plt.xlabel( 'Frame#')
plt.ylabel( 'z co-ordinate (m)')
plt.title( 'comparing z-cord prediction')
plt.legend()

plt.figure()
plt.plot( X['pred'][:,3], 'r-', label='pred' )
plt.plot( X['gt'][:,3], 'g-', label='gt' )
plt.xlabel( 'Frame#')
plt.ylabel( 'yaw (degrees)')
plt.title( 'comparing yaw prediction')
plt.legend()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X['gt'][:,0], X['gt'][:,1], X['gt'][:,2], c='g', label='gt' )
ax.plot(X['pred'][:,0], X['pred'][:,1], X['pred'][:,2], c='r', label='pred' )
ax.set_zlim3d(0, 120)
ax.set_ylim3d(-120, 120)                    
ax.set_xlim3d(-70, 70)
ax.set_xlabel( 'x')
ax.set_ylabel( 'y')
ax.set_zlabel( 'z')
plt.legend()



plt.show()
