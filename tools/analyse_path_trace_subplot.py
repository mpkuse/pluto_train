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

f, axarr = plt.subplots(2, 2)

axarr[0,0].plot( X['pred'][:,0], 'r-', label='pred' )
axarr[0,0].plot( X['gt'][:,0], 'g-', label='gt' )
axarr[0,0].set_xlabel( 'Frame#')
axarr[0,0].set_ylabel( 'x co-ordinate (m)')
axarr[0,0].set_title( 'comparing x-cord prediction')
# axarr[0,0].set_legend()

#plt.figure()
axarr[0,1].plot( X['pred'][:,1], 'r-', label='pred' )
axarr[0,1].plot( X['gt'][:,1], 'g-', label='gt' )
axarr[0,1].set_xlabel( 'Frame#')
axarr[0,1].set_ylabel( 'y co-ordinate (m)')
axarr[0,1].set_title( 'comparing y-cord prediction')
# axarr[0,0].legend()

# plt.figure()
axarr[1,0].plot( X['pred'][:,2], 'r-', label='pred' )
axarr[1,0].plot( X['gt'][:,2], 'g-', label='gt' )
axarr[1,0].set_xlabel( 'Frame#')
axarr[1,0].set_ylabel( 'z co-ordinate (m)')
axarr[1,0].set_title( 'comparing z-cord prediction')
# axarr[1,0].legend()

# plt.figure()
axarr[1,1].plot( X['pred'][:,3], 'r-', label='pred' )
axarr[1,1].plot( X['gt'][:,3], 'g-', label='gt' )
axarr[1,1].set_xlabel( 'Frame#')
axarr[1,1].set_ylabel( 'yaw (degrees)')
axarr[1,1].set_title( 'comparing yaw prediction')
# axarr[1,0].legend()


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
