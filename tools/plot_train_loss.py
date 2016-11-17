import numpy as np
import matplotlib.pyplot as plt
import argparse



parser = argparse.ArgumentParser()
parser.add_argument( "-w", "--window", help="window size for smoothing the loss", type=int )
parser.add_argument( "-f", "--file", help="file name of the training loss file. Default train_loss.npy" )
args = parser.parse_args()

if args.window:
	window = args.window
else:
	window = 10

if args.file:
	train_loss_file = args.file
else:
	train_loss_file = 'train_loss.npy'


train_loss = np.load( train_loss_file )
#test_loss = np.load( 'test_loss.npy' )


smooth_train_loss = np.convolve(train_loss[0:], np.ones( window )/window, 'valid' )
#mooth_test_loss = np.convolve(test_loss[0:], np.ones( 5*window )/(5*window), 'valid' )

plt.plot( train_loss[0:], 'r-', smooth_train_loss, 'b' )
plt.figure()
plt.plot( np.log(smooth_train_loss[0:] ) , 'r-' )
plt.show()



# fig, ax1 = plt.subplots()
# ax1.plot( train_loss[50:], 'r-', smooth_train_loss, 'b' )
# ax2 = ax1.twinx()
# ax2.plot( test_loss[50:], 'g-' )
#
# fig, ax_log = plt.subplots()
# ax_log.plot( np.log(smooth_train_loss[0:] ) , 'r-' )
# #ax_log.plot( np.log(smooth_test_loss[0:] ) , 'g-' )
# ax_log.plot( np.log(test_loss[0:] ) , 'g-' )
#
# plt.show()
# #fig.savefig( 'b.png')
