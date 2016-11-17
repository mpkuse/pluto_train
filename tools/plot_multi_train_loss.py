import argparse
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument("files", metavar='N', help="all files to overlay", nargs='+')
parser.add_argument("-w", "--window", help="Smoothing window size", type=int)
args = parser.parse_args()

if args.window:
	window = args.window
else:
	window = 10

# Plot all specified files
for fileName in args.files:
	print 'Read file : ', fileName

	train_loss = np.load( fileName )

	smooth_train_loss = np.convolve(train_loss[0:], np.ones( window )/window, 'valid' )

	plt.plot( np.log(smooth_train_loss[0:] ) , label=fileName )
	plt.legend(loc='best')

plt.show()
