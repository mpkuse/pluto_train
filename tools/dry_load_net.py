import sys
import caffe
import argparse
# Load a net and print its params and/or blobs dim


parser = argparse.ArgumentParser()
parser.add_argument( "-f", "--file", help="caffeNet prototxt file name" )
parser.add_argument( "-b", "--blobs", help="display all of net.blobs" )
parser.add_argument( "-p", "--params", help="display all of net.params" )
parser.add_argument( "-g", "--gpu", type=int, help="ID of the GPU to use" )
parser.add_argument( "-t", "--test", type=int, help="Set testing mode. Absense of this flag will set training mode" )
args = parser.parse_args()


## Load Net
print "Load Net"

if args.gpu:
    caffe.set_mode_gpu()
    caffe.set_device(int(args.gpu))
else:
    caffe.set_mode_cpu()

if args.file:
    caffe_net = args.file
else:
    caffe_net = 'resnet64_yaw.prototxt'

if args.test:
    mode = caffe.TEST
else:
    mode = caffe.TRAIN

net = caffe.Net(caffe_net, mode)


if args.blobs:
    print '## net.blobs'
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)
    print '## end net.blobs'

if args.params:
    print '## net.params'
    for layer_name, params in net.params.iteritems():
        for p in params: #some params hav a bias terms
            print layer_name + ':' + str(p.data.shape) + '\t',
        print ''
    print '## end net.params'

# print 'Display shape of intermediate layers-paramms'
# #Dat = []
# for layer_name, param in net.params.iteritems():
#     print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
# #    Dat[layer_name] = np.array( param[0].data )
