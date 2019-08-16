import sys
import classifier_runner

print(sys.argv)
params = dict()
if sys.argv[1] == "-d":
    try:
        params['description'] = sys.argv[2]
    except:
        params['description'] = ""
else:
    params['description'] = ""
    

num_classes = 9
### where to read images from
params['datafile'] = '/home/cjw/Code/DeepLearning/classifier/Data/images.mm'
### where are the labels
params['labelsfile'] = '/home/cjw/Code/DeepLearning/classifier/Data/labels.mm'
### what clusters from the labels to use
params['clusterlist'] = list(range(num_classes))
params['combine'] = None
params['tensorboard_log_dir'] = '/scratch/cjw/logs'
params['channels'] = [0,2,4]

params['iterations'] = 12000
params['learning_rate'] = 0.001 #0.0006
params['droprate'] = 0.0
params['l2f'] = 0.006 #0.004
params['batchsize'] = 256

### Name of output checkpoint directory
params['CheckpointDir'] = "Checkpoints/Snail_Redo_for_metrics/"
classifier_runner.run(params)

print("Done")
