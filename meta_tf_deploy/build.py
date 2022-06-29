from cnn2snn import check_model_compatibility, quantize, convert
from tensorflow import keras
import argparse
import json, os, shutil, sys

# parse arguments (--metadata FILE is passed in)
parser = argparse.ArgumentParser(description='Custom deploy block demo')
parser.add_argument('--metadata', type=str)
args = parser.parse_args()

# load the metadata.json file
with open(args.metadata) as f:
    metadata = json.load(f)

print('Copying files to build directory...')
# training directory mounted directly to access all artifacts
training_dir = '/data'
input_dir = metadata['folders']['input']
output_dir = metadata['folders']['output']

# create a build directory, the input / output folders are on network storage so might be very slow
build_dir = '/tmp/build'
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)
os.makedirs(build_dir)

# copy in the data from input folder
try:
    os.system('cp -r ' + training_dir + '/trained.fbz ' + build_dir)
except:
    print('ERROR: Failed to find .fbz model artifact. Ensure the project was trained with an Akida learning block')
    os.system('cp -r ' + input_dir + '/trained.savemodel.zip ' + build_dir)
    shutil.unpack_archive(os.path.join(build_dir, 'trained.savedmodel.zip'), build_dir, 'zip')
    os.system('rm ' + 'build_dir/*.zip')
    os.system('ls -l ' + build_dir)

print('Copying files to build directory OK')
print('')

shutil.make_archive(os.path.join(output_dir, 'akida_deployment'), 'zip', build_dir)
