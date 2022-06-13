"""
this file is like the main function of the whole program
"""

from scripts.martingales import SMM
from scripts.detector import StatefulDetector, StatelessDetector
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Out-of-distribution detection offline.')
parser.add_argument("-o", "--out", help="set test images from the out-of-distribution test set", action="store_true", default=False)
parser.add_argument("-v", "--vae", help="set nonconformity measure as VAE-based method", action="store_true", default=False)
parser.add_argument("-N", help="sliding window size for SVDD; # of generated examples for VAE", type=int, default=10)
args = parser.parse_args()


# load the test images
if args.out:
    data_path = "./data/out"
else:
    data_path = "./data/in"
    
# test image is loaded from the given path
# ? should the X.npy always be overwritten by new image from online data stream???
# or the loading/overwritting frequency should be slower than the detection delay
test_images = np.load(os.path.join(data_path, "X.npy"))

# using vae to detect anomalies
if args.vae:
    from scripts.icad_vae import ICAD_VAE
    # creating VAE network, hard-coded, no customization for the network
    # the returned value is the p-value
    icad = ICAD_VAE() # object can also be called as function to accept online image data
    
    # using stateful detector to avoid cosistent growth of martingale
    detector = StatefulDetector(sigma=6, tau=156)
    for idx, test_image in enumerate(test_images):
        smm = SMM(args.N) # object can be called as function
        for i in range(args.N):
            p = icad(test_image) # object returns p-value of given input image
            m = np.log(smm(p)) # object is called as function, which returns logarithmic martingale
        S, d = detector(m) # In the section VAEs of GUI S and m can be plotted
        print("Time step: {}\t p-value: {}\t logM: {}\t S: {}\t Detector: {}".format(idx, round(p,3), round(m, 3), round(S, 3), d))

# using SVDD to detect anomalies
else:
    from scripts.icad_svdd import ICAD_SVDD
    # creating SVDD network, hard-coded, no customization for the network
    icad = ICAD_SVDD()
    # setting the size of sliding window for SVDD
    smm = SMM(args.N)
    detector = StatelessDetector(tau=14)
    for idx, test_image in enumerate(test_images):
        p = icad(test_image)
        m = np.log(smm(p))
        d = detector(m)
        print("Time step: {}\t p-value: {}\t logM: {}\t Detector: {}".format(idx, round(p,3), round(m, 3), d))