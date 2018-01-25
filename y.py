from midi2bytes import midi2bytes
from midi2bytes import bytes2midi
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('infile')
parser.add_argument('outfile')
args = parser.parse_args()

times, b = midi2bytes(args.infile)

# print out the notes
for i in range(times.size):
    print(str(times[i]) + ':\t' + str(np.where(b[i,:] != 0)[0]))

f = bytes2midi(times, b)
f.save(args.outfile)
