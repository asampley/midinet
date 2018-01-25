import mido
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('-t', '--timed', action='store_true')
args = parser.parse_args()

filename = args.filename
mfile = mido.midifiles.MidiFile(filename)

if args.timed:
    for msg in mfile.play():
        print(msg)
else:
    track = mido.merge_tracks(mfile.tracks)
    for i in range(len(track)):
        print(track[i])

