import mido
import argparse

parser = argparse.ArgumentParser(description='Remove all but the notes from the file')
parser.add_argument('--tempo', '-t', help='Tempo of the file', default=500000)
parser.add_argument('inputfile', help='Midi file to read from')
parser.add_argument('outputfile', help='Midi file to output to')
args = parser.parse_args()

ifilename = args.inputfile
ofilename = args.outputfile
mfile = mido.midifiles.MidiFile(ifilename)
mofile = mido.midifiles.MidiFile()
mofile.add_track()
mofile.tracks[0].append(mido.MetaMessage('set_tempo', tempo=args.tempo))

track = mido.merge_tracks(mfile.tracks)

for msg in track:
    # change velocity to one value
    newmsg = msg.copy()
    
    if msg.type == 'note_on' or msg.type == 'note_off':
        pass
    else:
        print("Clobbered " + str(msg))
        if len(mofile.tracks[0]) > 0:
            mofile.tracks[0][-1].time += msg.time
            print('*' + str(mofile.tracks[0][-1]))
        continue
    mofile.tracks[0].append(newmsg)
    print(newmsg)

mofile.save(ofilename)
