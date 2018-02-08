import mido

mfile = mido.MidiFile()
mfile.add_track()
mfile.tracks[0].append(mido.Message('control_change', control=64, value=127))
mfile.tracks[0].append(mido.Message('control_change', control=91, value=127))
mfile.tracks[0].append(mido.Message('note_on', note=0, velocity=127, time=0))
mfile.tracks[0].append(mido.Message('note_on', note=1, velocity=127, time=120))
mfile.tracks[0].append(mido.Message('note_on', note=0, velocity=0, time=0))
mfile.tracks[0].append(mido.Message('note_on', note=2, velocity=127, time=60))
mfile.tracks[0].append(mido.Message('note_on', note=1, velocity=0, time=0))
mfile.tracks[0].append(mido.Message('note_on', note=3, velocity=127, time=0))

mfile.save('test/test.mid')
