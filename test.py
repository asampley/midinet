import mido

mfile = mido.MidiFile()
mfile.add_track()
track = mfile.tracks[0]

track.append(mido.Message('control_change', control=64, value=127))
track.append(mido.Message('control_change', control=91, value=127))

track.append(mido.Message('note_on', note=0, velocity=127, time=0))
track.append(mido.Message('note_on', note=1, velocity=127, time=120))
track.append(mido.Message('note_on', note=0, velocity=0, time=0))
track.append(mido.Message('note_on', note=2, velocity=127, time=60))
track.append(mido.Message('note_on', note=1, velocity=0, time=0))
track.append(mido.Message('note_on', note=3, velocity=127, time=0))

track.append(mido.Message('note_on', note=12, velocity=127, time=180))
track.append(mido.Message('note_on', note=13, velocity=127, time=0))
track.append(mido.Message('note_on', note=14, velocity=127, time=240))
track.append(mido.Message('note_on', note=13, velocity=127, time=300))
track.append(mido.Message('note_on', note=15, velocity=127, time=600))

mfile.save('test/test.mid')
