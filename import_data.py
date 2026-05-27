"""High-level script to import data from the Switchboard and Buckeye corpora.

Note that the Switchboard path given should be to the 'xml' subdirectory, and
the Buckeye path should be a directory containing all the data files,
uncompressed and without any subdirectories. For instance:

python3 import_data.py \
    /hd1/corpora/nxt-switchboard/xml \
    /hd1/corpora/buckeye/uncompressed
"""

import sys
import glob
import os
import json

import buckeye
import switchboard

def import_switchboard(directory):
    dialogue_speaker_id = switchboard.read_dialogue_speaker(
            os.path.join(directory, 'corpus-resources'))
    phonword_files = glob.glob(os.path.join(directory, 'phonwords', '*.xml'))
    prefixes = sorted({os.path.basename(path).split('.', 1)[0]
                       for path in phonword_files})
    for prefix in prefixes:
        data = switchboard.read_switchboard(
                os.path.join(directory, 'phonwords', prefix),
                dialogue_speaker_id=dialogue_speaker_id)
        yield data

def import_buckeye(directory):
    words_files = glob.glob(os.path.join(directory, '*.words'))
    for path in words_files:
        yield buckeye.read_buckeye(path)

def main():
    switchboard_directory, buckeye_directory = sys.argv[1:]
    switchboard_out = 'json/switchboard'
    buckeye_out = 'json/buckeye'
    os.makedirs(switchboard_out, exist_ok=True)
    os.makedirs(buckeye_out, exist_ok=True)
    for data in import_switchboard(switchboard_directory):
        json_path = os.path.join(switchboard_out, data['session']+'.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
    for data in import_buckeye(buckeye_directory):
        json_path = os.path.join(buckeye_out, data['session']+'.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == '__main__':
    main()

