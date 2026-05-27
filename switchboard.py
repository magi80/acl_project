"""Functions to import data from the Switchboard corpus."""

import xml.etree.ElementTree as ET
import re
import os.path

def normalize_form(s):
    if (m := re.match(r'\[laughter-(.*)\]$', s)):
        return m.group(1)
    return s

def read_words(path):
    """Read a .phonwords.xml file from the Switchboard corpus."""
    tree = ET.parse(path)
    words = []
    for phonword in tree.findall('phonword'):
        form = phonword.get('orth')
        start = float(phonword.get('{http://nite.sourceforge.net/}start'))
        end = float(phonword.get('{http://nite.sourceforge.net/}end'))
        words.append((start, end, normalize_form(form)))
    return words

def iterate_utterances(words, pause_max):
    utt = []
    for t0, t1, form in words:
        if utt and (t0 - utt[-1][1]) > pause_max:
            yield utt
            utt = []
        utt.append((t0, t1, form))
    if utt:
        yield utt

def read_switchboard(path_prefix, pause_max=0.3, dialogue_speaker_id=None):
    utterances = []
    dialogue = os.path.basename(path_prefix)
    assert dialogue.startswith('sw')
    for participant in ('A', 'B'):
        path = f'{path_prefix}.{participant}.phonwords.xml'
        words = read_words(path)
        for utt in iterate_utterances(words, pause_max):
            # No longer supported
            assert dialogue_speaker_id is not None
            # if dialogue_speaker_id is None:
            #     speaker = f'{dialogue}.{participant}'
            # else:
            speaker = dialogue_speaker_id[(dialogue[2:], participant)]
            utterances.append({
                't_start': utt[0][0],
                't_end': utt[-1][1],
                'words': [{'form': form,
                           't_start': t0,
                           't_end': t1}
                            for t0, t1, form in utt],
                'speaker': speaker,
                })
    utterances.sort(key=lambda u: u['t_start'])
    return {'session': dialogue, 'utterances': utterances}

def read_dialogue_speaker(corpus_resources_dir):
    dialogues = ET.parse(os.path.join(corpus_resources_dir, 'dialogues.xml'))
    speakers = ET.parse(os.path.join(corpus_resources_dir, 'speakers.xml'))
    speaker_properties = {}
    dialogue_speaker_id = {}
    for speaker in speakers.findall('speaker'):
        sex = speaker.get('sex')
        year = speaker.get('dob')
        dialect = speaker.get('dialect')
        speaker_id = speaker.get('{http://nite.sourceforge.net/}id')
        assert speaker_id.startswith('spkr')
        speaker_properties[speaker_id] = dict(
                id=speaker_id[4:],
                sex=sex,
                year=year,
                dialect=dialect)

    for dialogue in dialogues.findall('dialogue'):
        nr = dialogue.get('swbdid')
        for el in dialogue:
            href = el.get('href')
            if (m := re.match(r'speakers\.xml#id\((spkr\d+)\)$', href)):
                speaker = m.group(1)
                role = el.get('role')
                dialogue_speaker_id[(nr, role)] = speaker_properties[speaker]

    return dialogue_speaker_id

if __name__ == '__main__':
    import sys
    import pprint
    filename = sys.argv[1]
    corpus_path = os.path.dirname(os.path.dirname(filename))
    dialogue_speaker_id = read_dialogue_speaker(
            os.path.join(corpus_path, 'corpus-resources'))
    pprint.pp(dialogue_speaker_id)
    pprint.pp(read_switchboard(
        filename, dialogue_speaker_id=dialogue_speaker_id))
