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

def read_switchboard(path_prefix, pause_max=0.3):
    utterances = []
    dialogue = os.path.basename(path_prefix)
    # From corpus-resources/dialogues.xml and
    # corpus-resources/speakers.xml we can get speaker IDs and metadata
    # instead of just A or B below. Implement if needed.
    for participant in ('A', 'B'):
        path = f'{path_prefix}.{participant}.phonwords.xml'
        words = read_words(path)
        for utt in iterate_utterances(words, pause_max):
            utterances.append({
                't_start': utt[0][0],
                't_end': utt[-1][1],
                'words': [{'form': form,
                           't_start': t0,
                           't_end': t1}
                            for t0, t1, form in utt],
                'speaker': f'{dialogue}.{participant}' # TODO: speaker ID
                })
    utterances.sort(key=lambda u: u['t_start'])
    return {'utterances': utterances}

if __name__ == '__main__':
    import sys
    import pprint
    pprint.pp(read_switchboard(sys.argv[1]))
