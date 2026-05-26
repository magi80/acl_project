"""Functions to import data from the Buckeye corpus.

The main function is `read_buckeye`, which reads one of the .words files from
the Buckeye corpus and returns a dict intended to be exported as JSON.
See the bottom of this file for a simple usage example.
"""

import re
import os

def read_words(path):
    """Read a .words file from Buckeye.

    Returns a list of tuples containing:
        start time
        end time
        phones (annotator 1)
        phones (annotator 2)
        POS tag
    """
    WORD = re.compile(r'\s*(\d+\.\d+)\s+\d+\s+(.*)\s*$')
    words = []
    with open(path) as f:
        lineno = 1
        while next(f).strip() != '#':
            lineno += 1
        for line in f:
            if (m := WORD.match(line)) is None:
                print(f'{path}:{lineno} Invalid format: {line}')
            else:
                t = float(m.group(1))
                word, phones1, phones2, pos = m.group(2).split('; ')
                words.append((t, word, phones1, phones2, pos))
            lineno += 1
    return [(t, words[i+1][0], word, phones1, phones2, pos)
            for i, (t, word, phones1, phones2, pos) in enumerate(words[:-1])]


def iterate_utterances(words, pause_max):
    utt = []
    for i, (t0, t1, word, phones1, phones2, pos) in enumerate(words):
        if pos == 'null':
            pass
        else:
            if utt and (t0 - utt[-1][1]) > pause_max:
                yield utt
                utt = []
            utt.append((t0, t1, word, pos))
    if utt:
        yield utt


def read_buckeye(path, pause_max=0.3):
    words = read_words(path)
    if (m := re.match(r's(\d\d)\d\d[ab].words', os.path.basename(path))):
        speaker = int(m.group(1))
    else:
        raise ValueError(f'Invalid file name format: {path}')
    return {'utterances': [
                {'t_start': utt[0][0],
                 't_end': utt[-1][1],
                 'words': [
                     {'form': word,
                      't_start': t0,
                      't_end': t1,
                      'pos': pos}
                     for (t0, t1, word, pos) in utt
                     ]
                 } for utt in iterate_utterances(words, pause_max)
                ],
            'speaker': speaker
            }

if __name__ == '__main__':
    import sys
    import pprint
    pprint.pp(read_buckeye(sys.argv[1]))

