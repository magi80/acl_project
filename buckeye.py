"""Functions to import data from the Buckeye corpus.

The main function is `read_buckeye`, which reads one of the .words files from
the Buckeye corpus and returns a dict intended to be exported as JSON.
See the bottom of this file for a simple usage example.
"""

import re
import os

speaker_sex = {
        f'{i+1:02d}': sex
        for i, sex in enumerate(
            'ffmffmfffmmfmfmfffmffmmmfffmmmfmmmmmfmfm'.upper())}


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
    with open(path, 'rb') as f:
        data = f.read().replace(b'\r', b'')
    lines = str(data, 'utf-8').split('\n')
    lineno = 2 + lines.index('#')
    while lineno <= len(lines):
        line = lines[lineno-1].strip()
        if not line:
            pass
        elif (m := WORD.match(line)) is None:
            print(f'{path}:{lineno} Invalid format: {line}')
        else:
            t = float(m.group(1))
            # Now that we know the start time of this word, add that as the
            # end time of the preceding word, unless it already has one.
            # This allows us to use the start times of "words" that are not
            # kept, such as fillers and laughter.
            if words and len(words[-1]) == 5:
                words[-1] = (words[-1][0], t) + words[-1][1:]
            fields = m.group(2).split('; ')
            # The number of fields is not 100% consistent in the corpus, but
            # this covers the most common case:
            if fields[-1] == 'null':
                pass
            elif len(fields) == 4:
                word, phones1, phones2, pos = fields
                words.append((t, word, phones1, phones2, pos))
            else:
                print(f'{path}:{lineno} Invalid format: {line}')
        lineno += 1
    return words
    # return [(t, words[i+1][0], word, phones1, phones2, pos)
    #         for i, (t, word, phones1, phones2, pos) in enumerate(words[:-1])]


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
    if (m := re.match(r's(\d\d)(\d\d[ab]).words', os.path.basename(path))):
        speaker = m.group(1)
        recording = m.group(2)
    else:
        raise ValueError(f'Invalid file name format: {path}')
    # as far as I can tell, the recordings were all done at the same session,
    # but we keep them separate for now as in the original data
    return {'session': speaker+recording,
            'utterances': [
                {'t_start': utt[0][0],
                 't_end': utt[-1][1],
                 'words': [
                     {'form': word,
                      't_start': t0,
                      't_end': t1,
                      'pos': pos}
                     for (t0, t1, word, pos) in utt
                     ],
                 'speaker': {
                     'id': speaker,
                     'sex': speaker_sex[speaker],
                     },
                 } for utt in iterate_utterances(words, pause_max)
                ],
            }

if __name__ == '__main__':
    import sys
    import pprint
    pprint.pp(read_buckeye(sys.argv[1]))

