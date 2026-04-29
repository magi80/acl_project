# Update: Surprisal Estimation (April 2026)

The script ``cross_test_20260326_approach_3.py`` implements the 3<sup>rd</sup> approach for surprisal estimation discussed in late March.

This script is an implementation of ``cross_test_20260326.py``, which I send to Robert on March 26. The ``Surprisal`` class now includes the debugged sanity-check functions ``get_sentences_statistics`` and ``get_words_statistics``. The ``main`` function is implemented with the flag ``test_single_paragraph`` which enables testing on individual, longer paragraphs. All other functions within the class remain unchanged.

**Note**: if errors persist when ``test`` is ``False``, assign ``txt.split()`` to the ``tokens`` variable at line 659 in ``get_words_statistics``, or disable both sanity-check functions. 

## Inference Example
The function ``create_word_level_window`` computes surprisal estimates based on the probability of the target word $W_{k}$ conditioned by its left context $k-n$, right context $k+n$, **and** its internal subtokens $t_{k,n}$.

If the target word $W_{k}$ is tokenized into several subtokens $t_{k, n}$, the left context is one word ($k-1$), and the right context is masked ($k+0$), the surprisal score $S$ for $W_k$ is the sum of the probability of the masked subtokens $t_{k, n}$ conditioned by the previous word $W_{k-1}$ **and** the previous subtoken $t_{k,n-1}$ within $W_{k}$:

$S(W_{k}) = -log_{2} P(t_{k, 1}|W_{k-1}) + -log_{2}P(t_{k,2}|W_{k-1}, t_{k, 1}) + -log_{2}P(t_{k,3}|W_{k-1}, t_{k,1}, t_{k,2}) + ... $

When $W_{k-1}$ consists of a word with multiple subtokens, all the subtokens of $W_{k-1}$ are visible under inference.

The function extracts the surprisal for each subtoken within the target word $W_k$ incrementally, masking each subtoken to the right of the current subtoken $t_{k, n}$.
