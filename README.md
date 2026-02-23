## Update 23 February 2026

1. The script `cross_surprisal_acl_update.py` can be run locally. Download the folder `ENGKJV` and change the `self.root` path. Use `test` set to `True` for running single paragraphs. Choose a number between 1 and 260.
2. The `txt` files contain normalized text, except for the period mark, uppercased. 
3. The function `estimate_sentence_surprisal` has been corrected.
4. The file `inference` illustrates how surprisal is estimated for each context.
5. Two new `.png` files are added, illustrating surprisal for one chapter (260) at both paragraph and sentence level.
6. The file `cross_surp-output.txt` illustrates surprisal estimation on screen during inference
