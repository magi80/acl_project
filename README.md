## Update 18 March 2026 ##

1. The script `cross_surprisal_acl_march_update.py` is now updated and can be used for training.
2. The folder `ENGKJV` need to be downloaded on your local computer before training.
3. Set the variable `Test` to `False` in the `main` function to run all the 260 chapters/paragraphs of the Bible collection ENGKJV. You can evebtually run a test on single chapters (line 622 and 623) by choosing a key number between 1-260.
4. Set the variable `write_to_csv` and `write_to_json` in the `main` function to `True` to save the result locally. The `.csv` files are needed for the `surp_visualization.py` script. Saving `.csv` files can be avoided, but then the `.json` files have to be converted to Dataframes before visualizing the data.  
5. Run the `surp_visualization.py` script on the terminal: `python3 surp_visualization.py [FILE1.CSV] [FILE2.CSV]`.
