## Update 18 March 2026 ##

1. The script `cross_surprisal_acl_march_update.py` is now updated and can be used for training.
2. The folder `ENGKJV` need to be downloaded on your local computer before training.
3. Set the variable `write_to_csv` and `write_to_json` in the `main` function to `True` to save the result locally. The `.csv` files are needed for the `surp_visualization.py` script. Saving `.csv` files can be avoided, but then the `.json` files have to be converted to Dataframes before visualizing the data.  
4. Run the `surp_visualization.py` script on the terminal: `python3 surp_visualization.py [FILE1.CSV] [FILE2.CSV]`.
