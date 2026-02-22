# Get a list of xlsx files on a given path, compute the indicators and save them in path with the same name
import os
import pandas as pd
import openpyxl
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from indicateurs_opt import Indicator  # the Indicator class is defined in a module named indicateurs

def process_file(file, in_path, out_path):
    """Process a single file: read, compute indicators, and save"""
    try:
        df = pd.read_excel(os.path.join(in_path, file))
        # test if the dataframe has the expected nb of lines (175)
        if df.shape[0] < 175:
            return f"Warning: {file} does not have 175 lines, it has {df.shape[0]} lines."
        
        indicator = Indicator(df)
        df_indicators = indicator.construire_arnaud(df)
        # save the dataframe with the same name in out_path
        df_indicators.to_excel(os.path.join(out_path, file), index=False)
        return f"Success: {file}"
    except Exception as e:
        return f"Error processing {file}: {str(e)}"

def main():
    in_path = 'datas'
    out_path = 'datasindAV'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    files = [f for f in os.listdir(in_path) if f.endswith('.xlsx')]
    
    # Use all available CPU cores
    num_processes = cpu_count()
    print(f"Processing {len(files)} files using {num_processes} CPU cores...")
    
    # Create a partial function with fixed in_path and out_path
    process_func = partial(process_file, in_path=in_path, out_path=out_path)
    
    # Process files in parallel with progress bar
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_func, files), total=len(files)))
    
    # Print warnings and errors
    for result in results:
        if "Warning" in result or "Error" in result:
            print(result)

if __name__ == '__main__':
    main()





