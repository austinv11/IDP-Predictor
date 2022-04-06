import pandas as pd
import sys

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 1:
        print("Convert parquet to csv")
        print('Usage: parquet2csv.py <parquet_file>')

    df = pd.read_parquet(args[0])
    df.to_csv(args[0].replace(args[0].split('.')[-1], 'csv'), index=False)
