from IPython import embed
import csv
import pandas as pd
import datetime

filename = "/Users/ismglv/dev/lda2vec/metadata.csv"

import csv

def metadata():
    metadata = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metadata.append([row['filename'], datetime.datetime.strptime(row['date'], '%Y-%M-%d'), row['date_short']])


    return pd.DataFrame(metadata, columns=['filename', 'date', 'date_short'])

