import csv
import os
import pandas as pd

# os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main(file_name, color, text):
    
    FILE = pd.read_csv(file_name, encoding='cp949')
    SEARCH = str()

    SEARCH = FILE[ ( 
        ( FILE['color_front'] == color.lower() )& 
        ( FILE['text_front'] == text.upper()) ) 
        ]
    
    return(SEARCH.values)
