import pandas as pd
from collections import Counter


def main():
    df = pd.read_csv('cnn3d/output/raw_predictions_fold0.csv')
    df = df[df['helmet_probas'] >=0.4]
    counts = Counter(df.groupby(['video', 'frame']).size().values.tolist())
    print(counts)
    print(len(df))
if __name__ == '__main__':
    main()