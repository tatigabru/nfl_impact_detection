# make impact longer
"""
video_labels_with_impact = train_labels.fillna(0)[train_labels['impact'] > 0]
train_labels['long_impact'] = train_labels['impact']
for row in video_labels_with_impact[['video','frame','label']].values:
    frames = np.arange(-30, 30) + row[1]
    train_labels.loc[(train_labels['video'] == row[0])
                                 & (train_labels['frame'].isin(frames))
                                 & (train_labels['label'] == row[2]), 'long_impact'] = 1


train_labels.to_csv(data_fp + 'cache/train_labels_cache_impact_30.csv')
"""


def add_bottom_right(df):
    df['right'] = df['left'] + df['width']
    df['bottom'] = df['top'] + df['height']
    return df


def pad_boxes(df, alpha=0.1):
    df['left'] = df['left'] - alpha * df['width']
    df['right'] = df['right'] + alpha * df['width']
    df['top'] = df['top'] - alpha * df['height']
    df['bottom'] = df['bottom'] + alpha * df['height']
    return df


def iou(bbox1, bbox2):
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union