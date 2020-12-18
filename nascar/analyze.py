from visualize import annotate_image

from config import *

def show_sample(df, n=5):
    videos = df['video'].unique()[:n]
    #videos = ["57583_000082_Endzone.mp4"]
    #df = df[df['video'].isin(videos)]
    for video in videos:
        videodf = df[df['video'] == video].sort_values('frame')
        frames = videodf['frame'].unique()[:5]
        for frame in frames:
            framedf = videodf[videodf['frame'] == frame]
            image_name = video.split('.')[0] + '_' + str(frame) + '.png'
            image_fp = frames_fp + image_name
            annotate_image(image_fp, framedf)


def show_tp(gtdf, preddf, n=5):
    pass


def show_fp(gtdf, preddf, n=5):
    pass


def show_fn(gtdf, preddf, n=5):
    pass


def compare(gtdf, preddf, n=5):
    # show ground truth boxes and predicted boxes
    videos = preddf['video'].unique()[:n]
    for video in videos:
        pred_videodf = preddf[preddf['video'] == video]
        gt_videodf = gtdf[gtdf['video'] == video]
        pred_frames = pred_videodf['frame'].unique()
        gt_frames = gt_videodf['frame'].unique()
        frames = sorted(list(set(pred_frames) | set(gt_frames)))[:n]
        for frame in frames:
            pred_framedf = pred_videodf[pred_videodf['frame'] == frame]
            gt_framedf = gt_videodf[gt_videodf['frame'] == frame]
            image_name = video.split('.')[0] + '_' + str(frame) + '.png'
            image_fp = frames_fp + image_name
            annotate_image(image_fp, gt_framedf, pred_framedf)


if __name__ == '__main__':
    import pandas as pd
    from postprocess import apply_thresh, keep_box_overlaps
    preddf = pd.read_csv(project_fp + 'data/pred/public_kernel_impact_validation.csv')
    train_labels = pd.read_csv(train_labels_fp)
    train_labels = train_labels.query("frame != 0")

    gtdf = train_labels.query("impact == 1 and visibility > 0 and confidence > 1")
    preddf = apply_thresh(preddf, 0.4)
    compare(gtdf, preddf, n=3)



