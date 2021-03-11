from visualize import annotate_image
from matplotlib import pyplot as plt
import numpy as np
from config import *
import imageio
import cv2


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


def annotate_box_centers(image_fp, boxes, how='box'):
    # Set label colors for bounding boxes
    img = imageio.imread(image_fp)
    COLOR1 = (0, 0, 255)  # Black

    players = boxes['label'].unique()
    for player in players:
        playerdf = boxes[boxes['label'] == player]
        points = [(int(box.left + box.width/2), int(box.top + box.height/2)) for j, box in playerdf.iterrows()]
        print(points)
        for prev_point, point in zip(points, points[1:]):
            # Add a box around the helmet
            # Note that cv2.rectangle requires us to specify the top left pixel and the bottom right pixel
            cv2.line(img, prev_point, point, color=COLOR1, thickness=2)
    # Display the image with bounding boxes added
    plt.imshow(img)
    plt.show()


def annotate_box_seq(image_fp, boxes, how='box'):
    # Set label colors for bounding boxes
    img = imageio.imread(image_fp)
    COLOR1 = (0, 0, 255)  # Black

    cmap = np.concatenate([plt.cm.rainbow(np.linspace(0, 1, len(boxes)//2)), plt.cm.rainbow(np.linspace(1, 0, len(boxes) - len(boxes)//2))])

    count = 0
    for j, box in boxes.iterrows():
        # Add a box around the helmet
        # Note that cv2.rectangle requires us to specify the top left pixel and the bottom right pixel
        cv2.rectangle(img, (box.left, box.top), (box.left + box.width, box.top + box.height), color=cmap[count]*255, thickness=1)
        count += 1

    # Display the image with bounding boxes added
    plt.imshow(img)
    plt.show()


def sample_impact_trajectories(df, video, n=5, window=10, step=2, how='center'):
    #videos = gtdf['video'].unique()[:n]
    videodf = df[df['video'] == video]
    impactdf = videodf.query("impact == 1 and visibility > 0 and confidence > 1")
    frames = sorted(impactdf['frame'].unique())[:n]
    for frame in frames:
        windowdf = videodf[videodf['frame'].isin(list(np.arange(frame - window, frame + window + 1, step)))]
        players = impactdf[impactdf['frame'] == frame]['label'].unique()
        movements = windowdf[windowdf['label'].isin(players)]
        image_name = video.split('.')[0] + '_' + str(frame) + '.png'
        image_fp = frames_fp + image_name
        annotate_box_centers(image_fp, movements, how=how)


if __name__ == '__main__':
    import pandas as pd
    from postprocess import apply_thresh, keep_box_overlaps
    preddf = pd.read_csv(project_fp + 'data/pred/public_kernel_impact_validation.csv')
    train_labels = pd.read_csv(train_labels_fp)
    train_labels = train_labels.query("frame != 0")

    gtdf = train_labels#.query("impact == 1 and visibility > 0 and confidence > 1")
    preddf = apply_thresh(preddf, 0.4)
    #compare(gtdf, preddf, n=3)
    sample_impact_trajectories(gtdf, "57684_001985_Endzone.mp4", n=10, how='center')



