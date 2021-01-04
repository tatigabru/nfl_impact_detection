# Imports
from typing import List, Dict, Optional, Tuple, Union, Type, Callable
from collections import defaultdict, namedtuple
import cv2
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from evaluate import evaluate_df
# from pytorch_toolbelt.utils import image_to_tensor, fs, to_numpy, rgb_image_from_tensor
from ensemble_boxes import nms, soft_nms, weighted_boxes_fusion, non_maximum_weighted
from postprocessing import keep_maximums

BOX_COLOR = (0, 255, 255)

FOLD0 = ['57584_002674_Endzone.mp4', '57584_002674_Sideline.mp4',
 '57586_001934_Endzone.mp4', '57586_001934_Sideline.mp4',
 '57594_000923_Endzone.mp4', '57594_000923_Sideline.mp4',
 '57680_002206_Endzone.mp4', '57680_002206_Sideline.mp4',
 '57686_002546_Endzone.mp4', '57686_002546_Sideline.mp4',
 '57784_001741_Endzone.mp4', '57784_001741_Sideline.mp4',
 '57787_003413_Endzone.mp4', '57787_003413_Sideline.mp4',
 '57904_001367_Endzone.mp4', '57904_001367_Sideline.mp4',
 '57912_001325_Endzone.mp4', '57912_001325_Sideline.mp4',
 '57913_000218_Endzone.mp4', '57913_000218_Sideline.mp4',
 '57992_000350_Endzone.mp4', '57992_000350_Sideline.mp4',
 '58005_001254_Endzone.mp4', '58005_001254_Sideline.mp4',
 '58005_001612_Endzone.mp4', '58005_001612_Sideline.mp4',
 '58048_000086_Endzone.mp4', '58048_000086_Sideline.mp4',
 '58102_002798_Endzone.mp4', '58102_002798_Sideline.mp4']

PREDS_DIR = '../../preds'
MIX = ['../../preds/3dnn_predictions_fold0.csv',
       '../../preds/densenet121_val_impactp01_fold0.csv',]
      # '../../preds/densenet201_val_impactp01_fold2.csv']

PREDS = [f'../../preds/densenet121_no_keepmax_fold{fold}.csv' for fold in range(4)]
PRED_HITS = [f'../../preds/densenet121_hits_impactp01_fold{fold}.csv' for fold in range(4)]

TRACKING_IOU_THRESHOLD = 0.24
TRACKING_FRAMES_DISTANCE = 9
IMPACT_THRESHOLD_SCORE = 0.35

weights = [1, 1]
iou_thr = 0.2
skip_box_thr = 0.16

best_metric = -1
best_params = None
skip_box_wbf_params = [0.13 + 0.02*i for i in range(10)]
iou_wbf_params = [0.15 + 0.05*i for i in range(5)]
dist_params = [7] 
track_iou_params = [0.15 + 0.05*i for i in range(7)]
impact_thres_params = [0.31 + 0.01*i for i in range(8)]


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_boxes(image, image_id, boxes_list, scores_list, labels_list, image_size=(720, 1280)):
    thickness = 1
    colors_to_use = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (125, 125, 125), (206, 116, 112)]
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            x1 = int(image_size[1] * boxes_list[i][j][0])
            y1 = int(image_size[0] * boxes_list[i][j][1])
            x2 = int(image_size[1] * boxes_list[i][j][2])
            y2 = int(image_size[0] * boxes_list[i][j][3])
            cv2.rectangle(image, (x1, y1), (x2, y2), colors_to_use[i], int(thickness))
    plt.figure(figsize=(12,6))        
    plt.imshow(image) 
    plt.savefig(f'../../ensembling/{image_id}_bboxes.png')
    plt.show()
    #show_image(image)


def gen_color_list(model_num, labels_num):
    color_list = np.zeros((model_num, labels_num, 3))
    colors_to_use = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0)]
    total = 0
    for i in range(model_num):
        for j in range(labels_num):
            color_list[i, j, :] = colors_to_use[total]
            total = (total + 1) % len(colors_to_use)
    return color_list


def run_wbf(predictions, image_index, image_size=512, iou_thr=0.55, skip_box_thr=0.7, weights=None):
    boxes = [prediction[image_index]['boxes'].data.cpu().numpy()/(image_size-1) for prediction in predictions]
    scores = [prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]) for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels


def load_image_preds(image_id: str, preds: pd.DataFrame, image_shape=(720, 1280)):
    """
    Get bboxes, scores and labels for a single frame from preds DataFrame
    """
    df = preds[preds['image_name'] == image_id]
    # print(df.head())    
    boxes = df[['left', 'top', 'width', 'height']].values
    # xyxy
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2] 
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]    
    # normalize
    image_height, image_width = image_shape
    coords2norm = np.array([[1.0 / image_width, 1.0 / image_height, 1.0 / image_width, 1.0 / image_height]])
    norm2pixels = np.array([[image_width, image_height, image_width, image_height]])
    boxes = boxes.astype(float)
    boxes[:, 0] = boxes[:, 0]/image_width 
    boxes[:, 2] = boxes[:, 2]/image_width
    boxes[:, 1] = boxes[:, 1]/image_height
    boxes[:, 3] = boxes[:, 3]/image_height  
    #boxes = [b * coords2norm for b in boxes]    
    # print(boxes)
    scores = df.scores.values
    labels = df.label.values
    # print(f'boxes: {boxes} \n scores: {scores} \n labels: {labels}')
    return boxes, scores, labels


def wbf_image_preds(image_id, dfs, weights, iou_thr, skip_box_thr):
    """Ensemble boxes for a single image"""
    boxes_list, scores_list, labels_list = [], [], []
    # combine all preds for an image
    for df in dfs:
        boxes, scores, labels = load_image_preds(image_id, df)
        boxes_list.append(boxes) 
        scores_list.append(scores) 
        labels_list.append(labels)
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    # print(f'wbf boxes: {boxes} \n wbf scores: {scores} \n wbf labels: {labels}')  
    return boxes, scores, labels


def plot_wbf_image_preds(image_path: str, image_id: str, dfs, weights, iou_thr, skip_box_thr):
    """Ensemble boxes for a single image"""
    boxes_list, scores_list, labels_list = [], [], []
    # combine all preds for an image
    for df in dfs:
        boxes, scores, labels = load_image_preds(image_id, df)
        boxes_list.append(boxes) 
        scores_list.append(scores) 
        labels_list.append(labels)
    # plot all bboxes
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    show_boxes(image.copy(), image_id[:-4] + '_raw', boxes_list, scores_list, labels_list)
    # print(boxes_list, scores_list, labels_list)
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    print(f'wbf boxes: {boxes} \n wbf scores: {scores} \n wbf labels: {labels}') 
    show_boxes(image.copy(), image_id[:-4] + '_wbf', [boxes], [scores], [labels.astype(np.int32)])
    return boxes, scores, labels


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df['label'] = 1
    df['image_name'] = df['video'].str.replace('.mp4', '') + '_' + df['frame'].astype(str) + '.png'
    df["right"] = df["left"] + df["width"]
    df["bottom"] = df["top"] + df["height"]
    return df


def add_bottom_right(df: pd.DataFrame) -> pd.DataFrame:
    df['right'] = df['left'] + df['width']
    df['bottom'] = df['top'] + df['height']
    return df


def add_width_height(df: pd.DataFrame) -> pd.DataFrame:
    df['width'] = df['right'] - df['left']
    df['height'] = df['bottom'] - df['top'] 
    return df


def combine_preds_wbf(images: list, df: pd.DataFrame, dfs, image_size, weights, iou_thr, skip_box_thr) -> pd.DataFrame:
    """
    Helper, to combine raw predicitons for all images
    Args:
        images (List(str)): list of image names
        df: pd.DataFrame : dataframe for combined predicitons
        dfs (List(pd.DataFrame)): list of dataframes with predicitons to combine 
    """
    row = 0
    for image_id in images:
        # for youtube dataset
        #gameKey, playID, view = 0, 0, 0
        #video, frame =  image_id.split('_')
        #video = video + '.mp4'    
        # for competition dataset    
        gameKey, playID, view, frame = image_id.split('_')[:4]
        video = f'{gameKey}_{playID}_{view}.mp4'
        boxes, scores, labels = wbf_image_preds(image_id, dfs, weights, iou_thr, skip_box_thr)
        for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
            df.loc[row,"gameKey"] = gameKey
            df.loc[row,"playID"] = int(playID)
            df.loc[row,"view"] = view
            df.loc[row,"video"] = video
            df.loc[row,"frame"] = int(frame[:-4])
            df.loc[row,"left"] = int(x1*image_size[1])
            df.loc[row,"width"] = int((x2 - x1)*image_size[1])
            df.loc[row,"top"] = int(y1*image_size[0])
            df.loc[row,"height"] = int((y2 - y1)*image_size[0])
            df.loc[row,"right"] = int(x2*image_size[1])
            df.loc[row,"bottom"] = int(y2*image_size[0])
            df.loc[row,"scores"] = score
            df.loc[row,"label"] = label 
            df.loc[row,"image_name"] = image_id 
            row += 1
    return df


def test_load_image_preds(df):
    df = preprocess_df(df)
    print(df.head())
    image_ids = df['image_name'].unique()
    print(len(image_ids), image_ids[:5])
    image_id = image_ids[1]
    boxes, scores, labels = load_image_preds(image_id, preds = df)
    print(f'boxes: {boxes} \n scores: {scores} \n labels: {labels}')
    print(len(scores), len(labels), len(boxes))


def test_wbf(image_id, dfs, weights, iou_thr, skip_box_thr):
    print(f'image_id: {image_id}')
    boxes, scores, labels = wbf_image_preds(image_id, dfs, weights, iou_thr, skip_box_thr)
    print(f'wbf boxes: {boxes} \n wbf scores: {scores} \n wbf labels: {labels}') 
    # competiton data
    # image_path = f'../../data/test_preds/labeled_{image_id}'
    # youtube data
    image_path = f'../../data/hits_images/{image_id}'
    boxes, scores, labels =  plot_wbf_image_preds(image_path, image_id, dfs, weights, iou_thr, skip_box_thr)


def grid_search_wbf(dfs: list, gtdf: pd.DataFrame, images: list, weights: list, save_dir = '../../ensembling') -> dict:
    """
    Helper to find optimal WBF parameters
    """
    image_size = (720, 1280)
    best_metric = 0
    best_iou, best_skip = 0, 0
    num = 0
    for skip_box_thr in skip_box_wbf_params:
        for iou_thr in iou_wbf_params:  
            print(f'EXPERIMENT {num}, iou wbf thres {iou_thr}, skip {skip_box_thr}')  
            # combine WBF for all frames    
            df_combo = pd.DataFrame(columns = dfs[0].columns)
            df_combo = combine_preds_wbf(images, df_combo, dfs, image_size, weights, iou_thr, skip_box_thr)            
            print('Apply filtering after...') 
            df_combo = df_combo[df_combo.scores > IMPACT_THRESHOLD_SCORE]
            print('Apply postprocessing...')  
            df_keepmax = keep_maximums(df_combo, iou_thresh=TRACKING_IOU_THRESHOLD, dist=TRACKING_FRAMES_DISTANCE)
            #df_keepmax.to_csv('../../preds/keepmax_wbf_densenet121.csv', index = False) 
            prec, rec, f1 = evaluate_df(gtdf, df_keepmax, video_names=None, impact=True)
            print(f"EXPERIMENT {num}, thres {IMPACT_THRESHOLD_SCORE} \n Precision {prec}, recall {rec}, f1 {f1}")
            num += 1
            if f1 > best_metric:
                best_metric = f1 
                best_iou = iou_thr
                best_skip = skip_box_thr
                # log results
                out = open(f"{save_dir}/params_{iou_thr}_{skip_box_thr}.txt", 'w')
                out.write('skip_box_thr: {}\n'.format(skip_box_thr))
                out.write('weights: {}\n'.format(weights))
                out.write('iou_thr: {}\n'.format(iou_thr))
                out.write('precision: {}\n'.format(prec))
                out.write('recall: {}\n'.format(rec))
                out.write('f1: {}\n'.format(f1))
                out.close()
                print('Current best metric: {}'.format(best_metric))
                print(f'Current best params: wbf_iou {best_iou}, skip threshold {best_skip}')
    print('Best metric: {}'.format(best_metric))
    print(f'Best params: wbf_iou {best_iou}, skip threshold {best_skip}')
    results = {}
    results['best_f1'] = best_metric
    results['wbf_iou'] = best_iou
    results['best_skip'] = best_skip
    return results


def grid_search_tracking(dfs, gtdf: pd.DataFrame, images: list, weights: list, save_dir = '../../ensembling'):
    """
    Helper to find optimal tracking parameters
    """
    image_size = (720, 1280)
    best_metric = 0
    best_iou, best_dist = 0, 2
    num = 0
    for dist in dist_params:
        for track_iou_thr in track_iou_params:  
            print(f'EXPERIMENT {num}, iou track thres {track_iou_thr}, dist {dist}')  
            # combine WBF for all frames    
            df_combo = pd.DataFrame(columns = dfs[0].columns)
            df_combo = combine_preds_wbf(images, df_combo, dfs, image_size, weights, iou_thr=0.2, skip_box_thr=0.17)            
            print('Apply filtering after...') 
            df_combo = df_combo[df_combo.scores > IMPACT_THRESHOLD_SCORE]
            print('Apply postprocessing...')  
            df_keepmax = keep_maximums(df_combo, iou_thresh=track_iou_thr, dist=dist)
            #df_keepmax.to_csv('../../preds/keepmax_wbf_densenet121.csv', index = False) 
            prec, rec, f1 = evaluate_df(gtdf, df_keepmax, video_names=None, impact=True)
            print(f"EXPERIMENT {num}, iou track thres {track_iou_thr}, dist {dist}, thres {IMPACT_THRESHOLD_SCORE} \n Precision {prec}, recall {rec}, f1 {f1}")
            num += 1
            if f1 > best_metric:
                best_metric = f1 
                best_iou = track_iou_thr
                best_dist = dist
                # log results
                out = open(f"{save_dir}/params_{track_iou_thr}_{dist}.txt", 'w')
                out.write('skip_box_thr: {}\n'.format(skip_box_thr))
                out.write('weights: {}\n'.format(weights))
                out.write('iou_thr: {}\n'.format(iou_thr))
                out.write(f'iou track thres: {track_iou_thr}\n')
                out.write(f'distance: {dist}\n')
                out.write(f'impact thres {IMPACT_THRESHOLD_SCORE}\n')
                out.write('precision: {}\n'.format(prec))
                out.write('recall: {}\n'.format(rec))
                out.write('f1: {}\n'.format(f1))
                out.close()
    print('Best metric: {}'.format(best_metric))
    print(f'Best params: track_iou {best_iou}, distance {best_dist}')


def grid_search_all(dfs, gtdf: pd.DataFrame, images: list, weights: list, save_dir = '../../ensembling'):
    image_size = (720, 1280)
    dist = TRACKING_FRAMES_DISTANCE
    best_metric = 0
    best_iou, best_iou_wbf, best_impact_th, best_skip = 0, 0, 0, 0
    num = 0
    for impact_thres in impact_thres_params:
        for skip_box_thr in skip_box_wbf_params:
            for iou_thr in iou_wbf_params:  
                for track_iou_thr in track_iou_params:  
                    print(f'EXPERIMENT {num}')
                    # combine WBF for all frames    
                    df_combo = pd.DataFrame(columns = dfs[0].columns)
                    df_combo = combine_preds_wbf(images, df_combo, dfs, image_size, weights, iou_thr, skip_box_thr)            
                    print('Apply filtering after...') 
                    df_combo = df_combo[df_combo.scores > impact_thres]
                    print('Apply postprocessing...')  
                    df_keepmax = keep_maximums(df_combo, iou_thresh=track_iou_thr, dist=dist)
                    #df_keepmax.to_csv('../../preds/keepmax_wbf_densenet121.csv', index = False) 
                    prec, rec, f1 = evaluate_df(gtdf, df_keepmax, video_names=None, impact=True)
                    print(f"EXPERIMENT {num}, iou track thres {track_iou_thr}, dist {dist}, thres {impact_thres}, wbf_iou_thr {iou_thr}, skip_box_thr {skip_box_thr} \n Precision {prec}, recall {rec}, f1 {f1}")
                    num += 1
                    if f1 > best_metric:
                        best_metric = f1 
                        best_iou = track_iou_thr
                        best_iou_wbf = iou_thr
                        best_impact_th = impact_thres
                        best_skip = skip_box_thr
                        # log results
                        out = open(f"{save_dir}/params_{num}.txt", 'w')
                        out.write('skip_box_thr: {}\n'.format(skip_box_thr))
                        # out.write('weights: {}\n'.format(weights))
                        out.write('iou_thr: {}\n'.format(iou_thr))
                        out.write(f'iou track thres: {track_iou_thr}\n')
                        out.write(f'distance: {dist}\n')                    
                        out.write(f'impact thres {impact_thres}\n')
                        out.write('precision: {}\n'.format(prec))
                        out.write('recall: {}\n'.format(rec))
                        out.write('f1: {}\n'.format(f1))
                        out.close()
            print('Best metric: {}'.format(best_metric))
            print(f'Best params: track_iou {best_iou}, best_iou_wbf {best_iou_wbf}, besk skip box {best_skip}, best impact thres {best_impact_th}')


def combine_image_ids(dfs) -> list:
    # Accumulate all image_ids for all predictions
    images = set()
    for df in dfs:
        image_ids = df['image_name'].unique()        
        images = images.union(image_ids)
    images = list(sorted(images))
    print(len(images), images[:5])
    return images


def do_val_preds(dfs, gtdf: pd.DataFrame):
    # list preds dataframes
    dfs = [pd.read_csv(preds_file) for preds_file in MIX]
    dfs = [preprocess_df(df.copy()) for df in dfs]    
    #print('Apply filtering before...')
    #dfs = [df[df.scores > IMPACT_THRESHOLD_SCORE] for df in dfs]
    #dfs = [df[df.frame > 30] for df in dfs]
    images = combine_image_ids(dfs)
    # combine WBF for all frames
    image_size = (720, 1280)
    df_combo = pd.DataFrame(columns = dfs[0].columns)
    df_combo = combine_preds_wbf(images, df_combo, dfs, image_size, weights, iou_thr, skip_box_thr)            
    print('Apply filtering after...') 
    df_combo = df_combo[df_combo.scores > IMPACT_THRESHOLD_SCORE]
    # apply postprocessing    
    print('Apply postprocessing...')
    df_keepmax = keep_maximums(df_combo, iou_thresh=TRACKING_IOU_THRESHOLD, dist=TRACKING_FRAMES_DISTANCE)
    #df_keepmax.to_csv('../../preds/keepmax_wbf_densenet121.csv', index = False)
    video_names = gtdf['video'].unique()
    pred_video = df_keepmax['video'].unique()
    print('Number of videos for evaluation:', len(video_names), ' predicted video', len(pred_video))
    prec, rec, f1 = evaluate_df(gtdf, df_keepmax, video_names=None, impact=True)
    print(f"Precision {prec}, recall {rec}, f1 {f1}")


def grid_impact_threshold(dfs, gtdf: pd.DataFrame):        
    video_names = gtdf['video'].unique()
    print('Number of videos for evaluation:', len(video_names))
    image_size = (720, 1280)
    num = 0
    for impact_thres in impact_thres_params:
        print(f'EXPERIMENT {num}, thres {impact_thres}')
       # print('Apply filtering before...')
       # dfs = [df[df.scores > impact_thres] for df in dfs]      
       # images = combine_image_ids(dfs)               
        # combine WBF for all frames  
        print('Combined raw preds...')      
        df_combo = pd.DataFrame(columns = dfs[0].columns)
        df_combo = combine_preds_wbf(images, df_combo, dfs, image_size, weights, iou_thr, skip_box_thr)
        print('Apply filtering after...') 
        df_combo = df_combo[df_combo.scores > impact_thres]
        print('Apply postprocessing...')  
        df_keepmax = keep_maximums(df_combo, iou_thresh=TRACKING_IOU_THRESHOLD, dist=TRACKING_FRAMES_DISTANCE)
        
        prec, rec, f1 = evaluate_df(gtdf, df_keepmax, video_names=None, impact=True)
        print(f"'EXPERIMENT {num}, thres {impact_thres} \n Precision {prec}, recall {rec}, f1 {f1}")
        num += 1


if __name__ == "__main__":     
    # list preds dataframes
    dfs = [pd.read_csv(preds_file) for preds_file in MIX]
    dfs = [preprocess_df(df.copy()) for df in dfs]
    dfs = [df[df.frame > 30] for df in dfs] # remove first frames (and last)
    print(dfs[1].head())

   # gtdf = pd.read_csv('../../preds/hits_meta.csv')
    gtdf = pd.read_csv('../../data/train_labels.csv')
    gtdf = gtdf[gtdf['video'].isin(FOLD0)]
    gtdf = gtdf[(gtdf['impact'] == 1) &(gtdf['confidence'] > 1)&(gtdf['visibility']> 0)]
    gtdf = add_bottom_right(gtdf)
    print('Ground thruth: \n', gtdf.head())
    video_names = gtdf['video'].unique()
    print('Number of videos for evaluation:', len(video_names))
    images = combine_image_ids(dfs) 
    # test and plot WBF        
    #test_wbf(images[20], dfs, weights, iou_thr, skip_box_thr)

    #do_val_preds(dfs, gtdf)
   # grid_impact_threshold(dfs, gtdf)
   # results = grid_search_wbf(dfs, gtdf, images, weights, save_dir = '../../ensembling') 
    grid_search_tracking(dfs, gtdf, images, weights, save_dir = '../../ensembling') 
    #grid_search_all(dfs, gtdf, images, weights, save_dir = '../../ensembling')