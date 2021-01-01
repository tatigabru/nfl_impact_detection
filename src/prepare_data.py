import sys
import os
import cv2
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


DATA_DIR = '../../data/'
META_FILE = os.path.join(DATA_DIR, 'train_labels.csv')


def make_images_from_video(video_name, video_dir, out_dir, limit = None):
    """Helper to get image frames from videos"""
    video_path=f"{video_dir}/{video_name}"
    video_name = os.path.basename(video_path)
    vidcap = cv2.VideoCapture(video_path)
    print(video_path)
    frame = 0
    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        frame += 1
        # print(frame)
        image_path = f'{out_dir}/{video_name}'.replace('.mp4',f'_{frame}.png')
        success = cv2.imwrite(image_path, img)
        if not success:
            raise ValueError("couldn't write image successfully")
        if limit and frame > limit:
            print(f'Made maximum: {limit} frames')
            break
    

def write_frames(video_path):
    video_name = os.path.basename(video_path)
    output_base_path = "../../data/images_test"
    os.makedirs(os.path.join(output_base_path, video_name), exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    frame = 0
    while True:
        more_frames, img = vidcap.read()
        if not more_frames:
            break
        frame += 1
        img_name = "{}".format(frame).zfill(6) + ".png"
        success = cv2.imwrite(os.path.join(output_base_path, video_name, img_name), img)
        if not success:
            raise ValueError("couldn't write image successfully")


def make_test_frames():
    test_videos = os.listdir("../../data/test")    
    pool = Pool()
    pool.map(write_frames, map(lambda video_name: f"{video_dir}/{video_name}", test_videos))


def make_train():
    video_dir = '../../data/train'
    video_labels = pd.read_csv(META_FILE).fillna(0)
    uniq_video = video_labels.video.unique()    
    out_dir = '../../data/train_images_full/'
    os.makedirs(out_dir, exist_ok=True)
    for video_name in uniq_video:
        make_images_from_video(video_name, video_dir, out_dir)


def make_test():
    video_dir = '../../data/pred_test/densenet121'
    uniq_video = os.listdir(video_dir)
    out_dir = '../../data/test_preds/'
    os.makedirs(out_dir, exist_ok=True)
    for video_name in uniq_video:
        make_images_from_video(video_name, video_dir, out_dir)


def make_video_from_frames(video_name, imade_dir, start, stop):
    VIDEO_CODEC = "MP4V"
    writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*VIDEO_CODEC), 30, (1280, 720))
    for frame in range(start, stop+1):
        image_name = f'nfl_helmets_hits_{frame}.png'
        img_path = os.path.join(imade_dir, image_name) 
        img = cv2.imread(img_path)  
        print(img_path)      
        writer.write(img)
    cv2.destroyAllWindows()
    writer.release()


def create_meta_nfl_helmets_hits():   
    """
    Create labels for youtube video fragments
    """ 
    df = pd.DataFrame(columns =  ["video", "frame", "impact_type", 'left','top','right','bottom'])
    rows = []
    rows.append(['hit279.mp4', 279-267+1, 'head', 754, 325, 768, 335]) #only one visible
    rows.append(['hit598.mp4', 598-574+1, 'head', 591, 214, 641, 255]) 
    rows.append(['hit598.mp4', 598-574+1, 'head', 618, 207, 672, 268]) 
    rows.append(['hit598.mp4', 616-574+1, 'grass', 677, 379, 771, 498])
    rows.append(['hit598.mp4', 623-574+1, 'grass', 739, 399, 859, 533])
    rows.append(['hit744.mp4', 744-736+1, 'head', 389, 237, 471, 333])
    rows.append(['hit744.mp4', 744-736+1, 'head', 463, 237, 539, 354])
    rows.append(['hit744.mp4', 762-736+1, 'grass', 797, 403, 942, 569])
    rows.append(['hit744.mp4', 768-736+1, 'grass', 599, 415, 765, 561])
    rows.append(['hit926.mp4', 926-895+1, 'head', 419, 240, 507, 353])
    rows.append(['hit926.mp4', 926-895+1, 'head', 500, 245, 571, 348])
    rows.append(['hit926.mp4', 1028-895+1, 'grass', 809, 404, 953, 559])
    rows.append(['hit926.mp4', 1053-895+1, 'grass', 609, 408, 773, 555])    
    rows.append(['hit1184.mp4', 1284-1160+1, 'shoulder', 757, 292, 773, 316])
    rows.append(['hit1184.mp4', 1313-1160+1, 'head', 663, 256, 680, 279])
    rows.append(['hit2771.mp4', 2771-2600+1, 'head', 465, 204, 480, 223])
    rows.append(['hit2771.mp4', 2771-2600+1, 'head', 477, 211, 488, 225])    
    rows.append(['hit3194.mp4', 3147-3032+1, 'head', 491, 358, 513, 373])
    rows.append(['hit3194.mp4', 3147-3032+1, 'head', 481, 334, 512, 366])
    rows.append(['hit3194.mp4', 3194-3032+1, 'head', 637, 241, 668, 274])
    rows.append(['hit3194.mp4', 3194-3032+1, 'head', 625, 250, 638, 276]) # not very visible, behind the first one
    rows.append(['hit3613.mp4', 3613-3474+1, 'head', 612, 270, 627, 280])
    rows.append(['hit3613.mp4', 3613-3474+1, 'head', 627, 269, 645, 288])
    rows.append(['hit4457.mp4', 4457-4422+1, 'shoulder', 560, 320, 582,345])
    rows.append(['hit4934.mp4', 4934-4802+1, 'head', 721, 282, 746, 310])
    rows.append(['hit4934.mp4', 4934-4802+1, 'head', 735, 282, 750, 304])
    rows.append(['hit5319.mp4', 5319-5164+1, 'head', 781, 324, 800, 342])
    rows.append(['hit5319.mp4', 5319-5164+1, 'head', 793, 321, 809, 335])
    rows.append(['hit5926.mp4', 5926-5832+1, 'head', 593, 260, 614, 293])
    rows.append(['hit5926.mp4', 5926-5832+1, 'head', 612, 261, 634, 293])
    rows.append(['hit6401.mp4', 6401-6214+1, 'head', 777, 267, 816, 311])
    rows.append(['hit6401.mp4', 6401-6214+1, 'head', 755, 269, 786, 312])
    rows.append(['hit9585.mp4', 9585-9447+1, 'head', 761, 315, 785, 348])
    rows.append(['hit9585.mp4', 9585-9447+1, 'head', 773, 328, 801, 355])
    rows.append(['hit9585.mp4', 9615-9447+1, 'head', 506, 300, 527, 324])
    rows.append(['hit9585.mp4', 9615-9447+1, 'head', 523, 303, 550, 332])
    for idx, row in enumerate(rows, start = 1):
        df.loc[idx] = row

    #df['left','top','right','bottom'] = 0, 0, 0, 0
    print(df.head)

    return df


def make_youtube_frames(video_dir = '../../data/youtube', video_name = 'nfl_helmets_hits.mp4', out_dir = '../../data/helmet_hits/'):
    os.makedirs(out_dir, exist_ok=True)
    make_images_from_video(video_name, video_dir, out_dir) 



if __name__ == "__main__":  
    make_test()  
    

    #video_name = out_dir + 'hit4457.mp4'  
    #make_video_from_frames(video_name, out_dir, start=4422, stop=4514)
    
    #df = create_meta_nfl_helmets_hits()
    #df.to_csv( out_dir + 'hits_meta.csv', index=False)