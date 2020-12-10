EfficientDet [Train] Mixup + Cutmix + Stratified K-Fold
This notebook is mainly based on 2Class Object Detection Training, with addition to mixup, cutmix, and stratified k-fold capability. Some winners in the previous object detection competition (Global Wheat Detection) use the combination of mixup, cutmix and stratified k-fold in their solution. So I guess, these tools would be useful in this competition as well. The implementation of cutmix and mixup is based on this kernel.

Please kindly upvote this kernel if you find it useful
Data Preparation
video_labels = pd.read_csv('/kaggle/input/nfl-impact-detection/train_labels.csv').fillna(0)
video_labels_with_impact = video_labels[video_labels['impact'] > 0]
for row in tqdm(video_labels_with_impact[['video','frame','label']].values):
    frames = np.array([-4,-3,-2,-1,1,2,3,4])+row[1]
    video_labels.loc[(video_labels['video'] == row[0]) 
                                 & (video_labels['frame'].isin(frames))
                                 & (video_labels['label'] == row[2]), 'impact'] = 1
video_labels['image_name'] = video_labels['video'].str.replace('.mp4', '') + '_' + video_labels['frame'].astype(str) + '.png'
video_labels = video_labels[video_labels.groupby('image_name')['impact'].transform("sum") > 0].reset_index(drop=True)
video_labels['impact'] = video_labels['impact'].astype(int)+1
video_labels['x'] = video_labels['left']
video_labels['y'] = video_labels['top']
video_labels['w'] = video_labels['width']
video_labels['h'] = video_labels['height']
video_labels.head()
Stratified K-Fold
'''
np.random.seed(0)
video_names = np.random.permutation(video_labels.video.unique())
valid_video_len = int(len(video_names)*0.2)
video_valid = video_names[:valid_video_len]
video_train = video_names[valid_video_len:]
images_valid = video_labels[ video_labels.video.isin(video_valid)].image_name.unique()
images_train = video_labels[~video_labels.video.isin(video_valid)].image_name.unique()
images_all = video_labels[ video_labels.video.isin(video_names)].image_name.unique()
'''
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df_folds = video_labels[['image_name']].copy()
df_folds.loc[:, 'bbox_count'] = 1
df_folds = df_folds.groupby('image_name').count()
df_folds.loc[:, 'video'] = video_labels[['image_name', 'video']].groupby('image_name').min()['video']
df_folds.loc[:, 'stratify_group'] = np.char.add(
    df_folds['video'].values.astype(str),
    df_folds['bbox_count'].apply(lambda x: f'_{x // 20}').values.astype(str),
)

df_folds.loc[:, 'fold'] = 0
for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
Create Images from Videos
def mk_images(video_name, video_labels, video_dir, out_dir, only_with_impact=True):
    video_path=f"{video_dir}/{video_name}"
    video_name = os.path.basename(video_path)
    vidcap = cv2.VideoCapture(video_path)
    if only_with_impact:
        boxes_all = video_labels.query("video == @video_name")
        print(video_path, boxes_all[boxes_all.impact > 1.0].shape[0])
    else:
        print(video_path)
    frame = 0
    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        frame += 1
        if only_with_impact:
            boxes = video_labels.query("video == @video_name and frame == @frame")
            boxes_with_impact = boxes[boxes.impact > 1.0]
            if boxes_with_impact.shape[0] == 0:
                continue
        img_name = f"{video_name}_frame{frame}"
        image_path = f'{out_dir}/{video_name}'.replace('.mp4',f'_{frame}.png')
        _ = cv2.imwrite(image_path, img)
uniq_video = video_labels.video.unique()
video_dir = '/kaggle/input/nfl-impact-detection/train'
out_dir = 'train_images'
!mkdir -p $out_dir
for video_name in uniq_video:
    mk_images(video_name, video_labels, video_dir, out_dir)
Albumentations
def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(500, 720), height=720, width=720, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ], p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.JpegCompression(quality_lower=85, quality_upper=95, p=0.2),
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0)
            ],p=0.1),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )
    

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )