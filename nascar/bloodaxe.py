class SinglePlayHelmetCropDataset(MiniClipDataset):
    """
    Probably we dont need it
    """
    def __init__(
        self,
        # arguments from single_video_to_dataset_args_by_box function
        frames: List[str], # image fnames
        bboxes: List,
        labels: List,
        overlaps: List,
        frame_numbers: List,
        df_indices: List, # indices to keep track where box came from

        # clip characteristics
        clip_length: int,
        clip_frame_step: int,
        clip_center_frame: int,
        num_samples: int,

        # transforms
        spatial_transform: A.ReplayCompose,
        color_transform: Union[A.Compose, A.ReplayCompose, A.BasicTransform],

        # sampling
        reverse_p=0.0,
        impact_p=0.5,
        overlap_p=0.1,
    ):
        # arguments from single_video_to_dataset_args_by_box function
        self.images = frames
        self.image_ids = list(map(os.path.basename, frames))
        self.frame_numbers = frame_numbers
        self.bboxes = bboxes
        self.labels = labels
        self.overlaps = overlaps
        self.bbox_ids = df_indices
        self.num_boxes = len(df_indices)

        # mini clip characteristics
        self.clip_length = clip_length
        self.num_samples = num_samples
        self.clip_frame_step = clip_frame_step
        self.clip_center_frame = clip_center_frame

        # transforms
        self.spatial_transform = spatial_transform
        self.color_transform = color_transform
        #self.box_coder = box_coder # dont know if i need it

        # sampling
        self.normalize = A.Normalize(DATASET_MEAN, DATASET_STD)
        self.reverse_p = reverse_p # dont know if i need it
        self.impacts = np.array(self.labels)
        self.impact_p = impact_p
        self.overlap_p = overlap_p


    def __len__(self):
        return self.num_samples

    def __repr__(self):
        f"SinglePlayHelmetDataset(clip_length{self.clip_length}, num_samples={self.num_samples}, num_impact_frames={sum(self.impacts)})"

    def __getitem__(self, index):
        indexes_with_impacts = np.flatnonzero(self.impacts)

        # better to remove frames too close to end or start before calling this function
        if indexes_with_impacts.any() and random.random() < self.impact_p:
            impact_index = random.choice(indexes_with_impacts)
        else:
            impact_index = random.randrange(self.num_boxes)

        impact_frame = self.frame_numbers[impact_index]
        start = impact_frame - self.clip_center_frame
        end = start + self.clip_length

        frames = np.arange(start, end)
        bbox = self.bboxes[impact_index]

        # dont know if I need it
        #if random.random() < self.reverse_p:
        #    frames = frames[::-1]

        return self._get_mini_clip(frames=frames, impact_index=impact_index)


def single_video_to_dataset_args_by_box(df: pd.DataFrame):
    """
    df: one video dataframe
    """
    bboxes = []
    labels = []
    overlaps = []
    frame_numbers = []
    image_fnames = []
    df_indices = []
    videos = np.unique(df.video)

    if len(videos) != 1:
        raise ValueError("Must have only one video" + ",".join(videos))
    df = add_overlaps(df, pad=2)

    for i, row in df.iterrows():
        frame = int(row["frame"])
        image_fname = row["image_fname"]
        x1, y1, x2, y2 = row["left"], row["top"], row["right"], row["bottom"]
        has_impact = row["impact"] == 1
        has_overlap = row['has_overlap']

        image_fnames.append(image_fname)
        bboxes.append((x1, y1, x2, y2))
        labels.append(int(has_impact))
        frame_numbers.append(frame)
        df_indices.append(i)
        overlaps.append(has_overlap)

    return image_fnames, bboxes, labels, overlaps, frame_numbers, df_indices