import os

from .dataset import IterableImageDataset, ImageDataset

# output : /data/.../imagenet/train(val)
def _search_split(root, split):
    # look for sub-folder with name of split in root(train or split) and use that if it exists
    ################## flow102 ###############################
    if len(root.split('flow102')) != 1:
        if split == 'train':
            try_root = os.path.join(root, 'train_dir')
            return try_root
        elif split == 'validation':
            try_root = os.path.join(root, 'test_dir')
            return try_root
    ################## flow102 ###############################

    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root
    if split_name == 'validation':
        try_root = os.path.join(root, 'val')
        if os.path.exists(try_root):
            return try_root
    return root


# 2번 실행됨, create_dataset(dataset, data_dir, train(val)_split, is_training=T(F))
def create_dataset(name, root, split='validation', search_split=True, is_training=False, batch_size=None, **kwargs):
    name = name.lower()
    if name.startswith('tfds'):
        ds = IterableImageDataset(
            root, parser=name, split=split, is_training=is_training, batch_size=batch_size, **kwargs)
    else:
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        kwargs.pop('repeats', 0)  # FIXME currently only Iterable dataset support the repeat multiplier
        if search_split and os.path.isdir(root):
            root = _search_split(root, split)
        ds = ImageDataset(root, parser=name, **kwargs)
    return ds
