from fastai.vision.all import *

dataset_path = Path('dataset')  # Add training dataset with sub directories as 'bike' & 'car'
model_path = Path('model')
model_path.mkdir(exist_ok=True)

dataset_sub_dirs = ['bike', 'car']

for sub_dir_name in dataset_sub_dirs :
    source_path = dataset_path / sub_dir_name
    dest_path = dataset_path / sub_dir_name
    resize_images(source_path, max_size=400, dest = dest_path)

item_tfms = [Resize(192)]
batch_tfms = [Flip(), Rotate(10), Zoom(0.1)]

dls = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    splitter = RandomSplitter(valid_pct = 0.2, seed = 42),
    get_y = parent_label,
    item_tfms = item_tfms,
    batch_tfms = batch_tfms
).dataloaders(dataset_path)

model = vision_learner(dls, resnet18, metrics = [error_rate, accuracy])

model.fine_tune(3)

model.validate()

model.export(model_path/'car_or_bike_model.pkl')