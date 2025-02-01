from fastai.vision.all import *

pathOfTheImage = 'input/<IMAGE_NAME_WITH_EXTENSION>'
path = Path('dataset')

resize_images(path/'bike', max_size=400, dest=path/'bike')
resize_images(path/'car', max_size=400, dest=path/'car')

item_tfms = [Resize(192)]
batch_tfms = [Flip(), Rotate(10), Zoom(0.1)]

dls = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    splitter = RandomSplitter(valid_pct = 0.2, seed = 42),
    get_y = parent_label,
    item_tfms = item_tfms,
    batch_tfms = batch_tfms
).dataloaders(path)

learn = vision_learner(dls, resnet18, metrics = error_rate)

learn.load('car_or_bike_classifier_resnet18')

pred, _, probs = learn.predict(PILImage.create(f'{pathOfTheImage}'))

print(f'This is a {pred}')