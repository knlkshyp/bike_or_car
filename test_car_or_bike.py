from fastai.vision.all import *

image_path = 'input/'   # Add filename with extension after input/
model_path = 'model/car_or_bike_model.pkl'

model = load_learner(model_path)

prediction, _, probs = model.predict(PILImage.create(f'{image_path}'))

print(f'This is a {prediction}')