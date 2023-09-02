from starlette.testclient import TestClient
from services.gradcam.main import app
from tensorflow import keras

client = TestClient(app)

img_path = keras.utils.get_file(
    "cat_and_dog.jpg",
    "https://storage.googleapis.com/petbacker/images/blog/2017/dog-and-cat-cover.jpg",
)
img_size = (299, 299)
img = keras.preprocessing.image.load_img(img_path, target_size=img_size)
array = keras.preprocessing.image.img_to_array(img)

response = client.post("/heatmap", json={"img_array": array.tolist()})
