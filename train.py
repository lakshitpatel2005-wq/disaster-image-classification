import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

img_size = (224,224)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    "dataset",
    target_size=img_size,
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val = datagen.flow_from_directory(
    "dataset",
    target_size=img_size,
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(train.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train, validation_data=val, epochs=5)

model.save("disaster_model.h5")