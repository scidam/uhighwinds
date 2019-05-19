from keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import skimage.transform as trans

# ----------- Image cropping parameters -----------
IMG_SHAPE = (256, 256)

aug_pars = dict(rotation_range=0.2,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest')



# -------------------------------------------------

def get_train_images(batch_size, aug_parameters=aug_pars, seed=1):
    image_datagen = ImageDataGenerator(**aug_parameters)
    mask_datagen = ImageDataGenerator(**aug_parameters)

    image_generator = image_datagen.flow_from_directory(
                        'data/train/images',
                        classes=None,
                        class_mode=None,
                        color_mode="rgb",
                        target_size=IMG_SHAPE,
                        batch_size=batch_size,
                        save_to_dir=None,
                        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        'data/train/masks',
        classes=None,
        class_mode=None,
        color_mode="grayscale",
        target_size=IMG_SHAPE,
        batch_size=batch_size,
        save_to_dir=None,
        seed=seed)

    for img, mask in zip(image_generator, mask_generator):
        yield (img, mask)

