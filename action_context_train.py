from models import vgg_action, vgg_context
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import pandas as pd
import os

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

parser = argparse.ArgumentParser(description='tune vgg16 network on new dataset')


parser.add_argument(
    "--classes",
    type=int,
    default=7,
    help="number of classes in target dataset")

parser.add_argument(
    "--model-type",
    choices=['action_aware', 'context_aware'],
    default='action_aware',
    help="action-aware model or context-aware model")

parser.add_argument(
    "--epochs",
    default=32,
    type=int,
    help="number of epochs")

parser.add_argument(
    "--samples-per-epoch",
    default=None,
    type=int,
    help="samples per epoch, default=all")

parser.add_argument(
    "--save-model",
    metavar="<prefix>",
    default=None,
    type=str,
    help="save model at the end of each epoch")

parser.add_argument(
    "--save-best-only",
    default=True,
    action='store_true',
    help="only save model if it is the best so far")

parser.add_argument(
    "--num-val-samples",
    default=None,
    type=int,
    help="number of validation samples to use (default=all)")

parser.add_argument(
    "--fixed-width",
    default=224,
    type=int,
    help="crop or pad input images to ensure given width")

parser.add_argument(
    "--seed",
    default=10,
    type=int,
    help="random seed")

parser.add_argument(
    "--workers",
    default=1,
    type=int,
    help="number of data preprocessing worker threads to launch")

parser.add_argument(
    "--device",
    default=0,
    type=int,
    help="GPU device to be used")

parser.add_argument(
    "--learning-rate",
    default=0.001,
    type=float,
    help="initial/fixed learning rate")

parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="batch size")


args = parser.parse_args()

gpus = tf.config.list_physical_devices('GPU')

try:
    tf.config.set_visible_devices(gpus[args.device],'GPU')
    print('///////////////////////////////')
    print(gpus[args.device])
    print('///////////////////////////////')
except:
    print('///////////////////////////////')
    print('No GPU found')
    print('///////////////////////////////')

correct_model = False

if args.model_type == 'action_aware':
    model = vgg_action(args.classes, input_shape=(args.fixed_width,args.fixed_width,3))
    correct_model = True
elif args.model_type == 'context_aware':
    model = vgg_context(args.classes, input_shape=(args.fixed_width, args.fixed_width, 3))
    correct_model = True
else:
    print("Wrong model type name!")

model_weights_path = 'data/model_weights'
if not os.path.exists(model_weights_path):
    os.mkdir(model_weights_path)
    
if os.path.exists(args.save_model):
    model = load_model(args.save_model)
    
if correct_model:
    df = pd.read_csv('data/Le2iFall/OpenPose/Frames_label.csv')
    datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.1
        )

    train_generator = datagen.flow_from_dataframe(
            dataframe=df,
            directory='data/Frames',
            x_col='video',
            y_col='label',
            target_size=(args.fixed_width, args.fixed_width),
            batch_size=args.batch_size,
            class_mode='categorical',
            subset='training')

    validation_generator = datagen.flow_from_dataframe(
            dataframe=df,
            directory='data/Frames',
            x_col='video',
            y_col='label',
            target_size=(args.fixed_width, args.fixed_width),
            batch_size=args.batch_size,
            class_mode='categorical',
            subset='validation')

    if not os.path.exists(args.save_model): 
        sgd = SGD(lr=args.learning_rate, decay=0.005, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = []

    if args.save_model:
        callbacks.append(ModelCheckpoint(args.save_model,
                                         verbose=1,
                                         monitor='accuracy',
                                         save_best_only=args.save_best_only
                                         )
                        )

    samples_per_epoch = args.samples_per_epoch or train_generator.samples // args.batch_size
    samples_per_epoch -= (samples_per_epoch % args.batch_size)
    num_val_samples = args.num_val_samples or validation_generator.samples // args.batch_size


    print("Starting to train...")
    model.fit(train_generator,
              steps_per_epoch=train_generator.samples // args.batch_size,
              verbose=1,
              callbacks=callbacks,
              epochs=args.epochs,
              workers=args.workers,
              shuffle=True,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples // args.batch_size)

    if args.model_type == 'action_aware':
        model.save_weights(f'{model_weights_path}/action_aware_vgg16_final.h5')
    else:
        model.save_weights(f'{model_weights_path}/context_aware_vgg16_final.h5')


