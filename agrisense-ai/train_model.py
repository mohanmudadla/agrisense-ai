import argparse
import json
from pathlib import Path

import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description="Train a transfer-learning plant disease classifier.")
    parser.add_argument("--data_dir", required=True, help="Directory with one subfolder per class.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_dir", default="models")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Dataset directory not found: {data_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=42,
        image_size=(args.image_size, args.image_size), batch_size=args.batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=42,
        image_size=(args.image_size, args.image_size), batch_size=args.batch_size
    )
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(args.image_size, args.image_size, 3),
        include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(args.image_size, args.image_size, 3))
    x = augment(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "plant_disease_model.keras")
    (out / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"Saved model and class labels to: {out.resolve()}")
    print("Review validation metrics before using this model. Accuracy on the same dataset may not generalize to real field photos.")

if __name__ == "__main__":
    main()
