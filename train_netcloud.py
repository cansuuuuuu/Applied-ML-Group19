
import tensorflow as tf
import tensorflow.keras.layers as layers
from project_name.data.loading import load_dataset, split

from project_name.models.netcloud import NetCloud


MODEL_PATH = "project_name/models/netcloud"
BATCH_SIZE = 32


def make_dataset(images, labels, shuffle = True):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(labels),
            reshuffle_each_iteration=True,
        )

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def cross_validation():
    """
    Run cross-validation on the model
    """
    X_train, y_train, X_test, y_test, classes = load_dataset()

    train_ds = make_dataset(X_train, y_train)

    fold_results = []

    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skfold.split(train_ds, train_ds), start=1):
        tf.keras.backend.clear_session()

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        train_ds = make_dataset(X_train_fold, y_train_fold, shuffle=True)
        val_ds = make_dataset(X_val_fold, y_val_fold, shuffle=False)

        model = NetCloud(num_classes=len(classes))

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30,
            verbose=1,
            callbacks=[early_stopping],
        )

        val_loss, val_acc = model.evaluate(val_ds, verbose=2)
        fold_results.append((val_loss, val_acc))

        model.plot_confusion_matrix(
            val_ds,
            class_names=classes,
            save_path=f"project_name/data/val_confusion_matrix_fold_{fold}.png",
        )

        model.plot_history(
            save_path=f"project_name/data/history_plot_fold_{fold}.png"
        )

        print(f"Fold {fold} validation loss: {val_loss:.4f}")
        print(f"Fold {fold} validation accuracy: {val_acc:.4f}")


    mean_val_loss = sum(x[0] for x in fold_results) / len(fold_results)
    mean_val_acc = sum(x[1] for x in fold_results) / len(fold_results)

    print("\n===== Cross-validation results =====")
    print("Classes:", classes)
    print(f"Mean validation loss: {mean_val_loss:.4f}")
    print(f"Mean validation accuracy: {mean_val_acc:.4f}")

    return fold_results, history


def run_netcloud():

    X_train, y_train, X_test, y_test, classes = load_dataset()
    X_train, X_val, y_train, y_val = split(X_train, y_train)

    train_ds = make_dataset(X_train, y_train)
    val_ds = make_dataset(X_val, y_val)
    test_ds = make_dataset(X_test, y_test)

    model = NetCloud( num_classes=len(classes))

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.summary()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        verbose=1,
        callbacks=[early_stopping],
    )

    val_loss, val_acc = model.evaluate(val_ds, verbose=2)
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)

    model.plot_confusion_matrix(
        val_ds,
        class_names=classes,
        save_path="project_name/data/val_confusion_matrix.png",
    )

    model.model.save(MODEL_PATH)

    print("Classes:", classes)
    print("Validation loss:", val_loss)
    print("Validation accuracy:", val_acc)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
    print(f"Saved CNN model to {MODEL_PATH}")

    return model, history, (test_loss, test_acc)


if __name__ == "__main__":
    #run_netcloud()
    cross_validation()