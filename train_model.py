import tensorflow as tf
from tqdm import tqdm
from src.model import build_model
from src.dataset import build_yolo_dataset
from src.loss import YOLOLoss
import time
import json
import os

def train(model, optimizer, train_dataset, loss_fn, epochs = 50 , steps_per_epoch = None, VERBOSE_STEPS = True):

    @tf.function
    def train_step(images, targets):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss, box_loss, obj_loss, no_obj_loss, class_loss = \
                   loss_fn.compute_losses_components(targets, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, box_loss, obj_loss, no_obj_loss, class_loss

    history = {
    "loss": [],
    "box_loss": [],
    "obj_loss": [],
    "no_obj_loss": [],
    "class_loss": [],
    "lr": []
    }
    patience = 10
    wait = 0
    best_loss = float("inf")
    lr_wait = 0
    lr_patience = 4
    lr_factor = 0.1
    min_lr = 1e-6

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    num_batches = tf.data.experimental.cardinality(train_dataset).numpy()
    print(f"Training on {num_batches} batches per epoch")

    for epoch in range(epochs):

        start_time = time.time()

        print(f"\nEpoch {epoch+1}")
        epoch_total_loss = 0.0
        epoch_box_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_no_obj_loss = 0.0
        epoch_class_loss = 0.0

        num_steps = 0

        progress = tqdm(train_dataset, total = num_batches, desc=f"Epoch {epoch+1}")

        for step, (images, targets) in enumerate(progress):
            if steps_per_epoch is not None and step >= steps_per_epoch:
                break
            num_steps += 1
            loss, box_loss, obj_loss, no_obj_loss, class_loss = train_step(images, targets)

            epoch_total_loss += loss.numpy()
            epoch_box_loss += box_loss.numpy()
            epoch_obj_loss += obj_loss.numpy()
            epoch_no_obj_loss += no_obj_loss.numpy()
            epoch_class_loss += class_loss.numpy()
            
            progress.set_postfix(loss=f"{loss.numpy():.2f}")
            if VERBOSE_STEPS:
              print(
              f"Step {step+1}: "
              f"Loss={loss.numpy():.2f}, "
              f"Box={box_loss.numpy():.2f}, "
              f"Obj={obj_loss.numpy():.2f}, "
              f"NoObj={no_obj_loss.numpy():.2f}, "
              f"Class={class_loss.numpy():.2f}"
               )
        progress.close()

        avg_total_loss = epoch_total_loss / num_steps
        avg_box_loss = epoch_box_loss / num_steps
        avg_obj_loss = epoch_obj_loss / num_steps
        avg_no_obj_loss = epoch_no_obj_loss / num_steps
        avg_class_loss = epoch_class_loss / num_steps

        print("-" * 40)
        print(f"Epoch {epoch+1} Summary")
        print(f"Average Total Loss : {avg_total_loss:.2f}")
        print(f"Average Box Loss   : {avg_box_loss:.2f}")
        print(f"Average Obj Loss   : {avg_obj_loss:.2f}")
        print(f"Average NoObj Loss : {avg_no_obj_loss:.2f}")
        print(f"Average Class Loss : {avg_class_loss:.2f}")

        history["loss"].append(float(avg_total_loss))
        history["box_loss"].append(float(avg_box_loss))
        history["obj_loss"].append(float(avg_obj_loss))
        history["no_obj_loss"].append(float(avg_no_obj_loss))
        history["class_loss"].append(float(avg_class_loss))
        history["lr"].append(float(optimizer.learning_rate.numpy()))
        print(f"Learning rate: " f"{optimizer.learning_rate.numpy():.2e}")

        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {elapsed_time:.2f} seconds")
        min_delta = 0.1
        if avg_total_loss < best_loss - min_delta:
            best_loss = avg_total_loss
            wait = 0
            lr_wait = 0
            checkpoint.save(os.getenv("CHECKPOINT_PATH", "checkpoints/best_model"))

            print("Loss improved! Best model saved!")
        else:
            wait += 1
            lr_wait += 1
         '''
        if lr_wait >= lr_patience:

            old_lr = optimizer.learning_rate.numpy()

            new_lr = max(old_lr * lr_factor, min_lr)

            optimizer.learning_rate.assign(new_lr)

            print(
                 f"Reducing learning rate: "
                f"{old_lr:.2e} -> {new_lr:.2e}"
                )
            lr_wait = 0
        
        if wait >= patience:
            print("Early stopping triggered!")
            break
         '''
    print('TRAINING COMPLETE!')
    with open(os.getenv("HISTORY_PATH", "history/history.json"), "w") as f:
        json.dump(history, f)
    return history

if __name__ == "__main__":
    yolo_dataset = build_yolo_dataset()

    model = build_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    train_dataset = yolo_dataset.map(
        lambda batch: (batch["image"], batch["targets"]),
        num_parallel_calls=tf.data.AUTOTUNE)

    loss_fn = YOLOLoss(
    lambda_box=5.0,
    lambda_obj=1.0,
    lambda_noobj=0.5,
    lambda_class=1.0)

    history = train(model, optimizer, train_dataset, loss_fn)
