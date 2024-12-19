#%%
import copick
import pandas as pd
import zarr
from matplotlib import pyplot as plt
import os
import json
import numpy as np
from keras import layers, models
from keras.api.losses import BinaryCrossentropy
from keras.api.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tqdm import tqdm
from numba import njit, prange
import datetime
import tensorflow as tf
from keras import backend as K

#%%
def get_copick_root(split):
    """
    Loads the copick configuration based on the split.

    Args:
        split (str): 'train' or 'test'.

    Returns:
        copick_root: The loaded copick configuration.
    """
    config_path = '../../copick_config.json' if split == 'train' else '../../copick_config_test.json'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    copick_root = copick.from_file(config_path)
    return copick_root

#%%
def get_static_tomogram(run_name, split='train', tomo_type='denoised', zarr_group_idx=0):
    """
    Retrieves a static tomogram from the copick dataset.

    Args:
        run_name (str): Name of the run.
        split (str): 'train' or 'test'.
        tomo_type (str): Type of tomogram.
        zarr_group_idx (int): Index of the Zarr group.

    Returns:
        np.ndarray: The tomogram data as a NumPy array.
    """
    copick_root = get_copick_root(split)
    run = copick_root.get_run(run_name)
    voxel_spacing = run.get_voxel_spacing(10.000)
    tomogram = voxel_spacing.get_tomogram(tomo_type)
    zarr_store = tomogram.zarr()
    zarr_group = zarr.open(zarr_store, mode='r')
    try:
        tomogram_vals = zarr_group[str(zarr_group_idx)][:]  # Use string keys if necessary
    except KeyError:
        raise KeyError(f"Zarr group index {zarr_group_idx} not found in the store.")
    return tomogram_vals

#%%
def get_label_locations(run_name, copick_root, voxel_spacing=10):
    """
    Extracts label locations from JSON files for a given run.

    Args:
        run_name (str): Name of the run.
        copick_root: Copick root object.
        voxel_spacing (float): Voxel spacing for normalization.

    Returns:
        dict: Dictionary with particle names as keys and arrays of locations as values.
    """
    picks_folder = os.path.join(copick_root.config.overlay_root, 'ExperimentRuns', run_name, 'Picks')
    if not os.path.exists(picks_folder):
        raise FileNotFoundError(f"Picks folder not found: {picks_folder}")

    picks = {}
    for json_file in os.listdir(picks_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(picks_folder, json_file)
            with open(json_path, 'r') as file:
                pick_data = json.load(file)
            particle_name = json_file[:-5]  # Remove '.json' extension
            picks[particle_name] = np.array([
                [
                    point['location']['x'] / voxel_spacing,
                    point['location']['y'] / voxel_spacing,
                    point['location']['z'] / voxel_spacing
                ]
                for point in pick_data.get('points', [])
            ])
    return picks

#%%
@njit(parallel=True)
def add_gaussian_to_heatmap_max(heatmap, z, y, x, kernel, half_size, depth, height, width):
    """
    Adds a Gaussian kernel to the heatmap at the specified (z, y, x) location using maximum.
    """
    for i in prange(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            for k in range(kernel.shape[2]):
                zi = z + i - half_size
                yi = y + j - half_size
                xi = x + k - half_size
                if 0 <= zi < depth and 0 <= yi < height and 0 <= xi < width:
                    if kernel[i, j, k] > heatmap[zi, yi, xi]:
                        heatmap[zi, yi, xi] = kernel[i, j, k]

#%%
def generate_heatmaps_optimized_max(df, tomogram_shape, particle_types):
    """
    Optimized function to generate 3D heatmaps for particle locations with maximum Gaussian spread.
    """
    depth, height, width = tomogram_shape
    num_classes = len(particle_types)
    heatmaps = np.zeros((depth, height, width, num_classes), dtype=np.float32)

    # Map particle types to channels
    particle_to_channel = {particle: i for i, particle in enumerate(particle_types)}

    # Group by 'particle' and 'radius'
    grouped = df.groupby(['particle', 'radius'])

    # Precompute Gaussian kernels for each group with adjusted sigma
    kernels = {}
    for (particle, radius), group in grouped:
        # Adjust sigma to ensure rapid decay within radius
        sigma = radius  # You may adjust this to radius / 3 for faster decay
        if sigma == 0:
            # Handle cases where radius is zero to avoid division by zero
            kernel = np.zeros((1, 1, 1), dtype=np.float32)
            kernel[0, 0, 0] = 1.0
            kernels[(particle, radius)] = (kernel, 0)
            continue

        # Define the size of the kernel; typically 6*sigma to capture the Gaussian effectively
        kernel_size = int(6 * sigma) + 1  # Ensure kernel size is odd
        half_size = kernel_size // 2
        x = np.arange(-half_size, half_size + 1)
        y = np.arange(-half_size, half_size + 1)
        z_axis = np.arange(-half_size, half_size + 1)
        zz, yy_grid, xx = np.meshgrid(z_axis, y, x, indexing='ij')
        kernel = np.exp(-(xx ** 2 + yy_grid ** 2 + zz ** 2) / (2 * sigma ** 2))
        kernel /= kernel.max()  # Normalize the kernel peak to 1
        kernels[(particle, radius)] = (kernel.astype(np.float32), half_size)

    # Iterate over groups and add Gaussian kernels to the heatmaps
    for (particle, radius), (kernel, half_size) in tqdm(kernels.items(), desc="Generating Heatmaps"):
        channel = particle_to_channel[particle]
        # Extract all particles in the current group
        mask = (df['particle'] == particle) & (df['radius'] == radius)
        particles = df[mask][['z', 'y', 'x']].to_numpy().astype(np.int32)  # Order: z, y, x

        # Add Gaussian kernels to the heatmap using maximum
        for z, y, x in particles:
            add_gaussian_to_heatmap_max(
                heatmaps[..., channel],
                z, y, x,
                kernel, half_size,
                depth, height, width
            )

    return heatmaps

#%%
def overlay_heatmaps_on_tomogram(tomogram_slice, heatmaps_slice, particle_types, threshold=0.3):
    """
    Overlays colored heatmaps onto a tomogram slice.

    Args:
        tomogram_slice (np.ndarray): 2D tomogram slice.
        heatmaps_slice (np.ndarray): 3D heatmap slice (height, width, num_classes).
        particle_types (list): List of particle types.
        threshold (float): Intensity threshold for heatmap visibility.

    Returns:
        None: Displays the overlay plot.
    """
    # Normalize tomogram slice to [0, 1]
    tomogram_normalized = (tomogram_slice - tomogram_slice.min()) / (tomogram_slice.max() - tomogram_slice.min())
    tomogram_rgb = np.stack([tomogram_normalized] * 3, axis=-1)  # Convert grayscale to RGB

    # Generate distinct colors for each particle type
    cmap = plt.cm.get_cmap('tab10', len(particle_types))
    particle_colors = [cmap(i)[:3] for i in range(len(particle_types))]

    # Initialize an RGB array for the heatmap overlay
    overlay = np.zeros_like(tomogram_rgb, dtype=np.float32)

    for i, particle_color in enumerate(particle_colors):
        heatmap = heatmaps_slice[..., i]
        mask = heatmap > threshold
        for c in range(3):
            overlay[..., c] += particle_color[c] * mask.astype(np.float32)

    # Combine the tomogram with the overlay
    combined = tomogram_rgb + overlay
    combined = np.clip(combined, 0, 1)

    plt.figure(figsize=(10, 8))
    plt.imshow(combined, interpolation='nearest')
    plt.axis('off')
    plt.title('Tomogram Slice with Heatmap Overlay')
    plt.show()

#%%
def prepare_detection_labels(df, max_detections, num_classes, tomogram_shape):
    """
    Prepare labels for 3D object detection.

    Args:
        df (pd.DataFrame): DataFrame containing particle locations and classes.
        max_detections (int): Maximum number of detections per tomogram.
        num_classes (int): Number of particle classes.
        tomogram_shape (tuple): Shape of the tomogram (depth, height, width).

    Returns:
        np.ndarray: Array of shape (batch_size, max_detections, num_classes + 3).
    """
    batch_size = df['run'].nunique()
    labels = np.zeros((batch_size, max_detections, num_classes + 3), dtype=np.float32)

    runs = df['run'].unique()

    for i, run in enumerate(runs):
        run_df = df[df['run'] == run]
        for j in range(min(len(run_df), max_detections)):
            particle = run_df.iloc[j]
            class_idx = particle_types.index(particle['particle'])
            labels[i, j, class_idx] = 1  # One-hot for class
            # Normalize coordinates to [0, 1]
            labels[i, j, num_classes:num_classes+3] = np.array([
                particle['x'] / tomogram_shape[2],
                particle['y'] / tomogram_shape[1],
                particle['z'] / tomogram_shape[0]
            ])
        # Remaining detections can remain as zeros (dummy detections)

    return labels

#%%
def build_3d_object_detection_model(input_shape, num_classes, max_detections):
    """
    Builds a 3D CNN model for object detection.

    Args:
        input_shape (tuple): Shape of the input data (depth, height, width, channels).
        num_classes (int): Number of particle classes.
        max_detections (int): Maximum number of detections.

    Returns:
        keras.Model: The constructed 3D CNN model.
    """
    inputs = layers.Input(shape=input_shape)

    # Backbone (Example Architecture)
    x = layers.Conv3D(32, (3,3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2,2,2), padding='same')(x)

    x = layers.Conv3D(64, (3,3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2,2,2), padding='same')(x)

    x = layers.Conv3D(128, (3,3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2,2,2), padding='same')(x)

    # Flatten and Dense Layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Detection Head
    # Each detection has (num_classes + 3) outputs: class probabilities + (x, y, z)
    detections = layers.Dense(max_detections * (num_classes + 3), activation='sigmoid')(x)

    # Reshape to (max_detections, num_classes + 3)
    detections = layers.Reshape((max_detections, num_classes + 3))(detections)

    model = models.Model(inputs, detections)
    return model

#%%
def object_detection_loss(y_true, y_pred):
    """
    Custom loss for 3D object detection.

    Args:
        y_true (tensor): Ground truth tensor of shape (batch_size, max_detections, num_classes + 3).
        y_pred (tensor): Predicted tensor of the same shape.

    Returns:
        tensor: Combined loss.
    """
    num_classes = y_true.shape[-1] - 3

    # Split predictions and ground truths
    y_true_classes = y_true[..., :num_classes]
    y_true_coords = y_true[..., num_classes:]

    y_pred_classes = y_pred[..., :num_classes]
    y_pred_coords = y_pred[..., num_classes:]

    # Compute mask where at least one class is present
    # Assuming one-hot encoding, sum across classes to get presence
    object_mask = K.sum(y_true_classes, axis=-1, keepdims=True)  # Shape: (batch_size, max_detections, 1)

    # Classification loss
    classification_loss = tf.keras.losses.binary_crossentropy(y_true_classes, y_pred_classes)

    # Regression loss (only for objects)
    regression_loss = tf.keras.losses.mean_squared_error(y_true_coords, y_pred_coords)
    regression_loss *= object_mask  # Apply mask

    # Combine losses
    total_loss = K.mean(classification_loss + regression_loss)
    return total_loss

#%%
def generate_detection_csv(predictions, run_names, particle_types, tomogram_shape, threshold=0.5, output_csv='detections.csv'):
    """
    Generates a CSV file with detections.

    Args:
        predictions (np.ndarray): Model predictions of shape (batch_size, max_detections, num_classes + 3).
        run_names (list): List of run names corresponding to each prediction.
        particle_types (list): List of particle types.
        tomogram_shape (tuple): Shape of the tomogram (depth, height, width).
        threshold (float): Confidence threshold to consider a detection valid.
        output_csv (str): Path to the output CSV file.

    Returns:
        None
    """
    max_detections = predictions.shape[1]
    num_classes = len(particle_types)
    data = []

    for i in range(predictions.shape[0]):
        run = run_names[i]
        detections = predictions[i]  # Shape: (max_detections, num_classes + 3)
        for j in range(max_detections):
            detection = detections[j]
            class_probs = detection[:num_classes]
            coords = detection[num_classes:]

            # Determine if this detection is valid based on class probabilities
            # For multi-class, take the class with the highest probability
            class_idx = np.argmax(class_probs)
            class_prob = class_probs[class_idx]

            if class_prob < threshold:
                continue  # Skip low-confidence detections

            particle_class = particle_types[class_idx]

            # Convert normalized coordinates back to original scale
            x = coords[0] * tomogram_shape[2]
            y = coords[1] * tomogram_shape[1]
            z = coords[2] * tomogram_shape[0]
            x, y, z = int(round(x)), int(round(y)), int(round(z))

            data.append({
                'run_number': run,
                'particle_type': particle_class,
                'x_coord': x,
                'y_coord': y,
                'z_coord': z
            })

    # Create DataFrame and save to CSV
    detections_df = pd.DataFrame(data)
    detections_df.to_csv(output_csv, index=False)
    print(f"Detections saved to {output_csv}")

#%%
# Load copick configurations
copick_root_train = get_copick_root('train')
copick_root_test = get_copick_root('test')

#%%
# Retrieve run names
training_runs = copick_root_train.runs
testing_runs = copick_root_test.runs

training_run_names = [run.name for run in training_runs]
testing_run_names = [run.name for run in testing_runs]
extra_training_run_names = list(set(training_run_names) - set(testing_run_names))

print(f'Training runs: {training_run_names}')
print(f'Testing runs: {testing_run_names}')
print(f'Extra training runs not in testing: {extra_training_run_names}')

#%%
# Load tomograms
tomograms_train = []
tomograms_test = []

for training_run_name in training_run_names:
    try:
        tomo = get_static_tomogram(split='train', run_name=training_run_name)
        tomograms_train.append({'run': training_run_name, 'tomogram': tomo})
    except Exception as e:
        print(f"Error loading tomogram for run {training_run_name}: {e}")

for testing_run_name in testing_run_names:
    try:
        tomo = get_static_tomogram(split='test', run_name=testing_run_name)
        tomograms_test.append({'run': testing_run_name, 'tomogram': tomo})
    except Exception as e:
        print(f"Error loading tomogram for run {testing_run_name}: {e}")

print(f'Number of train tomograms: {len(tomograms_train)}')
print(f'Number of test tomograms: {len(tomograms_test)}')

#%%
# Verify the loaded tomograms
if tomograms_train:
    sample_tomo = tomograms_train[0]['tomogram']
    print(f"Type: {type(sample_tomo)}")
    print(f"Element Type: {type(sample_tomo[0])}")
    print(f"Shape: {sample_tomo.shape}")
else:
    print("No training tomograms loaded.")

#%%
# Visualize a slice
if tomograms_train:
    slice_index = 90  # Adjust as needed
    if slice_index < tomograms_train[0]['tomogram'].shape[0]:
        plt.imshow(tomograms_train[0]['tomogram'][slice_index], cmap='gray')
        plt.title(f"Tomogram Slice {slice_index}")
        plt.colorbar()
        plt.show()
    else:
        print(
            f"Slice index {slice_index} out of range for tomogram with depth {tomograms_train[0]['tomogram'].shape[0]}")
else:
    print("No training tomograms to visualize.")

#%%
# Create DataFrame with particle locations
rows = []
for run in training_run_names:
    try:
        label_data = get_label_locations(run, copick_root_train)
        for particle, locations in label_data.items():
            matching_particles = [obj.radius for obj in copick_root_train.config.pickable_objects if
                                  obj.name == particle]
            if not matching_particles:
                print(f"No matching particle found for {particle} in run {run}")
                continue
            radius = matching_particles[0] / 10
            for location in locations:
                rows.append({
                    'run': run,
                    'particle': particle,
                    'x': location[0],
                    'y': location[1],
                    'z': location[2],
                    'radius': radius
                })
    except Exception as e:
        print(f"Error processing labels for run {run}: {e}")

particle_locations_df = pd.DataFrame(rows)
print(particle_locations_df.sample(5))
print(f"Total labeled particles: {len(particle_locations_df)}")

#%%
# Define particle types and tomogram shape
particle_types = sorted(particle_locations_df['particle'].unique())
if tomograms_train:
    tomogram_shape = tomograms_train[0]['tomogram'].shape
    print(f"Tomogram shape: {tomogram_shape}")
else:
    raise ValueError("No training tomograms available to determine shape.")

#%%
# Select a small 3D region for training and validation
# Define the region coordinates (adjust these based on your data)
train_z_start, train_z_end = 50, 150
train_y_start, train_y_end = 250, 350
train_x_start, train_x_end = 250, 350

val_z_start, val_z_end = 50, 150
val_y_start, val_y_end = 150, 250
val_x_start, val_x_end = 150, 250

# Extract the regions from the first tomogram
trainer_tom = tomograms_train[0]['tomogram'][train_z_start:train_z_end, train_y_start:train_y_end,
              train_x_start:train_x_end]
validator_tom = tomograms_train[0]['tomogram'][val_z_start:val_z_end, val_y_start:val_y_end, val_x_start:val_x_end]

print(f"Training region shape: {trainer_tom.shape}")
print(f"Validation region shape: {validator_tom.shape}")

#%%
# Filter labels for the training region
trainer_labels = particle_locations_df[
    (particle_locations_df['z'] >= train_z_start) & (particle_locations_df['z'] < train_z_end) &
    (particle_locations_df['y'] >= train_y_start) & (particle_locations_df['y'] < train_y_end) &
    (particle_locations_df['x'] >= train_x_start) & (particle_locations_df['x'] < train_x_end)
    ].copy()

# Adjust label coordinates relative to the extracted region
trainer_labels['z'] = trainer_labels['z'] - train_z_start
trainer_labels['y'] = trainer_labels['y'] - train_y_start
trainer_labels['x'] = trainer_labels['x'] - train_x_start

#%%
# Filter labels for the validation region
validator_labels = particle_locations_df[
    (particle_locations_df['z'] >= val_z_start) & (particle_locations_df['z'] < val_z_end) &
    (particle_locations_df['y'] >= val_y_start) & (particle_locations_df['y'] < val_y_end) &
    (particle_locations_df['x'] >= val_x_start) & (particle_locations_df['x'] < val_x_end)
    ].copy()

# Adjust label coordinates relative to the extracted region
validator_labels['z'] = validator_labels['z'] - val_z_start
validator_labels['y'] = validator_labels['y'] - val_y_start
validator_labels['x'] = validator_labels['x'] - val_x_start

print(f"Number of training labels: {len(trainer_labels)}")
print(f"Number of validation labels: {len(validator_labels)}")

#%%
# Generate heatmaps for the training region
if not trainer_labels.empty:
    trainer_heatmaps = generate_heatmaps_optimized_max(trainer_labels, trainer_tom.shape, particle_types)
else:
    trainer_heatmaps = np.zeros((trainer_tom.shape[0], trainer_tom.shape[1], trainer_tom.shape[2], len(particle_types)),
                                dtype=np.float32)

# Generate heatmaps for the validation region
if not validator_labels.empty:
    validator_heatmaps = generate_heatmaps_optimized_max(validator_labels, validator_tom.shape, particle_types)
else:
    validator_heatmaps = np.zeros(
        (validator_tom.shape[0], validator_tom.shape[1], validator_tom.shape[2], len(particle_types)), dtype=np.float32)

#%%
# Prepare input and label data by adding a batch dimension
X_train = np.expand_dims(trainer_tom.astype(np.float32)[..., np.newaxis], axis=0)  # Shape: (1, depth, height, width, 1)
# Y_train will be detection labels, not heatmaps
# Prepare detection labels
max_detections = 10  # Adjust based on maximum expected particles
num_classes = len(particle_types)
detection_labels_train = prepare_detection_labels(trainer_labels, max_detections, num_classes, trainer_tom.shape)

# Similarly for validation data
X_val = np.expand_dims(validator_tom.astype(np.float32)[..., np.newaxis], axis=0)  # Shape: (1, depth, height, width, 1)
detection_labels_val = prepare_detection_labels(validator_labels, max_detections, num_classes, validator_tom.shape)

print(f"Training input shape: {X_train.shape}")
print(f"Training detection label shape: {detection_labels_train.shape}")
print(f"Validation input shape: {X_val.shape}")
print(f"Validation detection label shape: {detection_labels_val.shape}")

#%%
# Define the list of run names for the training data
train_run_names = tomograms_train[0]['run']  # Assuming single run; adjust if multiple runs
train_run_names = [tomograms_train[0]['run']]  # Correcting to list
print(f"Training run names: {train_run_names}")

# Similarly for validation data
val_run_names = tomograms_train[0]['run']  # Assuming single run; adjust if multiple runs
val_run_names = [tomograms_train[0]['run']]  # Correcting to list
print(f"Validation run names: {val_run_names}")

#%%
# Build the 3D Object Detection Model
input_shape = X_train.shape[1:]  # (depth, height, width, channels)
output_shape = detection_labels_train.shape[1:]
print(f"Input Shape: {input_shape}")
print(f"Output Shape: {output_shape}")

model = build_3d_object_detection_model(input_shape=input_shape, num_classes=num_classes, max_detections=max_detections)
model.summary()

#%%
# Compile the Model with Custom Loss
model.compile(optimizer='adam',
              loss=object_detection_loss,
              metrics=['accuracy'])  # You can add more metrics if needed

#%%
# Define Callbacks
log_dir = f"./logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_object_detection_model.keras', monitor='val_loss', save_best_only=True)

#%%
# Train the Model on the Detection Task
epochs = 20  # Set to a suitable number based on your data
batch_size = 1  # Use batch_size=1 for minimal memory usage; increase if possible

history = model.fit(
    X_train, detection_labels_train,
    validation_data=(X_val, detection_labels_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[tensorboard_callback, early_stopping, model_checkpoint]
)

#%%
# Post-Training: Make Predictions and Save to CSV

# Make predictions on the validation data
predictions = model.predict(X_val)  # Shape: (batch_size, max_detections, num_classes + 3)

# Generate CSV for validation detections
generate_detection_csv(
    predictions=predictions,
    run_names=val_run_names,
    particle_types=particle_types,
    tomogram_shape=validator_tom.shape,
    threshold=0.5,  # Adjust threshold as needed
    output_csv='validation_detections.csv'
)

#%%
# Example: Visualize a Prediction on the Validation Data
if X_val.shape[0] > 0:
    # Select a sample from validation data
    sample_idx = 0
    tomogram_slice = X_val[sample_idx, :, :, :, 0]
    # No need for ground truth heatmap; we're doing object detection
    # Make a prediction
    predicted_detections = model.predict(X_val[sample_idx:sample_idx + 1])[0]  # Shape: (max_detections, num_classes + 3)

    # Generate CSV for this prediction
    generate_detection_csv(
        predictions=np.expand_dims(predicted_detections, axis=0),
        run_names=[val_run_names[sample_idx]],
        particle_types=particle_types,
        tomogram_shape=validator_tom.shape,
        threshold=0.5,  # Adjust threshold as needed
        output_csv='single_prediction_detections.csv'
    )
else:
    print("No validation data available for visualization.")

#%%
# Optional: Visualize Detections on a Specific Slice
# Adjust slice_index based on your tomogram depth
slice_index = 50  # Example slice

# Load the detections from the CSV
detections_df = pd.read_csv('validation_detections.csv')

# Filter detections for the current run and slice
current_run = val_run_names[0]
current_detections = detections_df[
    (detections_df['run_number'] == current_run) &
    (detections_df['z_coord'] == slice_index)
]

# Load the tomogram slice
tomogram_slice = validator_tom[slice_index, :, :]

# Initialize heatmap for visualization
heatmap_visual = np.zeros_like(tomogram_slice, dtype=np.float32)

# Overlay detections
for _, row in current_detections.iterrows():
    x, y, z = row['x_coord'], row['y_coord'], row['z_coord']
    particle = row['particle_type']
    if 0 <= x < tomogram_slice.shape[1] and 0 <= y < tomogram_slice.shape[0]:
        heatmap_visual[y, x] = 1  # Simple marker; customize as needed

# Normalize tomogram slice
tomogram_normalized = (tomogram_slice - tomogram_slice.min()) / (tomogram_slice.max() - tomogram_slice.min())
tomogram_rgb = np.stack([tomogram_normalized]*3, axis=-1)

# Generate distinct colors for each particle type
cmap = plt.cm.get_cmap('tab10', len(particle_types))
particle_colors = {particle: cmap(i)[:3] for i, particle in enumerate(particle_types)}

# Overlay detections
overlay = np.zeros_like(tomogram_rgb, dtype=np.float32)
for _, row in current_detections.iterrows():
    x, y, z = row['x_coord'], row['y_coord'], row['z_coord']
    particle = row['particle_type']
    if 0 <= x < tomogram_slice.shape[1] and 0 <= y < tomogram_slice.shape[0]:
        color = particle_colors.get(particle, [1, 0, 0])  # Default to red if unknown
        overlay[y, x] += color

# Combine tomogram with overlay
combined = tomogram_rgb + overlay
combined = np.clip(combined, 0, 1)

plt.figure(figsize=(8, 8))
plt.imshow(combined, interpolation='nearest')
plt.axis('off')
plt.title(f'Detections on Slice {slice_index}')
plt.show()
