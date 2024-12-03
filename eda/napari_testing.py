import copick
import zarr
import napari

# Load the configuration
root = copick.from_file('../copick_config.json')

run = root.get_run('TS_5_4')

voxel_spacing = run.get_voxel_spacing(10.000)

# Access the specific tomogram
tomogram = voxel_spacing.get_tomogram("denoised")

# Access the Zarr data
zarr_store = tomogram.zarr()
zarr_group = zarr.open(zarr_store)

array_0 = zarr_group['0']

# (Optional) Load true labels from another dataset in the Zarr group
# Replace 'true_labels' with the actual dataset name for your labels
# Load the true labels for the overlay (assume they are stored as 'overlay')
if 'overlay' in zarr_group:
    overlay = zarr_group['overlay']
else:
    raise ValueError("Overlay dataset not found in the Zarr group.")


# viewer = napari.Viewer()

# viewer.add_image(array_0,name="Tomogram",scale = [10.012,10.012,10.012])

# Start Napari
# napari.run()