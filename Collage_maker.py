from PIL import Image
import os
import matplotlib.pyplot as plt


def extract_tkb_tuab(filename):
    """
    Extract Tkb and Tuab values from the filename.
    Example filename: "50x50_3Tkb_10T_plot.jpg"
    """
    parts = filename.split('_')
    tkb = int(parts[1].replace('Tkb', '').replace('T', '').strip())  # Clean and convert Tkb
    tuab = int(parts[2].replace('ua', '').replace('Tu', '').replace('B', '').replace('a', '').replace('T', '').strip())  # Clean and convert Tuab
    return tkb, tuab


def create_collage_and_label(image_folder, output_filename, grid_size, crop_size, resize_factor=0.5):
    # Get list of image files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # Sort images based on Tkb and Tuab values
    images.sort(key=lambda img: extract_tkb_tuab(img))

    # Open images, crop them, and resize to reduce memory usage
    opened_images = []
    file_grid = []

    for img in images:
        image = Image.open(os.path.join(image_folder, img))
        # Crop the image
        width, height = image.size
        left = (width - crop_size[0]) / 2
        top = (height - crop_size[1]) / 2
        right = (width + crop_size[0]) / 2
        bottom = (height + crop_size[1]) / 2
        cropped_image = image.crop((left, top, right, bottom))
        # Resize the image to reduce memory usage
        resized_image = cropped_image.resize((int(cropped_image.width * resize_factor),
                                              int(cropped_image.height * resize_factor)))
        opened_images.append(resized_image)

    # Create a 2D list (grid) of filenames
    for i in range(grid_size[0]):
        file_grid.append(images[i * grid_size[1]:(i + 1) * grid_size[1]])

    # Calculate the size of the collage
    img_width, img_height = opened_images[0].size
    collage_width = img_width * grid_size[1]
    collage_height = img_height * grid_size[0]

    # Create a new blank image for the collage
    collage = Image.new('RGB', (collage_width, collage_height))

    # Paste images into the collage
    for index, img in enumerate(opened_images):
        row = index // grid_size[1]
        col = index % grid_size[1]
        collage.paste(img, (col * img_width, row * img_height))

    # Create a labeled plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(collage)
    ax.axis('off')  # Hide axes

    # Add labels for each column (T_B values) and row (T_KB values)
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            filename = file_grid[row][col]
            tkb, tuab = extract_tkb_tuab(filename)

            # Add column labels (T_B values) for the top of the grid
            if row == 0:  # Only add column labels once at the top
                ax.text(
                    col * img_width + img_width / 2,
                    -15,  # Above the images
                    f"${tuab}T_{{b}}$",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="black",
                    fontweight="bold",
                )

            # Add row labels (T_KB values) for the left of the grid
            if col == 0:  # Only add row labels once at the left
                ax.text(
                    -50,  # To the left of the images
                    row * img_height + img_height / 2,
                    f"${tkb}TK_{{b}}$",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="black",
                    fontweight="bold",
                    rotation=0,
                )

    # Save the plot with labeled axes
    plt.savefig(output_filename.replace('.jpg', '_labeled.jpg'), bbox_inches='tight', dpi=300)
    plt.close()


# Example usage
image_folder = r"C:\Users\omare\PycharmProjects\Crytal Detector BA Trial\2D Crystals\output PNG Folders\ɸ = 0.5024 (2.21rp) PNG output"
output_filename = '2.21rp_collage_output.jpg'
grid_size = (6, 5)  # Adjust based on your desired grid structure
crop_size = (300, 300)  # Adjust based on your desired crop size
resize_factor = 1  # Resize images to 100% of their original size

create_collage_and_label(image_folder, output_filename, grid_size, crop_size, resize_factor)
