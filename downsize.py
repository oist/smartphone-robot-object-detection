import os
import argparse
from PIL import Image

def downsample_images(input_dir, output_dir, target_size, overwrite):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all files in the input directory
    for filename in os.listdir(input_dir):
        # Construct the full file path
        file_path = os.path.join(input_dir, filename)

        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Open the image
            with Image.open(file_path) as img:
                # Resize the image
                img_resized = img.resize(target_size, Image.LANCZOS)
                
                # Determine the output path
                if overwrite:
                    output_path = file_path
                else:
                    output_path = os.path.join(output_dir, filename)

                # Save the resized image
                img_resized.save(output_path)
    
    print("Downsampling complete!")

def main():
    parser = argparse.ArgumentParser(description="Downsample images to a specified resolution.")
    parser.add_argument("input_dir", help="Directory containing the input images.")
    parser.add_argument("output_dir", help="Directory to save the downsampled images.")
    parser.add_argument("--target_size", type=int, nargs=2, default=[480, 640], metavar=('width', 'height'),
                        help="Target size for the downsampled images. Default is 480x640.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting of input images.")
    
    args = parser.parse_args()

    # If overwrite is specified, set output_dir to input_dir
    if args.overwrite:
        args.output_dir = args.input_dir

    downsample_images(args.input_dir, args.output_dir, tuple(args.target_size), args.overwrite)

if __name__ == "__main__":
    main()
