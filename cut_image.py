import os
import random
from PIL import Image
import argparse



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file_name', type=str, default='ex.jpg', help='original image')
    parser.add_argument('--column_num', type=int, default=2, help='column_num')
    parser.add_argument('--row_num', type=int, default=2, help='row_num')
    parser.add_argument('--prefix_output_filename', type=str, default='sub_image', help='filename')
    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    # Open the original image
    original = opt.image_file_name
    img = Image.open(original)

    # Get the size of the original image
    width, height = img.size

    # Define the size of sub-image
    sub_width = width // opt.column_num
    sub_height = height // opt.row_num

    # Create the subdirectory if it doesn't exist
    if not os.path.exists("sub"):
        os.makedirs("sub")

    # Split the original image into sub-images
    sub_images = []
    for i in range(opt.column_num):
        for j in range(opt.row_num):
            # Calculate the bounding box for this sub-image
            left = i * sub_width
            upper = j * sub_height
            right = left + sub_width
            lower = upper + sub_height
            bbox = (left, upper, right, lower)
            
            # Crop the sub-image from the original image
            sub_img = img.crop(bbox)

            transform = []
            if random.random() < 0.5:
                sub_img = sub_img.rotate(90,expand = 1)
                transform.append("rotate_left")

            if random.random() < 0.5:
                sub_img = sub_img.transpose(Image.FLIP_LEFT_RIGHT)
                transform.append("horizontal_flip")

            if random.random() < 0.5:
                sub_img = sub_img.transpose(Image.FLIP_TOP_BOTTOM)
                transform.append("vertical_flip")
                sub_images.append(sub_img)

            
            # Save the sub-image to a file in the subdirectory
            filename = f"sub/{opt.prefix_output_filename}_{random.randint(0, 99999)}.jpg"
            sub_img.save(filename)


    print("Sub-images saved successfully!")

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)