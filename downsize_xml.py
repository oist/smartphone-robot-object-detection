import os
import xml.etree.ElementTree as ET

# Define the directory containing the XML files
directory = '.'

# Define the scaling factors
width_scale = 480 / 1944
height_scale = 640 / 2592

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.xml'):
        file_path = os.path.join(directory, filename)
        
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Update width and height
        size = root.find('size')
        if size is not None:
            width = size.find('width')
            height = size.find('height')
            if width is not None and height is not None:
                width.text = str(480)
                height.text = str(640)

        # Scale bounding boxes
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = bndbox.find('xmin')
                ymin = bndbox.find('ymin')
                xmax = bndbox.find('xmax')
                ymax = bndbox.find('ymax')
                if xmin is not None and ymin is not None and xmax is not None and ymax is not None:
                    xmin.text = str(round(int(xmin.text) * width_scale))
                    ymin.text = str(round(int(ymin.text) * height_scale))
                    xmax.text = str(round(int(xmax.text) * width_scale))
                    ymax.text = str(round(int(ymax.text) * height_scale))

        # Write the modified XML back to the file
        tree.write(file_path)

print("Processing completed.")
