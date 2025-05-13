import xml.etree.ElementTree as ET
import os
from pathlib import Path

def parse_voc_annotation(xml_file_path: Path):
    """
    Parses a PASCAL VOC XML file.
    Returns a dictionary with 'filename' (str) and 'objects' (list of dicts).
    Each object dict has 'name' (str) and 'bbox' (list of 4 ints: [xmin, ymin, xmax, ymax]).
    Returns None if parsing fails or essential data is missing.
    """
    try:
        tree = ET.parse(str(xml_file_path)) # ET.parse expects a string path or file object
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"XML ParseError for {xml_file_path}: {e}")
        return None
    except FileNotFoundError:
        print(f"Annotation file not found: {xml_file_path}")
        return None

    data = {'objects': []}

    filename_node = root.find('filename')
    if filename_node is not None and filename_node.text:
        data['filename'] = filename_node.text
    else:
        # If filename tag is missing, infer from the XML file's name
        data['filename'] = xml_file_path.stem + ".jpg" # Assume .jpg, can be refined
        print(f"Warning: 'filename' tag missing or empty in {xml_file_path}. Inferred as {data['filename']}")

    for obj_node in root.findall('object'):
        try:
            name_node = obj_node.find('name')
            bndbox_node = obj_node.find('bndbox')

            if name_node is None or not name_node.text:
                print(f"Warning: Object in {xml_file_path} missing name. Skipping object.")
                continue
            if bndbox_node is None:
                print(f"Warning: Object '{name_node.text if name_node is not None else 'Unknown'}' in {xml_file_path} missing bndbox. Skipping object.")
                continue

            name = name_node.text
            
            xmin_node = bndbox_node.find('xmin')
            ymin_node = bndbox_node.find('ymin')
            xmax_node = bndbox_node.find('xmax')
            ymax_node = bndbox_node.find('ymax')

            if None in [xmin_node, ymin_node, xmax_node, ymax_node] or \
               any(c is None or not c.text for c in [xmin_node, ymin_node, xmax_node, ymax_node]):
                print(f"Warning: Object '{name}' in {xml_file_path} missing one or more bbox coordinates or text. Skipping object.")
                continue

            # Convert to float first for robustness, then to int
            xmin = int(float(xmin_node.text))
            ymin = int(float(ymin_node.text))
            xmax = int(float(xmax_node.text))
            ymax = int(float(ymax_node.text))
            
            data['objects'].append({'name': name, 'bbox': [xmin, ymin, xmax, ymax]})
        except ValueError as e:
            print(f"Warning: Error parsing bbox coordinates for an object in {xml_file_path}: {e}. Skipping object.")
            continue
        except AttributeError as e: 
            print(f"Warning: Missing text in a sub-element of an object in {xml_file_path}: {e}. Skipping object.")
            continue
            
    return data



def get_image_details_for_class(selected_class:list, annotations_folder_path: Path, inclusivo:bool):
    """
    Finds all images and their specific bounding boxes for a given class.
    Args:
        selected_class (str): The class name to search for.
        annotations_folder_path (Path): Path object for the annotations folder.
    Returns:
        list: A list of dictionaries. Each dict contains:
              'image_filename' (str): Filename of the image.
              'bboxes' (list of lists): List of [xmin, ymin, xmax, ymax] for the selected_class in this image.
              'xml_path' (Path): Path to the annotation file.
              Returns an empty list if no images are found for the class or if folder doesn't exist.
    """
    images_for_class = []
    if not annotations_folder_path.is_dir():
        print(f"Error: Annotations folder not found or is not a directory: {annotations_folder_path}")
        return images_for_class

    for xml_file_path in annotations_folder_path.glob('*.xml'):
        annotation_data = parse_voc_annotation(xml_file_path)
        if annotation_data and annotation_data.get('filename'):
            specific_bboxes_for_class = []
            if inclusivo==False:
                for clase in selected_class:
                    for obj in annotation_data.get('objects', []):
                        if obj.get('name') == clase:
                            specific_bboxes_for_class.append(obj['bbox'])
            else:
                objets =[]
                for obj in annotation_data.get('objects', []):
                    objets.append([obj['name'], obj['bbox']])
                if set([x[0] for x in objets])==set(selected_class):
                   for obj in objets:
                       specific_bboxes_for_class.append(obj[1])

                    

            
            if specific_bboxes_for_class: # Only add if the selected class was found in this image
                images_for_class.append({
                    'image_filename': annotation_data['filename'],
                    'bboxes': specific_bboxes_for_class,
                    'xml_path': xml_file_path
                })
    return images_for_class