import zipfile
from pykml import parser
from lxml import etree

# Extract KML content from KMZ file
def extract_kml(kmz_path, kml_filename):

    print("Starting KMZ extraction...")

    # Open the KMZ as a zip file
    with zipfile.ZipFile(kmz_path, 'r') as kmz:
        # file_list = kmz.namelist()
        # print("Files in archive:", file_list)

        kml_content = kmz.open(kml_filename, 'r').read()
        print("KML extracted.")

    # Parse the KML content
    root = parser.fromstring(kml_content)

    return root

def read_kml(kml_filename):
    with open(kml_filename, 'rb') as file:
        kml_content = file.read()
    
    root = parser.fromstring(kml_content)
    return root

# -- KMZ Example --
# print("=== KMZ Extraction and KML Parsing Example ===")
# kmz_path = "../data/new-brunswick.c_1000.curves.kmz"
# kml_filename = 'new-brunswick/doc.kml'

# root = extract_kml(kmz_path, kml_filename)
# # Find the first Placemark (road segment)
# try:
#     print(f"Total number of placemarks: {len(root.Document.Folder.Placemark)}")
#     first_placemark = root.Document.Folder.Placemark[0]
#     print("--- Sample Placemark XML ---")
#     print(etree.tostring(first_placemark, pretty_print=True).decode('utf-8'))
# except AttributeError:
#     print("Could not find a Placemark. Check structure with root.getchildren()")


# Access the Styles defined in the Document
# if hasattr(root.Document, 'Style'):
#     print(f"{'Style ID':<20} | {'Color (AABBGGRR)':<15}")
#     print("-" * 40)
#     for s in root.Document.Style:
#         style_id = s.get('id')
#         # Navigate to the color tag
#         color = s.LineStyle.color.text
#         print(f"{style_id:<20} | {color:<15}")

# line style
# for pm in root.Document.Folder.Placemark:
#     # print(pm)
#     style = pm.styleUrl.text # e.g., "#lineStyle0"
#     coords = pm.LineString.coordinates.text.strip()

#     if style == "#lineStyle0": # Check your KML top section for which color this is
#         radius_desc = "Straight (> 175m)"
#     elif style == "#lineStyle1":
#         radius_desc = "Broad Turns (100m - 175m)"
    
#     print(f"Segment: {radius_desc} | Start Coords: {coords.split()[0]}")

# -- KML Example --
# kml_filename = '../data/new_brunswick.curves.kml'
# root = read_kml(kml_filename)

# print("=== KML Parsing Example ===")
# try:
#     print(f"Total number of placemarks: {len(root.Document.Folder.Placemark)}")
#     first_placemark = root.Document.Folder.Placemark[0]
#     print("--- Sample Placemark XML ---")
#     print(etree.tostring(first_placemark, pretty_print=True).decode('utf-8'))
# except AttributeError:
#     print("Could not find a Placemark. Check structure with root.getchildren()")