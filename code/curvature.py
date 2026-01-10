# # import zipfile
# # from pykml import parser
# from shapely.geometry import Point, LineString
# from extract import read_kml
# import pandas as pd
# from datetime import datetime

# # 1. Setup your Radius Mapping (based on our previous decode)
# RADIUS_MAP = {
#     "#lineStyle0": "> 175m (Straight/Broad)",
#     "#lineStyle1": "100m - 175m (Broad)",
#     "#lineStyle2": "60m - 100m (Moderate)",
#     "#lineStyle3": "25m - 60m (Tight)",
#     "#lineStyle4": "< 25m (Very Sharp)",
# }

# def get_curvature_at_coord(kml_root, car_lat, car_lon):
#     car_point = Point(car_lon, car_lat)
#     closest_segment = None
#     min_dist = float('inf')

#     # Find all Placemarks, even those nested in sub-folders
#     # .xpath covers the whole document for any 'Placemark' element
#     all_placemarks = kml_root.xpath('//kml:Placemark', namespaces={'kml': 'http://www.opengis.net/kml/2.2'})

#     for pm in all_placemarks:
#         if not hasattr(pm, 'LineString'):
#             continue
            
#         coords_str = str(pm.LineString.coordinates).strip()
#         # Ensure we handle the coordinates correctly
#         coords = [tuple(map(float, c.split(',')[:2])) for c in coords_str.split()]
        
#         if len(coords) < 2: continue
        
#         road_line = LineString(coords)
#         dist = road_line.distance(car_point)

#         if dist < min_dist:
#             min_dist = dist
#             closest_segment = pm


#     # 0.001 degrees is roughly 111 meters. 
#     # 0.0005 is roughly 55 meters.
#     if closest_segment is not None and min_dist < 0.001: 
#         style = str(closest_segment.styleUrl).strip()
#         radius = RADIUS_MAP.get(style, f"Unknown Style ({style})")
#         return radius, style, min_dist
#     else:
#         return "No road found nearby", None, min_dist

# def process_gps_csv(csv_filename, kml_root):
#     start_time = datetime.now()
#     gps_df = pd.read_csv(csv_filename, header=None, names=['timestamp', 'latitude', 'longitude'])

#     for index, row in gps_df.iterrows():
#         car_lat = row['latitude']
#         car_lon = row['longitude']

#         radius, style, d = get_curvature_at_coord(root, car_lat, car_lon)
#         print(f"At {row['timestamp']} and ({car_lat}, {car_lon}): Curvature is {radius}")
#     end_time = datetime.now()
#     print(f"Processed {len(gps_df)} GPS points in {end_time - start_time} seconds.")

# # --- USAGE ---
# # car_lat, car_lon = 46.4659, -67.6663
# car_lat, car_lon = 46.0900, -67.5704    # Highway 2
# # car_lat, car_lon = 46.089307, -67.570189
# # car_lat, car_lon = 46.0996, -67.5903    # Highway 2 ramp
# car_lat, car_lon = 45.9510, -66.6885


# kml_filename = '../data/new_brunswick.curves.kml'
# root = read_kml(kml_filename)
# # radius, style, d = get_curvature_at_coord(root, car_lat, car_lon)
# # print(f"At {car_lat}, {car_lon}: Curvature is {radius}")

# process_gps_csv('../../ramp_gps.csv', root)




## OPTIMIZED
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from extract import read_kml
import pandas as pd
from datetime import datetime

RADIUS_MAP = {
    "#lineStyle0": "> 175m (Straight/Broad)",
    "#lineStyle1": "100m - 175m (Broad)",
    "#lineStyle2": "60m - 100m (Moderate)",
    "#lineStyle3": "25m - 60m (Tight)",
    "#lineStyle4": "< 25m (Very Sharp)",
}

def prepare_spatial_index(kml_root):
    """
    Parses KML once and builds a spatial index for lightning-fast lookups.
    """
    print("Building spatial index...")
    all_placemarks = kml_root.xpath('//kml:Placemark', namespaces={'kml': 'http://www.opengis.net/kml/2.2'})
    
    lines = []
    metadata = []

    for pm in all_placemarks:
        if not hasattr(pm, 'LineString'):
            continue
        
        coords_str = str(pm.LineString.coordinates).strip()
        coords = [tuple(map(float, c.split(',')[:2])) for c in coords_str.split()]
        
        if len(coords) < 2: 
            continue
        
        ls = LineString(coords)
        lines.append(ls)
        # Store style and segment info separately but mapped to the same index
        metadata.append({
            'style': str(pm.styleUrl).strip(),
            'placemark': pm
        })
    
    # STRtree is the magic for performance
    tree = STRtree(lines)
    return tree, lines, metadata

def get_curvature_fast(car_point, spatial_tree, lines, metadata, last_idx=None):
    """
    Queries the spatial index.
    """
    # # Check if we are still on the same road segment (Fastest path)
    # if last_idx is not None:
    #     dist = lines[last_idx].distance(car_point)
    #     if dist < 0.0002:  # Roughly 20 meters
    #         return last_idx, dist

    # If not on the last road, query the tree for nearby segments
    # We look for anything within ~200 meters (0.002 degrees)
    buffer_zone = car_point.buffer(0.002)
    nearby_indices = spatial_tree.query(buffer_zone)
    
    if len(nearby_indices) == 0:
        return None, float('inf')

    # Only calculate exact distances for the few candidates found by the tree
    best_dist = float('inf')
    best_idx = None
    
    for idx in nearby_indices:
        dist = lines[idx].distance(car_point)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
            
    return best_idx, best_dist

def process_gps_csv(csv_filename, kml_filename):
    root = read_kml(kml_filename)
    spatial_tree, lines, metadata = prepare_spatial_index(root)
    
    gps_df = pd.read_csv(csv_filename, header=None, names=['timestamp', 'latitude', 'longitude'])
    
    start_time = datetime.now()
    results = []
    last_idx = None

    for _, row in gps_df.iterrows():
        car_point = Point(row['longitude'], row['latitude'])
        
        curr_idx, dist = get_curvature_fast(car_point, spatial_tree, lines, metadata, last_idx)
        
        if curr_idx is not None and dist < 0.001: # 111m threshold
            m_data = metadata[curr_idx]
            category = RADIUS_MAP.get(m_data['style'], "Unknown")
            last_idx = curr_idx # Update for next iteration
            results.append(category)
        else:
            last_idx = None
            results.append("No road found nearby")

    gps_df['curvature'] = results
    
    end_time = datetime.now()
    print(f"Processed {len(gps_df)} points in {end_time - start_time} seconds.")
    
    # Save the result
    gps_df.to_csv('../output/processed_gps_output.csv', index=False)
    return gps_df

# --- RUN ---
process_gps_csv('../../ramp_gps.csv', '../data/new_brunswick.curves.kml')