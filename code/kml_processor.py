## OPTIMIZED for kml file
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
    output_csv_directory = '../output/' + csv_directory.split('/')[-1].replace('.csv', '_curvature.csv')
    gps_df.to_csv(output_csv_directory, index=False)
    return gps_df

# --- RUN ---
# csv_directory = '../data/csv/ramp_gps.csv'
csv_directory = '../data/csv/20F_gps.csv'
kml_directory = '../data/new_brunswick_raw.curves.kml'
process_gps_csv(csv_directory, kml_directory)