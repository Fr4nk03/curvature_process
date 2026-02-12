import msgpack
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

def build_spatial_index(msgpack_directory):
    """
    Builds a spatial index (STRtree) from road segments stored in a msgpack file.
    Each segment is represented as a Shapely LineString.
    seg: {
        'start': [lat, lon],
        'end': [lat, lon],
        'radius': float,
        'length': float
    }
    """
    # Load segments from msgpack
    segments_geoms = []
    # segment_data_map = {}
    segment_data_list = []

    # Read the msgpack file and extract segments
    with open(msgpack_directory, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False, strict_map_key=False)
        for collection in unpacker:
            collection_name = collection.get('join_data', 'Unnamed')
            for way in collection.get('ways', []):
                for seg in way.get('segments', []):
                    # Create a Shapely LineString for the segment
                    line = LineString([(seg['start'][1], seg['start'][0]), 
                                    (seg['end'][1], seg['end'][0])])
                    
                    segments_geoms.append(line)
                    # Map the geometry ID to the actual segment data (radius ...)
                    segment_data_list.append({
                        **seg, 
                        'collection': collection_name  # Include collection name for convenience
                    })

    # STRtree
    tree = STRtree(segments_geoms)
    return tree, segments_geoms, segment_data_list

def get_curvature_from_map(car_point, spatial_tree, segment_data_list):
    """
    Given a car's position (as a Point), find the nearest road segment and return its curvature data.
    """
    # Query the spatial index for the nearest segment
    nearest_segment = spatial_tree.nearest(car_point)
    
    if nearest_segment:
        # print(f"Nearest segment id: {nearest_segment}, using {nearest_segment + 1}")
        seg_data = segment_data_list[nearest_segment + 1] if nearest_segment + 1 < len(segment_data_list) else segment_data_list[nearest_segment]
        return seg_data  # Returns the segment data including radius, length, etc.
    else:
        print("!!! No nearest segment is found...")
        return None