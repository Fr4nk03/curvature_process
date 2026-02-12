import msgpack
import sys

# Re-configure stdout to handle UTF-8 regardless of terminal settings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

file_path = '../data/new_brunswick_raw_radius.msgpack'

with open(file_path, 'rb') as f:
    unpacker = msgpack.Unpacker(f, raw=False, strict_map_key=False)
    
    for collection in unpacker:
        name = collection.get('join_data', 'Unnamed')
        print(f"Collection: {name}", flush=True)
        
        for way in collection['ways']:
            for segment in way['segments']:
                radius = segment.get('radius')
                length = segment.get('length')
                start = segment.get('start')
                end = segment.get('end')
                print(f" Segment from {start} to {end}:", flush=True)
                
                # Check for highway curves
                if radius < 2000:
                    print(f"  --- Sharp Curve Detected! Radius: {radius:.2f}m, Length: {length:.2f}m")
                if 2000 <= radius <= 5000:
                    print(f"  ! Highway Curve Found! Radius: {radius:.2f}m, Length: {length:.2f}m")
                elif radius >= 5000:
                    print(f"  | Straight/Broad Segment with radius {radius:.2f}m (Radius capped at 10km)")