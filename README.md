# Description
The goal of this project is to extract the level of curvature given GPS coordinates using the tool [roadcurvature.com](https://github.com/adamfranco/curvature) and [OpenStreetMap(OSM)](https://www.openstreetmap.org/#map=5/38.01/-95.84)

## Example Steps
- Clone the curvature repository
  ```
  git clone https://github.com/adamfranco/curvature.git
  ```

- Download the osm.pbf file from the OpenStreetMap and run the command
  ```
  bin/curvature-collect -v --highway_types 'motorway,trunk,primary,secondary,tertiary,service,motorway_link,trunk_link,primary_link,secondary_link,service' ../curvature_process/data/new-brunswick-260105.osm.pbf \
  | bin/curvature-pp filter_out_ways_with_tag --tag surface --values 'unpaved,dirt,gravel,fine_gravel,sand,grass,ground,pebblestone,mud,clay,dirt/sand,soil' \
  | bin/curvature-pp filter_out_ways_with_tag --tag service --values 'driveway,parking_aisle,drive-through,parking,bus,emergency_access' \
  | bin/curvature-pp add_segments \
  | bin/curvature-pp add_segment_length_and_radius \
  | bin/curvature-pp add_segment_curvature \
  | bin/curvature-pp filter_segment_deflections \
  | bin/curvature-pp split_collections_on_straight_segments --length 10000 \
  | bin/curvature-pp roll_up_length \
  | bin/curvature-pp roll_up_curvature \
  | bin/curvature-pp filter_collections_by_curvature --min 0 \
  | bin/curvature-pp sort_collections_by_sum --key curvature --direction DESC \
  > new_brunswick.msgpack
  ```

- Run the following commands to generate KML file
  ```
  cat new_brunswick.msgpack \
  | bin/curvature-pp filter_collections_by_curvature --min 0 \
  | bin/curvature-output-kml --min_curvature 0 --max_curvature 20000 \
  > new_brunswick.kml
  ```

  ```
  cat new_brunswick.msgpack \
  | bin/curvature-pp filter_collections_by_curvature --min 0 \
  | bin/curvature-output-kml-curve-radius \
  > new_brunswick.curves.kml
  ```
- With the help of [Exiftool](https://exiftool.org/) to extract GPS coordinates from the dashcam video and the python scripts, an output CSV file will be generated with the level of curvature at each timestamp and location.
