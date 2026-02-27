# Vision-based lane curvature estimation with BEV transformation and sliding window method
# GPS-based curvature data integration
import time
import cv2
import numpy as np
import pandas as pd
from curvature_gps import build_spatial_index, get_curvature_from_map
from shapely.geometry import Point
from tool.SMA import SMA

# --- Helper method --- #
def get_curvature(lane_points, ym_per_pixel, xm_per_pixel, upper=False):
    if lane_points is None or len(lane_points) < 3:
        print("Insufficient lane points for curvature calculation.")
        return None

    # Extract x and y coordinates
    x = np.array([p[0] for p in lane_points])
    y = np.array([p[1] for p in lane_points])

    # Fit a second-order polynomial (x = Ay^2 + By + C)
    fit = np.polyfit(y * ym_per_pixel, x * xm_per_pixel, 2)

    # Curvature formula: R = (1 + (2Ay + B)^2)^(3/2) / |2A|
    A, B, _ = fit
    if not upper:
        y_eval = np.max(y) * ym_per_pixel  # Evaluate at the bottom of the points
    else:
        y_eval = np.min(y) * ym_per_pixel  # Evaluate at the top of the points
    curvature = ((1 + (2 * A * y_eval + B) ** 2) ** 1.5) / np.abs(2 * A)
    return curvature

def get_gps(gps_df, original_start_time, current_frame, input_fps):
    # Calculate the timestamp for the current frame
    current_sec = int(current_frame / input_fps) + original_start_time - 1
    # print(f"Current second in original video: {current_sec + 1}")
    if current_sec < len(gps_df):
        latitude, longitude = float(gps_df.iloc[current_sec]['latitude']), float(gps_df.iloc[current_sec]['longitude'])
        # print(f"Longitude: {longitude}, Latitude: {latitude}")
        return latitude, longitude
    else:
        print("No GPS data available for this frame.")
        return None, None

# SMA filter for curvature smoothing
upper_sma_filter = SMA(window_size=5)
lower_sma_filter = SMA(window_size=5)

# --- Build spatial index for GPS-based curvature data --- #
msgpack_directory = '../data/new_brunswick_raw_radius.msgpack'
current_time = time.time()
spatial_tree, segments_geoms, segment_data_list = build_spatial_index(msgpack_directory)
time_taken = time.time() - current_time
print(f"STRtree built with {len(segments_geoms)} road segments in {time_taken:.2f} seconds.")

# STR query time
str_query_time = []

# video_directory = '../../videos/NO20251004-095008-000013F.mp4'
video_directory = '../../videos/33F_curve_1_4K.mp4'
original_start_time = 76

# video_directory = '../../videos/33F_curve_2_4K.mp4'
# original_start_time = 155
section = video_directory.split('/')[-1].split('.')[0]
# video_directory = '../../videos/23F_straight_1_4K.mp4'

# Load GPS dataframe
csv_directory = video_directory.split('/')[-1].split('_')[0] + '_gps.csv'
gps_csv_directory = '../data/csv/' + csv_directory
gps_df = pd.read_csv(gps_csv_directory, header=None, names=['timestamp', 'latitude', 'longitude'])

# --- Video processing --- #
vidcap = cv2.VideoCapture(video_directory)
success, image = vidcap.read()

# bev_w, bev_h = 1640, 590
bev_w, bev_h = 400, 800
input_fps = vidcap.get(cv2.CAP_PROP_FPS)

# Input resolution
input_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
input_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output resolution
output_width = int(input_width * (bev_h/input_height)) + bev_w
output_height = bev_h

print(f"Input Video: {video_directory}, FPS: {input_fps}, Resolution: {input_width}x{input_height}")
print(f"Output Video Resolution: {output_width}x{output_height}")

# Based on GPS Coordinate: (45.9604, -67.4442)
xm_per_pixel = 3.7 / 260
ym_per_pixel = 12.2 / 80           # dash + gap = 12.2m

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'../output/trapezoid_bev_output_new_{section}.mp4', fourcc, input_fps, (output_width, output_height))
sliding_window_out = cv2.VideoWriter(f'../output/debug/sliding_window_output_{section}.mp4', fourcc, input_fps, (bev_w, bev_h))

# Trackbar for adjusting HSV thresholds
def nothing(x):
    pass
cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 220, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)


# BEV Transformation Matrix
# Choose 4 trapezoid points (region of interest)
    ## IL
    # tl = [660, 450]  # top-left
    # tr = [925, 450]  # top-right
    # br = [1540, 590]  # bottom-right
    # bl = [100, 590]  # bottom-left

    ## old
    # tl = [700, 340]  # top-left
    # tr = [940, 340]  # top-right
    # bl = [440, 420]  # bottom-left
    # br = [1200, 420]  # bottom-right

    # For 1640 * 590 video
    # tl = [787, 300]  # top-left
    # tr = [853, 300]  # top-right
    # bl = [405, 420]  # bottom-left
    # br = [1235, 420]  # bottom-right

# For 3840 * 2160 video
tl = [1860, 1100]  # top-left
tr = [1980, 1100]  # top-right
bl = [1120, 1520]  # bottom-left
br = [2720, 1520]  # bottom-right

# Source and destination points for perspective transformation
src = np.float32([tl, tr, br, bl])
dst = np.float32([[0, 0], [bev_w, 0], [bev_w, bev_h], [0, bev_h]])

# Transformation matrix and warp the frame
BEV_M = cv2.getPerspectiveTransform(src, dst)

prevL = []
prevR = []

frame_cnt = 0
while success:
    success, image = vidcap.read()
    if not success:
        # break
        continue
    # frame = cv2.resize(image, (1640, 590))
    frame = image

    # cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    # cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    # cv2.circle(frame, br, 5, (0, 0, 255), -1)
    # cv2.circle(frame, bl, 5, (0, 0, 255), -1)

    output_frame = cv2.resize(image, (int(input_width * (bev_h/input_height)), bev_h))


    bev_frame = cv2.warpPerspective(frame, BEV_M, (bev_w, bev_h))
    cv2.imshow("BEV Frame", bev_frame)
    # bev_resized = cv2.resize(bev_frame, (int(bev_w * (input_height/bev_h)), input_height))
    # cv2.imshow("BEV Resized", bev_resized)
    # print("BEV resized shape:", bev_resized.shape)

    # HSV and LAB Frame Thresholding
    hsv_frame = cv2.cvtColor(bev_frame, cv2.COLOR_BGR2HSV)
    lab_frame = cv2.cvtColor(bev_frame, cv2.COLOR_BGR2LAB)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])

    # Lane mask
    mask = cv2.inRange(hsv_frame, lower, upper)
    yellow_mask = cv2.inRange(lab_frame[:,:,2], 155, 200)

    combined_binary = cv2.bitwise_or(mask, yellow_mask)
    combined_binary_bgr = cv2.cvtColor(combined_binary, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Lane Mask", combined_binary_bgr)
    if cv2.waitKey(1) == ord('q'):
        break

    # # Sobel Filtering
    # grad_x = cv2.Sobel(combined_binary_bgr, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)

    # # Normalize
    # abs_grad_x = np.absolute(grad_x)
    # normalized_grad = np.uint8(255 * abs_grad_x / np.max(abs_grad_x))

    # Histogram
    histogram = np.sum(combined_binary[combined_binary.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding Window
    y = 800
    left = []
    right = []

    msk = combined_binary.copy()
    window_h = 10
    window_half_w = 20

    while y > 0:
        ## Left threshold
        img = combined_binary[y-window_h:y, left_base-window_half_w:left_base+window_half_w]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                left.append((left_base-window_half_w + cx, y-window_h + cy))
                left_base = left_base-window_half_w + cx
                cv2.rectangle(msk, (left_base-window_half_w,y), (left_base+window_half_w,y-window_h), (255,255,255), 2)
        
        ## Right threshold
        img = combined_binary[y-window_h:y, right_base-window_half_w:right_base+window_half_w]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                right.append((right_base-window_half_w + cx, y-window_h + cy))
                right_base = right_base-window_half_w + cx
                cv2.rectangle(msk, (right_base-window_half_w,y), (right_base+window_half_w,y-window_h), (255,255,255), 2)
        
        # cv2.rectangle(msk, (left_base-50,y), (left_base+50,y-window_h), (255,255,255), 2)
        # cv2.rectangle(msk, (right_base-50,y), (right_base+50,y-window_h), (255,255,255), 2)
        y -= window_h
        
    # Ensure lx and rx are not empty
    if len(left) == 0:
        left = prevL
    else:
        prevL = left
    if len(right) == 0:
        right = prevR
    else:
        prevR = right

    # Lower Half
    left_lower = [p for p in left if p[1] >= bev_h//2]
    right_lower = [p for p in right if p[1] >= bev_h//2]

    # Upper Half
    left_upper = [p for p in left if p[1] <= bev_h//2]
    right_upper = [p for p in right if p[1] <= bev_h//2]

    if (frame_cnt == 100):
        print("Left lane points:", left_lower)
        print("Right lane points:", right_lower)

    # ONLY fit if we have at least 3 points
    if len(left_lower) >= 3 and len(right_lower) >= 3:
        # Calculate curvature directly
        left_curvature = get_curvature(left_lower, ym_per_pixel, xm_per_pixel, upper=False)
        right_curvature = get_curvature(right_lower, ym_per_pixel, xm_per_pixel)
        curvature = (left_curvature + right_curvature) / 2
        smoothed_curvature = lower_sma_filter.update(curvature)
        if smoothed_curvature >= 10000:
            cv2.putText(output_frame, f"Current Curvature: {int(smoothed_curvature)} m > 10000m (Straight/Broad)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (165, 86, 5), 2)
        else:
            cv2.putText(output_frame, f"Current Curvature: {int(smoothed_curvature)} m", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (5, 95, 255), 2)
    else:
        # Use GPS Data
        current_time = time.time()
        latitude, longitude = get_gps(gps_df, original_start_time, frame_cnt, input_fps)
        seg_data = get_curvature_from_map(Point(longitude, latitude), spatial_tree, segment_data_list)
        smoothed_seg_data = lower_sma_filter.update(seg_data['radius']) if seg_data else None
        str_query_time.append(time.time() - current_time)
        # print(seg_data)
        cv2.putText(output_frame, f"Lane cannot be detected! Using GPS Data: {seg_data['radius']} m", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Check upper half for curvature forecast
    if len(left_upper) >= 3 and len(right_upper) >= 3:
        left_upper_curvature = get_curvature(left_upper, ym_per_pixel, xm_per_pixel, upper=True)
        right_upper_curvature = get_curvature(right_upper, ym_per_pixel, xm_per_pixel, upper=True)
        upper_curvature = (left_upper_curvature + right_upper_curvature) / 2
        smoothed_upper_curvature = upper_sma_filter.update(upper_curvature)
        if smoothed_upper_curvature >= 10000:
            cv2.putText(output_frame, f"Curvature up front: {int(smoothed_upper_curvature)} m > 10000m (Straight/Broad)", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (165, 86, 5), 2)
        else:
            cv2.putText(output_frame, f"Curvature up front: {int(smoothed_upper_curvature)} m", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (5, 95, 255), 2)
    else:
        current_time = time.time()
        latitude, longitude = get_gps(gps_df, original_start_time, frame_cnt, input_fps)
        upper_seg_data = get_curvature_from_map(Point(longitude, latitude), spatial_tree, segment_data_list)
        smoothed_upper_seg_data = upper_sma_filter.update(upper_seg_data['radius']) if upper_seg_data else None
        str_query_time.append(time.time() - current_time)
        # print(seg_data)
        cv2.putText(output_frame, f"Lane up front cannot be detected! Using GPS Data: {upper_seg_data['radius']} m", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Calculate the curvature
    # y_eval = 800
    # left_curvature = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.abs(2*left_fit[0])
    # right_curvature = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.abs(2*right_fit[0])
    # curvature = (left_curvature + right_curvature) / 2

    
    cv2.imshow("Lane Detection - Sliding Windows", msk)
    msk_bgr = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
    sliding_window_out.write(msk_bgr)

    combined = np.hstack((output_frame, combined_binary_bgr))
    out.write(combined)

    # if frame_cnt == 0:
    #     cv2.imwrite("../../output_image/bev_poor_road_condition.jpg", combined)

    frame_cnt += 1
    if frame_cnt == 100:
        cv2.imwrite(f"../output/debug/trapezoid_frame_{section}_{frame_cnt}.jpg", output_frame)
        cv2.imwrite(f"../output/debug/bev_frame_{section}_{frame_cnt}.jpg", combined_binary_bgr)
        cv2.imwrite(f"../output/debug/bev_combined_frame_{section}_{frame_cnt}.jpg", combined)
        break

vidcap.release()
out.release()
sliding_window_out.release()
print(f"Output saved as trapezoid_bev_output_{section}.mp4")
print(f"STR average query time: {np.mean(str_query_time)} s")