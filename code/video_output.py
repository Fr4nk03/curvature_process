# Video output for only curvature data from map
import cv2
import pandas as pd

# ffmpeg command
# csv_directory = '../output/20F_gps_curvature.csv'
# df = pd.read_csv(csv_directory)
# print(df.columns)
# # Build the filter string
# filters = []
# for i, row in df.iterrows():
#     text = f"Curvature: {row['curvature']}".replace(':', r'\:').replace('>', r'\>')
#     filters.append(f"drawtext=text='{text}':x=20:y=20:fontsize=32:fontcolor=yellow:enable='between(t,{i},{i+1})'")


# filter_script = ",".join(filters)
# intput_directory = '../../videos/NO20260120-165747-000020F.mp4'
# output_directory = '../output/20F_curvature.mp4'
# cmd = f"ffmpeg -i {intput_directory} -vf \"{filter_script}\" -c:a copy {output_directory}"
# print(cmd)


# Load the CSV
csv_directory = '../output/20F_gps_curvature.csv'
df = pd.read_csv(csv_directory)
df.columns = df.columns.str.strip()

intput_directory = '../../videos/NO20260120-165747-000020F.mp4'
output_directory = '../output/20F_curvature.mp4'
cap = cv2.VideoCapture(intput_directory)
fps = cap.get(cv2.CAP_PROP_FPS) # Get frames per second

# Setup VideoWriter to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_directory, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate current second in the video
    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    current_sec = int(current_frame / fps)
    # print(current_sec)

    # Get data from CSV if the second exists in your data index
    if current_sec < len(df):
        curvature_text = str(df.iloc[current_sec]['curvature'])
        
        # Draw text on the frame
        cv2.putText(frame, f"Curvature: {curvature_text}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    out.write(frame) # Save frame
    
cap.release()
out.release()
print("Video processing complete.")