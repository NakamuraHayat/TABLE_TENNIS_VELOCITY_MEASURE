import cv2
import numpy as np
import sys
import csv



def readVideo(input_file):
    video = cv2.VideoCapture(input_file)
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    return video


def extractColor(frame, hsv_lower, hsv_upper):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_extract_color = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)
    return frame_extract_color


def backgroundSubtraction(frame_begin_extract_color, frame_middle_extract_color, frame_last_extract_color):
    frame_diff_begin_middle = cv2.absdiff(frame_begin_extract_color, frame_middle_extract_color)
    frame_diff_middle_last = cv2.absdiff(frame_middle_extract_color, frame_last_extract_color)
    frame_diff = cv2.bitwise_and(frame_diff_begin_middle, frame_diff_middle_last)
    return frame_diff


def calculateContours(frame_diff_noise_removal, calculate_contours_flag):
    contours, hierarchy = cv2.findContours(frame_diff_noise_removal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pixel_mm = 0
    center_x = 0
    center_y = 0
    if (len(contours) == 1):
        np_contours = np.array(contours)
        contours_y = np_contours[:, :, :, 1]
        contours_y_max = np.max(contours_y)
        contours_y_min = np.min(contours_y)
        contours_y_pixel = contours_y_max - contours_y_min
        if contours_y_pixel != 0:
            pixel_mm = 40 / contours_y_pixel # 1mm = pixel_mm[pixcl]
        moment = cv2.moments(frame_diff_noise_removal)
        center_x = moment["m10"] / moment["m00"]
        center_y = moment["m01"] / moment["m00"]
        if calculate_contours_flag == 0:
            calculate_contours_flag = 1
            pixel_mm = 0
    else:
        calculate_contours_flag = 0
    return pixel_mm, center_x, center_y, calculate_contours_flag


def calculateVilocity(center_x_previous, center_y_previous, center_x, center_y, pixel_mm, fps):
    distance_x = center_x - center_x_previous
    distance_y = center_y - center_y_previous
    distance_pixel = (distance_x ** 2 + distance_y ** 2) ** 0.5
    if distance_pixel < 0:
        distance_pixel = distance_pixel * -1
    distance_mm = distance_pixel * pixel_mm
    vilocity_m_s = distance_mm / 1000 * fps
    return vilocity_m_s


def saveVideo(frame_middle, frame_diff_noise_removal, fps, output_file, vilocity_m_s):
    frame_middle_2 = cv2.resize(frame_middle, dsize = None, fx = 0.5, fy = 0.5)
    frame_diff_noise_removal_2 = cv2.resize(frame_diff_noise_removal, dsize = None, fx = 0.5, fy = 0.5)
    frame_diff_noise_removal_2_3 = cv2.merge((frame_diff_noise_removal_2, frame_diff_noise_removal_2, frame_diff_noise_removal_2))
    frame_connect = cv2.hconcat([frame_middle_2, frame_diff_noise_removal_2_3])
    cv2.putText(frame_connect, text = str(vilocity_m_s), org = (100, 300), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1.0, color = (0, 255, 0), thickness = 2, lineType = cv2.LINE_4)
    save_video.write(frame_connect)
    return save_video




#設定値-----------------------------------------
input_file = "MVI_2209.MOV"
output_file = "output.mp4"
hsv_lower = np.array([0, 132, 61])
hsv_upper = np.array([21, 255, 255])
median_blue_value = 25
#-----------------------------------------------
#初期値-----------------------------------------
center_x_previous = 0
center_y_previous = 0
calculate_contours_flag = 1
vilocity_m_s = 0
vilocity_m_s_list = []
#-----------------------------------------------

video = readVideo(input_file)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = video.get(cv2.CAP_PROP_FPS)
width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2
save_video = cv2.VideoWriter(output_file, fourcc, fps, (int(width), int(height)))

frame_ok, frame_begin = video.read()
if not frame_ok:
    print("Could not read video")
frame_ok, frame_middle = video.read()

while True:
    frame_ok, frame_last = video.read()
    if not frame_ok:
        break

    frame_begin_extract_color = extractColor(frame_begin, hsv_lower, hsv_upper)
    frame_middle_extract_color = extractColor(frame_middle, hsv_lower, hsv_upper)
    frame_last_extract_color = extractColor(frame_last, hsv_lower, hsv_upper)
    frame_diff = backgroundSubtraction(frame_begin_extract_color, frame_middle_extract_color, frame_last_extract_color)
    frame_diff_noise_removal = cv2.medianBlur(frame_diff, median_blue_value)

    pixel_mm, center_x, center_y, calculate_contours_flag = calculateContours(frame_diff_noise_removal, calculate_contours_flag)
    if center_x != 0:
        vilocity_m_s = calculateVilocity(center_x_previous, center_y_previous, center_x, center_y, pixel_mm, fps)

    save_video = saveVideo(frame_middle, frame_diff_noise_removal, fps, output_file, vilocity_m_s)
    if vilocity_m_s != 0 and center_x != 0 :
        print(vilocity_m_s)
        vilocity_m_s_list.append(vilocity_m_s)
        
    frame_begin = frame_middle
    frame_middle = frame_last
    center_x_previous = center_x
    center_y_previous = center_y
    
    
with open("output_vilocity.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(vilocity_m_s_list)
    

    
video.release()
save_video.release()
cv2.destroyAllWindows