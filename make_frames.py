import os
import cv2

def cv2_dump_frames(fn, output_path,input_path,vid, fmt="jpg", quality=90):

    cap = cv2.VideoCapture(fn)

    index = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        index += 1

        cv2_ext = fmt
        
        if 'Coffee_room' in input_path:
            fn = os.path.join(output_path, 'Coffee_room_'+vid.split('.')[0]+'_%d.%s' % (index, cv2_ext))
            cv2.imwrite(fn, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif 'Home' in input_path:
            fn = os.path.join(output_path, 'Home_'+vid.split('.')[0]+'_%d.%s' % (index, cv2_ext))
            cv2.imwrite(fn, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    

    return index


