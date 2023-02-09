import os

import make_frames

# prefix = '/mnt/c/Users/danish/Downloads/Compressed/'
prefix = ''

input_dirs=[f'{prefix}data/Coffee_room/Videos',
            f'{prefix}data/Home/Videos']
           
output_dir = f'{prefix}data/Frames'


if not os.path.exists(output_dir):
    os.mkdir(output_dir)
total_frames = 0
for input_dir in input_dirs:
    for vid in sorted(os.listdir(input_dir)):
        vpath=os.path.join(input_dir,vid)
        
        total_frames += make_frames.cv2_dump_frames(vpath, output_dir,input_dir,vid, 'png', 94)

print("Total frames decoded: %d" % total_frames)



