import os

res = "results/Le2iFall/AlphaPose/CSV/Coffee_room"
vid = "data/Coffee_room/Videos"
done = os.path.join(vid, "done")
os.mkdir(done)
res_files = os.listdir(res)
vid_files = os.listdir(vid)

for v in vid_files:
    json_file =   "Coffee_room_mslstm_" + v.split(".")[0].replace(" ", "") + ".json"
    if json_file in res_files:
        print("File available: ", json_file)
        os.rename(os.path.join(vid, v), os.path.join(done, v))