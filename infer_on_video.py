from model import BallTrackerNet
import torch
import cv2
from general import postprocess
from tqdm import tqdm
import numpy as np
import argparse
from itertools import groupby
from scipy.spatial import distance
import time

"""
python infer_on_video.py --model_path /app/models/model.pth --video_path /app/videos/example/examplexxx.mp4 --video_out_path /app/videos/output/examplexxx.avi --extrapolation
-- model_path 指定要用的模型权重
-- video_path 指定要推理的视频
-- video_out_path 指定输出视频
"""

def read_video(path_video):
    """ Read video file    
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
    start_read_time = time.time()
    
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    # 模仿 tqdm 进度条，显示视频读取进度
    pbar = tqdm(total=total_frames, desc="Reading video")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            pbar.update(1)
        else:
            break
    pbar.close()
    cap.release()
    
    end_read_time = time.time()
    print(f"视频读取总耗时: {end_read_time - start_read_time:.2f} 秒")
    
    return frames, fps

def infer_model(frames, model):
    """ Run pretrained model on a consecutive list of frames    
    :params
        frames: list of consecutive video frames
        model: pretrained model
    :return    
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
    """
    start_infer_time = time.time()
    height = 360
    width = 640
    dists = [-1]*2
    ball_track = [(None,None)]*2
    
    # tqdm 已经存在，保留
    for num in tqdm(range(2, len(frames)), desc="Model inference"):
        img = cv2.resize(frames[num], (width, height))
        img_prev = cv2.resize(frames[num-1], (width, height))
        img_preprev = cv2.resize(frames[num-2], (width, height))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32)/255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output)
        ball_track.append((x_pred, y_pred))

        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist) 
    
    end_infer_time = time.time()
    print(f"模型推理总耗时: {end_infer_time - start_infer_time:.2f} 秒")
    return ball_track, dists 

def remove_outliers(ball_track, dists, max_dist = 100):
    start_time = time.time()
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if (dists[i+1] > max_dist) | (dists[i+1] == -1):      
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i-1] == -1:
            ball_track[i-1] = (None, None)
    end_time = time.time()
    print(f"移除异常点总耗时: {end_time - start_time:.2f} 秒")
    return ball_track  

def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    start_time = time.time()
    list_det = [0 if x[0] else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []
    for i, (k, l) in enumerate(groups):
        if (k == 1) & (i > 0) & (i < len(groups) - 1):
            dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor+l])
            if (l >=max_gap) | (dist/l > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + l - 1         
        cursor += l
    if len(list_det) - min_value > min_track: 
        result.append([min_value, len(list_det)]) 
    end_time = time.time()
    print(f"轨迹分割总耗时: {end_time - start_time:.2f} 秒")
    return result     

def interpolation(coords):
    start_time = time.time()
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
    y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

    nons, yy = nan_helper(x)
    x[nons]= np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)
    y[nans]= np.interp(xx(nans), xx(~nans), y[~nans])

    track = [*zip(x,y)]
    end_time = time.time()
    print(f"轨迹插值总耗时: {end_time - start_time:.2f} 秒")
    return track

def write_track(frames, ball_track, path_output_video, fps, trace=7):
    start_write_time = time.time()
    height, width = frames[0].shape[:2]
    print(height, width)
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'), 
                          fps, (width, height))
    
    # 模仿 tqdm 进度条，显示视频写入进度
    pbar = tqdm(total=len(frames), desc="Writing video")
    for num in range(len(frames)):
        frame = frames[num]
        for i in range(trace):
            if (num-i > 0):
                if ball_track[num-i][0]:
                    x = int(ball_track[num-i][0])
                    y = int(ball_track[num-i][1])
                    # print(x, y)
                    frame = cv2.circle(frame, (x,y), radius=0, color=(0, 0, 255), thickness=10-i)
                else:
                    break
        out.write(frame)
        pbar.update(1)
    pbar.close()
    out.release() 
    end_write_time = time.time()
    print(f"视频写入总耗时: {end_write_time - start_write_time:.2f} 秒")

if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--video_path', type=str, help='path to input video')
    parser.add_argument('--video_out_path', type=str, help='path to output video')
    parser.add_argument('--extrapolation', action='store_true', help='whether to use ball track extrapolation')
    args = parser.parse_args()
    
    model = BallTrackerNet()
    device = 'cuda'
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    frames, fps = read_video(args.video_path)
    ball_track, dists = infer_model(frames, model)
    
    start_post_process_time = time.time()
    ball_track = remove_outliers(ball_track, dists)
    if args.extrapolation:
        subtracks = split_track(ball_track)
        for r in subtracks:
            ball_subtrack = ball_track[r[0]:r[1]]
            ball_subtrack = interpolation(ball_subtrack)
            ball_track[r[0]:r[1]] = ball_subtrack
    end_post_process_time = time.time()
    print(f"后处理（移除异常点和插值）总耗时: {end_post_process_time - start_post_process_time:.2f} 秒")
    
    write_track(frames, ball_track, args.video_out_path, fps)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"程序运行总耗时: {total_time:.2f} 秒")