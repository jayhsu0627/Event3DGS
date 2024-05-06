import os
import numpy as np
from matplotlib import pyplot as plt
import math
import json
import cv2
import imageio
import bm4d
import os
import colour
from colour_demosaicing import (
    ROOT_RESOURCES_EXAMPLES,
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)

num_events = 53644

class EventImageDatamanager:
    """
    A datamanager that outputs full images and cameras instead of raybundles.
    This makes the datamanager more lightweight since we don't have to do generate rays.
    Useful for full-image training e.g. rasterization pipelines.
    deblur_method = ["bilinear", "Malvar2004", "Menon2007"]
    The BM4D is the deblur stage, which take quite a long time. Default is off, but it can significantly
    improve synthetic datasets.
    """
    def __init__(self, event_file_path, pose_directory, out_directory, width, height, debayer_method=None, is_real= False , sigma=0):
        self.img_size = (height, width)
        self.debayer = False
        self.is_real = is_real # Real or Synthetic data
        self.cycle = True
        if self.is_real: self.cycle, self.max_winsize = True, 100
        else: self.max_winsize = 50
        self.randomize_winlen = True
        self.events_collections = {}
        self.files = []
        self.idx = []
        self.idx_pre = []
        self.frame = []
        self.frame_pre  = []

        # self.img = np.zeros(self.img_size, dtype=np.int8)
        self.event_file_path = event_file_path
        self.pose_directory = pose_directory
        self.out_directory = out_directory
        self.F = np.array([[[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1]]])
        self.F_tile = np.tile(self.F, (int(height/2), int(width/2), 1))
        self.debayer_method = debayer_method
        self.deblur = False

        if sigma>0:
            self.deblur = True
            self.sigma = sigma
        self.getEventData()

        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(self.out_directory):
            os.makedirs(self.out_directory)

    def loadEventNPZData(self):
        """Idx: Store the event line of the timestep for each output frames. range from [0-999]
           Usage: idx[t_i-1] to idx[t_i] capture events motion image.
                   0         to idx[t_i] capture RGB image
        """
        self.event_data = np.load(self.event_file_path)
        self.timestamp, self.x, self.y, self.pol = self.event_data['t'], self.event_data['x'], self.event_data['y'], self.event_data['p']
        print("Data length:", len(self.timestamp))

        self.get_Event_Pair()

        # print("idx: ", self.idx)
        # print("pre_idx: ", self.idx_pre)
        print("events_collections: ")
        for key in self.events_collections:
            print(key)
            print(self.events_collections[key])

    def loadEventTXTData(self):
        self.event_data = np.loadtxt(self.event_file_path,dtype={'formats': ('f4', 'i4', 'i4', 'i4')})
        # print(self.event_data)
        # print(self.event_data.shape)
        self.timestamp, self.x, self.y, self.pol = self.event_data[:,0], self.event_data[:,1], self.event_data[:,2], self.event_data[:,3]

        print("Data length:", len(self.timestamp))
        self.get_Event_Pair()

        # print("idx: ", self.idx)
        # print("pre_idx: ", self.idx_pre)
        print("events_collections: ")
        for key in self.events_collections:
            print(key)
            print(self.events_collections[key])

    def get_Event_Pair(self):
        max_winsize = self.max_winsize
        start_range = 0 if self.cycle else 1+max_winsize
        j=0
        t = self.timestamp[0]
        frame = 0
        for j in range(107500):
            if self.timestamp[j]>t:
                print("Data shutter:", j)
                break
        start_pt = 0
        for k in range(start_pt,len(self.timestamp)):
            if self.timestamp[k]>t:
                # print('Data streams per frame:',k-start_pt,t)
                start_pt = k
                t+= (self.timestamp[-1] - self.timestamp[0])/1000
                self.files.append(k-1)
                continue

        # self.events_collections["idx"] = self.idx
        
        self.events_collections["events"] = []

        # Calculate start and end streams and load events

        for i in range(start_range, len(self.files)):
            if self.randomize_winlen:
                winsize = np.random.randint(1, max_winsize+1)
            else:
                winsize = max_winsize

            # what is -1 for? for i=last frame covering all events
            start_time = (i-winsize)/(len(self.files)-1)
            if start_time < 0:
                start_time += 1
            end_time = (i)/(len(self.files)-1)

            # Ending Streams lines
            end = np.searchsorted(self.timestamp, end_time * self.timestamp.max())

            # if win_constant_count != 0:
            #     # TODO: there could be a bug with windows in the start, e.g., end-win_constant_count<0
            #     #       please, check if the windows are correctly composed in that case
            #     start_time = self.timestamp[end-win_constant_count]/self.timestamp.max()

            #     if win_constant_count > end:
            #         start_time = start_time - 1

            #     winsize = int(i-start_time*(len(self.files)-1))
            #     assert(winsize>0)
            #     start_time = (i-winsize)/(len(self.files)-1)

            #     if start_time < 0:
            #         start_time += 1

            # Find Start Streams lines
            start = np.searchsorted(self.timestamp, start_time * self.timestamp.max())
            # print(start, end, end-start)
            if start <= end:
                # normal case: take the interval between
                events = (self.x[start:end], self.y[start:end], self.timestamp[start:end], self.pol[start:end])
            else:
                # loop over case: compose tail with head events
                events = (np.concatenate((self.x[start:], self.x[:end])),
                        np.concatenate((self.y[start:], self.y[:end])),
                        np.concatenate((self.timestamp[start:], self.timestamp[:end])),
                        np.concatenate((self.pol[start:], self.pol[:end])),
                        )
            # print(events)

            # print(start_time, end_time, start, end, (i-winsize+len(self.files))%len(self.files), i )
            # Corresponding stream number of t0
            self.idx_pre.append(start)
            # Corresponding stream number of t
            self.idx.append(end)
            # t0 frame: for example 3 (uniform pick from 0-100)
            self.frame_pre.append((i-winsize+len(self.files))%len(self.files))
            # t frame: for example 100
            self.frame.append(i)
            self.events_collections["events"].append(events)

        self.events_collections["idx_pre"] = self.idx_pre
        self.events_collections["idx"] = self.idx
        self.events_collections["frame_pre"] = self.frame_pre
        self.events_collections["frame"] = self.frame
        
        return self.events_collections

    def getFileType(self):
        root, ext = os.path.splitext(self.event_file_path)
        return ext

    def getEventData(self):
        if self.getFileType() == ".txt":
            print("Loaded txt")
            self.loadEventTXTData()

        if self.getFileType() == ".npz":
            print("Loaded npz")
            self.loadEventNPZData()

    def scale_img(self, array):
        min_val = np.min(array)
        max_val = np.max(array)
        sf = 255 / (max_val - min_val)
        scaled_array = ((array - min_val) * sf).astype(np.uint8)
        return scaled_array

    def convertCameraImg(self, num_events, start=0):
        # print("before:", np.max(img), np.min(img))
        # self.img = np.zeros(self.img_size , dtype=np.float32)
        # self.img = np.zeros(self.img_size , dtype=np.float32) + np.log(125) / 2.2
        self.img = np.zeros(self.img_size , dtype=np.float32) + np.log(127) / 2.2
        self.bayer = np.zeros(self.img_size, np.float32)
        start = start

        # print(self.img[:5,:5])

        print("Load event img at :", num_events)
        self.t_ref = self.timestamp[0] # time of the last event in the packet
        self.tau = 0.03 # decay parameter (in seconds)
        self.dt = num_events*10

        for i in range(start, num_events):
            self.img[self.y[i], self.x[i]] += self.pol[i]

        bg_mask = self.img == np.log(127) / 2.2

        self.img = np.tile(self.img[..., None], (1, 1, 3))
        # img = np.tile(img[..., None], (1, 1, 3)) + np.log(159) / 2.2

        print(self.img[:2,:2])
        print("1. Before:", np.max(self.img), np.min(self.img), self.img.shape)

        self.bayer = self.scale_img(self.img)

        # Apply mask
        # self.bayer[bg_mask] = 198

        self.img_gray = self.bayer
        self.bayer = self.F_tile * self.img_gray
        print("bayer input", np.max(self.bayer), np.min(self.bayer))
        self.bayer = np.clip(np.exp(self.bayer * 2.2), 0, 255).astype(np.uint8)

        if self.debayer_method:
            # mosaic
            self.CFA = mosaicing_CFA_Bayer(self.img_gray)
            print('here')
            # Menon2007
            if self.debayer_method == "bilinear":
                self.bayer = demosaicing_CFA_Bayer_bilinear(self.CFA)
            if self.debayer_method == "Malvar2004":
                self.bayer = demosaicing_CFA_Bayer_Malvar2004(self.CFA)
            if self.debayer_method == "Menon2007":
                self.bayer = demosaicing_CFA_Bayer_Menon2007(self.CFA)
            self.bayer = self.scale_img(self.bayer)
            print("debayer_method", np.max(self.bayer), np.min(self.bayer))

        if self.deblur:
            self.bayer = bm4d.bm4d(self.bayer, self.sigma); # white noise: include noise std
            self.bayer = self.scale_img(self.bayer)

        # # Set background color for RGB
        # self.bayer[bg_mask] = 125

        # # Apply mask
        # self.bayer[bg_mask] = 198
        # self.bayer = np.clip(np.exp(self.bayer * 2.2), 0, 255).astype(np.uint8)
        # self.bayer = np.exp(self.bayer * 2.2)

        print(self.bayer[:2,:2])
        print("2. After:", np.max(self.bayer), np.min(self.bayer), self.bayer.shape)

        return self.img_gray, self.bayer

    def AccuDiffCameraImg(self, t_0, t):
        """ Eq.3 the observed events {Ei}_{i=1}^N between rendered views (multiplied by Bayer colour filter)
        taken at two different time instants t0 and t. (t - t_0).
        """
        return self.convertCameraImg(t, t_0)

    def ImgPlot(self, img):
        fig = plt.figure(figsize=(21,6))
        plt.subplot(1,4,1)
        plt.imshow(img[:,:,0], clim=(0, 255))
        plt.subplot(1,4,2)
        plt.imshow(img[:,:,1], clim=(0, 255))
        plt.subplot(1,4,3)
        plt.imshow(img[:,:,2], clim=(0, 255))
        plt.subplot(1,4,4)
        plt.imshow(img, clim=(0, 255))
        plt.show()

    def ImgSave(self, img, file_path, file_name):
        # Save the image to a PNG file
        imageio.imwrite(file_path + '/' + str(file_name) +'.png', img)

    def convert_to_images(self):
        """nerfstudio style/format
        """
        OUT_PATH = os.path.normpath(self.out_directory + '/' + "images" + "/")
        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(OUT_PATH):
            os.makedirs(OUT_PATH)
        # Loop all the images
        frame = 0
        self.img = np.zeros(self.img_size , dtype=np.float32) + np.log(127) / 2.2
        self.bayer = np.zeros(self.img_size, np.float32)
        num_events = len(self.timestamp)
        
        # if not self.is_real:

        start = 0
        while frame < len(self.files)-1:
            for i in range(start, num_events-1):
                img_name = str(f"r_{'{:05d}'.format(frame)}")
                self.img[self.y[i], self.x[i]] += self.pol[i]
                # If the event frame reach the camera frame, trigger save
                if frame == len(self.files): break
                if i == self.idx[frame]:
                    print(i, frame,self.idx[frame])
                    frame += 1
                    bg_mask = self.img == np.log(127) / 2.2
                    self.img_temp = np.tile(self.img[..., None], (1, 1, 3))
                    self.bayer = self.scale_img(self.img_temp)

                    self.img_gray = self.bayer
                    self.bayer = self.F_tile * self.img_gray
                    self.bayer = np.clip(np.exp(self.bayer * 2.2), 0, 255).astype(np.uint8)

                    if self.debayer_method:
                        # mosaic
                        self.CFA = mosaicing_CFA_Bayer(self.img_gray)
                        # Menon2007
                        if self.debayer_method == "bilinear":
                            self.bayer = demosaicing_CFA_Bayer_bilinear(self.CFA)
                        if self.debayer_method == "Malvar2004":
                            self.bayer = demosaicing_CFA_Bayer_Malvar2004(self.CFA)
                        if self.debayer_method == "Menon2007":
                            self.bayer = demosaicing_CFA_Bayer_Menon2007(self.CFA)
                        self.bayer = self.scale_img(self.bayer)

                    if self.deblur:
                        self.bayer = bm4d.bm4d(self.bayer, self.sigma); # white noise: include noise std
                        self.bayer = self.scale_img(self.bayer)
                    print(OUT_PATH, img_name)

                    self.ImgSave( self.bayer, OUT_PATH, img_name)

    def convert_to_images_uniform(self):
        """nerfstudio style/format; Use the uniform selected previous frame stroed in self.events_collections
            Designed for real event camera data.
        """
        OUT_PATH = os.path.normpath(self.out_directory + '/' + "images" + "/")
        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(OUT_PATH):
            os.makedirs(OUT_PATH)
        
        # if not self.is_real:

        start = 0
        for frame in self.events_collections["frame"]:
            print(frame, self.events_collections["idx_pre"][frame], self.events_collections["idx"][frame])
            temp_event = self.events_collections["events"][frame]
            # print(temp_event)
            print(temp_event[0].shape)
            curr_x, curr_y, _, curr_pol = temp_event[0], temp_event[1], temp_event[2], temp_event[3]
            self.img = np.zeros(self.img_size , dtype=np.float32) + np.log(127) / 2.2
            self.bayer = np.zeros(self.img_size, np.float32)

            for i in range(0, len(curr_x)):
                img_name = str(f"r_{'{:05d}'.format(frame)}")
                self.img[curr_y[i], curr_x[i]] += curr_pol[i]

            print(frame, img_name)
            bg_mask = self.img == np.log(127) / 2.2
            self.img_temp = np.tile(self.img[..., None], (1, 1, 3))
            self.bayer = self.scale_img(self.img_temp)

            self.img_gray = self.bayer
            self.bayer = self.F_tile * self.img_gray
            self.bayer = np.clip(np.exp(self.bayer * 2.2), 0, 255).astype(np.uint8)

            if self.debayer_method:
                # mosaic
                self.CFA = mosaicing_CFA_Bayer(self.img_gray)
                # Menon2007
                if self.debayer_method == "bilinear":
                    self.bayer = demosaicing_CFA_Bayer_bilinear(self.CFA)
                if self.debayer_method == "Malvar2004":
                    self.bayer = demosaicing_CFA_Bayer_Malvar2004(self.CFA)
                if self.debayer_method == "Menon2007":
                    self.bayer = demosaicing_CFA_Bayer_Menon2007(self.CFA)
                self.bayer = self.scale_img(self.bayer)

            if self.deblur:
                self.bayer = bm4d.bm4d(self.bayer, self.sigma); # white noise: include noise std
                self.bayer = self.scale_img(self.bayer)
            print(OUT_PATH, img_name)

            self.ImgSave( self.bayer, OUT_PATH, img_name)
            # self.ImgSave( self.img_gray, OUT_PATH, img_name)

            print(f"writing {OUT_PATH}")

    def PoseRead(self, file_path, frame_num):
        file_path = os.path.join(file_path, 'r_'+'{:05d}'.format(frame_num) + ".txt")
        try:
            with open(file_path, 'r') as file:
                # Read each line in the file
                lines = file.readlines()
                # Parse each line to extract the elements of the camera matrix
                camera_matrix = []
                for line in lines:
                    elements = line.split()
                    camera_matrix.append([float(element) for element in elements])
                camera_matrix = np.array(camera_matrix)
                # print("test:", camera_matrix)
                return camera_matrix

        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def rotmat(self, a, b):
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

    def closest_point_2_lines(self, oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
        da = da / np.linalg.norm(da)
        db = db / np.linalg.norm(db)
        c = np.cross(da, db)
        denom = np.linalg.norm(c)**2
        t = ob - oa
        ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
        tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
        if ta > 0:
            ta = 0
        if tb > 0:
            tb = 0
        return (oa+ta*da+ob+tb*db) * 0.5, denom

    def convert_to_json(self):
        """nerfstudio style/format
        """
        AABB_SCALE = 1
        text = os.path.normpath(self.pose_directory) # + '/text')
        OUT_PATH = os.path.normpath(self.out_directory + '/' + "transforms.json")
        # sparce = os.path.normpath(args.scenedir + '/sparse')

        # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
        # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
        # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443

        w = self.img_size[1]
        h = self.img_size[0]
        fl_x = 389.25 if self.is_real else 480.55
        fl_y = 389.25 if self.is_real else 480.55
        k1 = 0
        k2 = 0
        p1 = 0
        p2 = 0
        cx = w / 2
        cy = h / 2

        # angle_x = math.atan(w / (fl_x * 2)) * 2
        # angle_y = math.atan(h / (fl_y * 2)) * 2
        # fovx = angle_x * 180 / math.pi
        # fovy = angle_y * 180 / math.pi

        # with open(os.path.join(text,"images.txt"), "r") as f:
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        out = {
            # "camera_angle_x": angle_x,
            # "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "camera_model": "OPENCV",
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            # "aabb_scale": AABB_SCALE,
            "frames": [],
        }

        up = np.zeros(3)

        for i in range(len(self.files)):
            if  i % 1 == 0:
                # elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                pose_path = self.pose_directory
                name = str(f"./images/r_{'{:05d}'.format(i)}.png")
                print(name)
                # b=sharpness(os.path.normpath(f"{args.scenedir}/{args.images}/{elems[9]}"))
                # image_id = int(elems[0])

                # qvec = np.array(tuple(map(float, elems[1:5])))
                # tvec = np.array(tuple(map(float, elems[5:8])))
                # R = qvec2rotmat(-qvec)
                # t = tvec.reshape([3,1])
                # m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)

                # m = self.PoseRead(pose_path, i)
                # c2w = np.linalg.inv(m)
                # up += c2w[0:3,1]

                # https://github.com/nerfstudio-project/nerfstudio/issues/1504
                c2w = self.PoseRead(pose_path, i)
                c2w[0:3,2] *= -1 # flip the y and z axis
                c2w[0:3,1] *= -1
                c2w = c2w[[1,0,2,3],:] # swap y and z
                c2w[2,:] *= -1 # flip whole world upside down

                # Add pre_camera for the re-ordering in event_nerfstudio_dataparser.py
                frame={"file_path":name, "pre_camera": self.events_collections["frame_pre"][i], "transform_matrix": c2w}
                out["frames"].append(frame)

        nframes = len(out["frames"])



        # up = up / np.linalg.norm(up)
        # print("up vector was", up)
        # R = self.rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
        # R = np.pad(R,[0,1])
        # R[-1, -1] = 1


        # for f in out["frames"]:
        #     f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

        # # find a central point they are all looking at
        # print("computing center of attention...")
        # totw = 0.0
        # totp = np.array([0.0, 0.0, 0.0])
        # for f in out["frames"]:
        #     mf = f["transform_matrix"][0:3,:]
        #     for g in out["frames"]:
        #         mg = g["transform_matrix"][0:3,:]
        #         p, w = self.closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
        #         if w > 0.01:
        #             totp += p*w
        #             totw += w
        # totp /= totw
        # print(totp) # the cameras are looking at totp

        # for f in out["frames"]:
        #     f["transform_matrix"][0:3,3] -= totp

        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)

        for f in out["frames"]:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

        for f in out["frames"]:
            f["transform_matrix"] = f["transform_matrix"].tolist()

        print(nframes,"frames")
        print(f"writing {OUT_PATH}")
        with open(OUT_PATH, "w") as outfile:
            json.dump(out, outfile, indent=2)
        print("Writing Completed")


# def scale_img( array):
#     min_val = np.min(array)
#     max_val = np.max(array)
#     sf = 255 / (max_val - min_val)
#     scaled_array = ((array - min_val) * sf).astype(np.uint8)
#     return scaled_array

# events_path = 'C:\\Users\\sjxu\\Downloads\\data\\data\\nerf\\lego\\train\\events\\test_lego1_color_init159_1e-6eps.npz'
# eventData = EventImageDatamanager(events_path, "C:\\Users\\sjxu\\Downloads\\data\\data\\nerf\\lego\\train\\pose",
#                                 "C:\\Users\\sjxu\\3_Event_3DGS\\Data\\nerfstudio\\lego", 346, 260, debayer_method="Menon2007", is_real=False, sigma=0)

events_path = 'C:\\Users\\sjxu\\Downloads\\data\\data\\nerf\\drums\\train\\events\\events.npz'
eventData = EventImageDatamanager(events_path, "C:\\Users\\sjxu\\Downloads\\data\\data\\nerf\\drums\\train\\pose",
                                "C:\\Users\\sjxu\\3_Event_3DGS\\Data\\nerfstudio\\drums", 346, 260, debayer_method="Menon2007", is_real=False, sigma=0)


# events_path = 'C:\\Users\\sjxu\\Downloads\\data\\data\\real\\sewing\\train\\events\\b10_cal1_45rpm_gfox_eonly-2022_05_12_01_17_04_shift_ts1.npz'
# eventData = EventImageDatamanager(events_path, "C:\\Users\\sjxu\\Downloads\\data\\data\\real\\sewing\\train\\pose",
#                                 "C:\\Users\\sjxu\\3_Event_3DGS\\Data\\nerfstudio\\sewing", 346, 260, debayer_method="Menon2007", is_real=True, sigma=0)

# events_path = 'C:\\Users\\sjxu\\Downloads\\data\\data\\real\\chick\\train\\events\\b10_cal1_45rpm_gfox_eonly-2022_05_12_00_53_10_shift_ts1.npz'
# eventData = EventImageDatamanager(events_path, "C:\\Users\\sjxu\\Downloads\\data\\data\\real\\chick\\train\\pose",
#                                 "C:\\Users\\sjxu\\3_Event_3DGS\\Data\\nerfstudio\\chick", 346, 260, debayer_method="Menon2007", is_real=False, sigma=0)

# eventData.events_collections['frame']
eventData.convert_to_json()
# eventData.convert_to_images()
eventData.convert_to_images_uniform()