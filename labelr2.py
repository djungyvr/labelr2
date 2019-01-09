import cv2
import sys
import numpy as np
import os

def match_files(idir, prefix):
    potential_files = os.listdir(idir)
    return sorted([os.path.join(idir, f) for f in potential_files if f.startswith(prefix)])

if __name__=="__main__":
    input_dir = sys.argv[1]
    output_filename = 'np_' + input_dir

    files = match_files(input_dir, 'yuv_')

    print("Labeling %d yuv frames" % (len(files)))

    frames = []
    labels = []

    for idx, yuv_fname in enumerate(files):
        with open(yuv_fname, 'rb') as yuv_file:
            raw = yuv_file.read()
            yuv_flat = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv_flat.reshape((1080, 1920, 2))
            gray = cv2.resize(cv2.cvtColor(yuv, cv2.COLOR_YUV2GRAY_YUYV), (256, 256))

            cv2.imshow(str(idx), gray)

            k = cv2.waitKey(0)

            assert isinstance(k, int)

            cv2.destroyWindow(str(idx))

            frames.append(gray)
            labels.append(int(chr(k)))

    np_frames = np.array(frames)
    np_labels = np.array(labels)

    print(np_frames.shape)
    print(np_labels.shape)

    np.savez_compressed(output_filename, frames=np_frames, labels=np_labels)
