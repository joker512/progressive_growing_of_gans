#!/usr/bin/python2

import sys
import os
import dlib
import glob
from skimage import io
import pool

if len(sys.argv) < 5:
    print("Usage:\n    ./dlib-extract.py shape_predictor_68_face_landmarks.dat base_dir faces_dir landmarks.txt [threads]")
    exit()

predictor_file = sys.argv[1]
base_dir = sys.argv[2]
faces_dir = sys.argv[3]
landmarks_file = sys.argv[4]
threads = int(sys.argv[5]) if len(sys.argv) >= 6 else 1

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

with open(os.path.join(base_dir, landmarks_file), "a+") as out:
    names = glob.glob(os.path.join(base_dir, faces_dir, "*.jpg"))
    print('Total {} names in the list'.format(len(names)))
    # for the source nvidia format
    #out.write("{}\n".format(len(names)))
    #out.write("leye_x leye_y reye_x reye_y nose_x nose_y lmouth_x lmouth_y rmouth_x rmouth_y\n")
    existing_names = set([e.split()[0] for e in out.readlines()[2:]])
    print('{} existing names'.format(len(existing_names)))
    names = [e for e in names if os.path.basename(e) not in existing_names]
    print('{} names after skipping existing names'.format(len(names)))

    def process_func(name):
        try:
            img = io.imread(name)
            dets = detector(img, 1)
        except:
            print("Skipping {} as wrong format".format(name))
            img = None
            dets = None
        return name, img, dets

    with pool.ThreadPool(threads) as threads:
        for k, (f, img, dets) in enumerate(threads.process_items_concurrently(names, process_func=process_func)):
            if dets is None:
                continue

            print("\nProcessing file {}: {}".format(k, f))
            print("Number of faces detected: {}".format(len(dets)))
            for det in dets:
                print("Detection: Left: {} Top: {} Right: {} Bottom: {}".format(
                    det.left(), det.top(), det.right(), det.bottom()))
                shape = predictor(img, det)
                print("Part 36: {}, Part 42: {}, ...".format(shape.part(36), shape.part(42)))
                points = []
                for index1, index2 in [(36, 42), (42, 48)]:
                    y, x = 0, 0
                    for index in xrange(index1, index2):
                        y += shape.part(index).y
                        x += shape.part(index).x
                    y /= index2 - index1
                    x /= index2 - index1
                    points.append((x, y))
                for index in [33, 48, 54]:
                    x, y = shape.part(index).x, shape.part(index).y
                    points.append((x, y))

                out.write(os.path.basename(f))
                for point in points:
                    out.write("  {}  {}".format(point[0], point[1]))
                out.write("\n")
