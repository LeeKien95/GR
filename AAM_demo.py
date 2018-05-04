#import and show data
import os
import glob
import menpo.io as mio
from menpo.visualize import print_progress
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_68_trimesh
from menpowidgets import visualize_images
from menpofit.aam import HolisticAAM
from menpofit.aam import PatchAAM
from menpo.feature import fast_dsift
from menpo.feature import igo
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from menpodetect import load_dlib_frontal_face_detector

# %matplotlib inline


def process(image, crop_proportion=0.2, max_diagonal=400):
    if image.n_channels == 3:
        image = image.as_greyscale()
    image = image.crop_to_landmarks_proportion(crop_proportion)
    d = image.diagonal()
    if d > max_diagonal:
        image = image.rescale(float(max_diagonal) / d)
    
    labeller(image, 'PTS', face_ibug_68_to_face_ibug_68_trimesh)
    return image

path_to_database = '/Programing/GR/dataset/lfpw/trainset'
#print(mio.input.base.glob_with_suffix(path_to_database))
training_images = []
for img in print_progress(mio.import_images(path_to_database, verbose=True)):
    training_images.append(process(img))

#visualize_images(training_images)

aam = HolisticAAM(training_images, group='face_ibug_68_trimesh', diagonal=150,
                  scales=(0.5, 1.0), holistic_features=fast_dsift, verbose=True,
                  max_shape_components=20, max_appearance_components=150)

#aam = PatchAAM(training_images, group='PTS', patch_shape=[(15, 15), (23, 23)],
#                     diagonal=150, scales=(0.5, 1.0), holistic_features=fast_dsift,
#                     max_shape_components=20, max_appearance_components=150,
#                     verbose=True)

#aam.view_shape_models_widget()

from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional

fitter = LucasKanadeAAMFitter(aam, lk_algorithm_cls=WibergInverseCompositional,
                              n_shape=[5, 20], n_appearance=[30, 150])
#print(fitter)




path_to_landmarks = '/Programing/GR/Code/Python/test-images'
path_to_images = '/Programing/GR/Code/Python/test-images'

def save_landmark(filename, result):
    landmark_filename = filename
    print(landmark_filename)
    output = open(path_to_images + '/' + landmark_filename + '.pts', 'w')
    output.write("version: 1" + '\n' + "n_points:  68" + '\n' + "{" + '\n')
    array = result.final_shape.points
    for point in array:
        output.write(str(point[0]) + ' ' + str(point[1]) + '\n')
    output.write('}')
    output.close()


cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("test", gray)
    if not ret:
        break
    
    k = cv2.waitKey(1)
    
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "test-images/opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, gray)
        print("{} written!".format(img_name))
        img_counter += 1
        break

cam.release()

cv2.destroyAllWindows()


import menpo.io as mio
from menpodetect import load_dlib_frontal_face_detector

path_to_test = Path('test-images')

# Load and convert to grayscale
image = mio.import_image(path_to_test / 'opencv_frame_0.png')
if image.n_channels == 3:
        image = image.as_greyscale()
# Detect face

detect = load_dlib_frontal_face_detector()
bboxes = detect(image)

# Crop the image for better visualization of the result
image = image.crop_to_landmarks_proportion(0.3, group='dlib_0')
bboxes[0] = image.landmarks['dlib_0'].lms

if len(bboxes) > 0:
    # Fit AAM
    result = fitter.fit_from_bb(image, bboxes[0], max_iters=[70, 5])
    # print(result)

    # # Visualize
    # plt.subplot(131);
    # image.view()
    # bboxes[0].view(line_width=3, render_markers=False)
    # plt.gca().set_title('Bounding box')

    # plt.subplot(132)
    # image.view()
    # result.initial_shape.view(marker_size=4)
    # plt.gca().set_title('Initial shape')

    # plt.subplot(133)
    # image.view()
    # result.final_shape.view(marker_size=4, figure_size=(15, 13))
    # plt.gca().set_title('Final shape')

#save to file
landmark_filename = os.listdir(path_to_images)[0].split('.')[0]
save_landmark(landmark_filename, result)