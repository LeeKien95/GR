import menpo.io as mio

from menpo.visualize import print_process
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_68_trimesh
path_to_images = '/001/'
training_images = []

for img in print_process(mio.import_images(path_to_images, verbose = TRUE)):
    #conver to grayscale
    if img.n_channel == 3:
        img = img.as_grayscale()
    
    #crop to landmarks bounding box with an extra 20% padding
    img = img.crop_to_landmark_proportion(0.2)
    
    #resclae image if its diagonal is bigger than 400 pixels
    
    d = img.diagonal()
    
    if d > 400:
        img = img.rescale(400.0/d)
    
    #define a TriMesh which will be userful for Piecewise Affine Warp of HolisticAAM
    laberller(img, 'PTS', face_ibug_68_to_face_ibug_68_trimesh)
    
    #append to list
    training_images.append(img)
    
%matplotlib inline
from menpowidgets import visualize_images
visualize_images(training_images)
    