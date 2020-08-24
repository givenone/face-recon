import numpy as np
import dlib
import eos
import cv2
import sys

def read_pts(filename):
    """A helper function to read the 68 ibug landmarks from a .pts file."""
    lines = open(filename).read().splitlines()
    lines = lines[3:71]

    landmarks = []
    ibug_index = 1  # count from 1 to 68 for all ibug landmarks
    for l in lines:
        coords = l.split()
        landmarks.append(eos.core.Landmark(str(ibug_index), [float(coords[0]), float(coords[1])]))
        ibug_index = ibug_index + 1

    return landmarks

def visualize_dominant(counts, palette, img) :
    import matplotlib.pyplot as plt

    indices = np.argsort(counts)[::-1]   
    freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
    rows = np.int_(img.shape[0]*freqs)

    dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    plt.imshow(cv2.cvtColor(dom_patch, cv2.COLOR_BGR2RGB))
    plt.show()

def generate(i, o, mask) : # input name, output name
    # load detector,shape predictor and image
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('res/68.dat')
    img = cv2.imread(i) # i : cropped square image.
    image_height, image_width, _ = img.shape

    # get bounding box and facial landmarks
    boxes = detector(img)
    landmarks = []
    for box in boxes:
        shape = shape_predictor(img, box)
        index = 1
        for part in shape.parts():
            landmarks.append(eos.core.Landmark(str(index),[float(part.x),float(part.y)]))
            index +=1

    #landmarks = read_pts('res/image_0010.pts')
    # load eos model
    model = eos.morphablemodel.load_model("res/sfm_shape_3448.bin")
    blendshapes = eos.morphablemodel.load_blendshapes("res/expression_blendshapes_3448.bin")
    # Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
    morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                        color_model=eos.morphablemodel.PcaModel(),
                                                                        vertex_definitions=None,
                                                                        texture_coordinates=model.get_texture_coordinates())
    landmark_mapper = eos.core.LandmarkMapper('res/ibug_to_sfm.txt')
    edge_topology = eos.morphablemodel.load_edge_topology('res/sfm_3448_edge_topology.json')
    contour_landmarks = eos.fitting.ContourLandmarks.load('res/ibug_to_sfm.txt')
    model_contour = eos.fitting.ModelContour.load('res/sfm_model_contours.json')

    (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(morphablemodel_with_expressions,
        landmarks, landmark_mapper, image_width, image_height, edge_topology, contour_landmarks, model_contour)

    # read segmetation mask
    seg = cv2.imread(mask) # 0 : background , 127 : hair, 254 : face // grayscale image
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    
    # need to up-sample mask so that mask is same size with input image.
    seg = cv2.resize(seg, (img.shape[0], img.shape[1]))
    # mask
    background = seg == 0
    hair = seg == 127
    face = seg == 254
    # find donminant face color..
    #pixels = np.float32(img[face].reshape(-1, 3))
    
    pixels = img[face].astype('float32')
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    
    # option 1
    dominant = palette[np.argmax(counts)] # dominant color.
    
    # option 2
    dominant = np.average(palette, axis=0 ,weights=counts)
    
    #visualize_dominant(counts, palette, img)
    # for debugging
    
    img[hair] = dominant
    img[background] = dominant
    #img

    isomap = eos.render.extract_texture(mesh, pose, img)
    isomap = cv2.transpose(isomap)
    eos.core.write_textured_obj(mesh, o + ".obj")
    cv2.imwrite(o + ".isomap.png", isomap)

if __name__ == "__main__":
    generate(sys.argv[1], sys.argv[2], sys.argv[3])