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

def generate(i) : # input name, output name
    # load detector,shape predictor and image
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('res/68.dat')
    img = cv2.imread(i + "cropped.png") # i : cropped square image.
    image_height, image_width, _ = img.shape

    # get bounding box and facial landmarks
    boxes = detector(img)
    landmarks = []
    for box in boxes:
        shape = shape_predictor(img, box)
        index = 1

        center_point = [shape.parts()[5].x + shape.parts()[48].x,  shape.parts()[5].y + shape.parts()[48].y ]
         
        cl = img[int(center_point[1]/2), int(center_point[0]/2)] # face random color
        maxy = shape.parts()[24].y # eyebrow
        maxy_2 = shape.parts()[19].y # eyebrow

        # 얼굴 가장 좌, 우 지점
        x0 = shape.parts()[0].x
        y0 = shape.parts()[0].y
        x16 = shape.parts()[16].x
        y16 = shape.parts()[16].y

        # box for "good" face skin area.
        mx, my, MX, MY = shape.parts()[47].x + 20, shape.parts()[47].y + 20, shape.parts()[14].x - 10, shape.parts()[14].y - 10
        pts1 = np.float32([[mx, my],[mx, MY],[MX, my],[MX, MY]])
        pts2 = np.float32([[0, 0],[0, 512],[512, 0],[512, 512]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (512,512))

        # make inpainted image using good face skin area
        #inpaint_img = np.zeros_like(img)
        #inpaint_img[mx:MX, my:MY] = img[mx:MX, my:MY]
        #inpaint_mask = np.zeros_like(img)
        #inpaint_mask[mx:MX, my:MY] = [255, 255, 255]
        #inpaint_mask = cv2.cvtColor(inpaint_mask, cv2.COLOR_BGR2GRAY)
        #flip = inpaint_mask == 255
        #flip_inverse = inpaint_mask != 255
        #inpaint_mask[flip] = 0
        #inpaint_mask[flip_inverse] = 255

        #inpaint_img = cv2.inpaint(inpaint_img, inpaint_mask, 10, cv2.INPAINT_TELEA)
        """
        from matplotlib import pyplot as plt
        cv2.circle(img, (shape.parts()[0].x, shape.parts()[0].y), 20, (0, 50, 255), -1)
        cv2.circle(img, (shape.parts()[16].x, shape.parts()[16].y), 20, (0, 50, 255), -1)
        plt.imshow(img)
        plt.show()
        exit()
        """
        for part in shape.parts():
            landmarks.append(eos.core.Landmark(str(index),[float(part.x),float(part.y)]))
            index +=1
        break

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
    seg = cv2.imread(i+"mask.png") # 0 : background , 127 : hair, 254 : face // grayscale image
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    
    # need to up-sample mask so that mask is same size with input image.
    seg_alias = cv2.resize(seg, (img.shape[0], img.shape[1]))
    seg = cv2.resize(seg, (img.shape[0], img.shape[1]),interpolation=cv2.INTER_NEAREST) # no anti-alising
    face_boarder = seg != seg_alias 
    
    # mask
    background = seg == 0
    hair = seg == 127
    face = seg == 254
    # find donminant face color..
    #pixels = np.float32(img[face].reshape(-1, 3))
    
    #pixels = img[face].astype('float32')
    #n_colors = 5
    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    #flags = cv2.KMEANS_RANDOM_CENTERS

    #_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    #_, counts = np.unique(labels, return_counts=True)
    
    # option 1
    #dominant = palette[np.argmax(counts)] # dominant color.
    
    # option 2
    #dominant = np.average(palette, axis=0 ,weights=counts)
    
    # option 3 : any face color...
    #dominant = np.average(img[face], axis=0)
    
    # option 4 : face color
    # dominant = cl

    #visualize_dominant(counts, palette, img)
    # for debugging

    # key color
    key_color = [7, 28, 98]
    hair_color = [28, 7, 98]
    boarder_color = [98, 7, 28]

    img[hair] = key_color
    img[background] = key_color
    img[face_boarder] = boarder_color

    print(face_boarder.shape)
    
    #앞머리
    img[ : maxy-100 , :] = key_color
    img[ : maxy_2 - 100, :] = key_color
    
    # 옆면
    #img[:y0,:x0] = key_color
    #img[:y16, x16: ] = key_color
    

    isomap = eos.render.extract_texture(mesh, pose, img)
    isomap = cv2.transpose(isomap)


    empty = np.all( isomap == [0, 0, 0, 0], axis=-1)
    key_color = [7, 28, 98, 255]
    key_hair_color = [28, 7, 98, 255]
    key_boarder_color = [98, 7, 28, 255]

    kc = np.all( isomap == key_color, axis = -1)
    #kc_hair = np.all( isomap == key_hair_color, axis = -1)
    kc_boarder = np.all( isomap == key_boarder_color, axis = -1)

    use_dst = kc 
    mask = kc_boarder
    mask_gray = np.zeros_like(mask).astype('uint8')
    mask_gray[mask] = 255
    kernel = np.ones((5,5), np.uint8)
    mask_gray = cv2.dilate(mask_gray, kernel, iterations=3)
    mask_gray[empty] = 255
    #isomap[empty] = (dominant[0], dominant[1], dominant[2], 255)
    
    from scipy.ndimage import gaussian_filter
    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.show()

    eos.core.write_textured_obj(mesh, i + "face.obj")

    isomap = cv2.cvtColor(isomap, cv2.COLOR_RGBA2RGB)
    isomap[use_dst] = dst[use_dst] 

        
    
    plt.imshow(mask_gray)
    plt.show()

    isomap = cv2.inpaint(isomap, mask_gray, 21, cv2.INPAINT_TELEA) # kernel size (third parameter) could be lower to reduce time delay.
    
    plt.imshow(cv2.cvtColor(isomap, cv2.COLOR_BGR2RGB))
    plt.show()



    cv2.imwrite(i + "face.isomap.png", isomap)

if __name__ == "__main__":
    generate(sys.argv[1])
