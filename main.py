from imutils import contours
from skimage import feature
from skimage.transform import probabilistic_hough_line
from scipy.spatial import distance as dist
import numpy as np
import cv2, random,copy, time, game, os

STOP_ON_IMAGE_1 = True
STOP_ON_IMAGE_2 = True
STOP_ON_IMAGE_2b = True
STOP_ON_IMAGE_2c = True
STOP_ON_IMAGE_2d = True
STOP_ON_IMAGE_3 = True
STOP_ON_IMAGE_4 = True
STOP_ON_IMAGE_5 = True

#for interior coloration
STOP_ON_IMAGE_5b = True


STOP_ON_IMAGE_6 = True
STOP_ON_IMAGE_7 = True
STOP_ON_IMAGE_8 = True
STOP_ON_IMAGE_9 = False

CAM_ENABLED = True

class Card:
    def __init__(self,shape_text,color_text,numb_text,inter_text):
        self.shape_text = shape_text
        self.color_text = color_text
        self.numb_text = numb_text
        self.inter_text = inter_text

        self.shape = None
        self.color = None
        self.numb = None
        self.inter = None

        self.coord = None
        self.pic = None

def get_color(i):
    color_bank = [
        (238,130,238),
        (208,32,144),
        (255, 182, 193),
        (238, 162, 173),
        (205, 129, 98),
        (176, 226, 255),
        (3, 168, 158),
        (227, 168, 105),
        (245, 255, 250),
        (189, 252, 201),
        (205, 55, 0),
        (219, 112, 147),
        (205, 104, 137),
        (72, 118, 255),
        (255, 245, 238),
        (130, 255, 127),
    ]

    return color_bank[i]

def determine_color_from_card(card,num):
    #card = cv2.UMat.get(card)
    card_scaled = cv2.resize(card, (228, 113))
    card_scaled = card_scaled[5:103, 14:214]  # trim the TINIEST big off so there are less edges

    ### green ####

    # define range of yellow color in HSV
    lower_g = np.array([45,15,100])
    upper_g = np.array([65,255,255])

    # Convert BGR to HSV
    hsv = cv2.cvtColor(card_scaled, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    g_mask = cv2.inRange(hsv, lower_g, upper_g)

    #Apply the mask

    green = cv2.countNonZero(g_mask)
    g_per = float(green) / (card_scaled.shape[0] * card_scaled.shape[1])

    ### purple ####
    # color:  [[[128  74 138]]]

    # define range of yellow color in HSV
    lower_p = np.array([115,20,25])
    upper_p = np.array([132,255,255])

    # Convert BGR to HSV
    hsv = cv2.cvtColor(card_scaled, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    p_mask = cv2.inRange(hsv, lower_p, upper_p)

    #Apply the mask

    purp = cv2.countNonZero(p_mask)
    p_per = float(purp) / (card_scaled.shape[0] * card_scaled.shape[1])

    ### red ####

    # define range of yellow color in HSV
    lower_r = np.array([0,20,50])
    upper_r = np.array([10,255,255])

    # Convert BGR to HSV
    hsv = cv2.cvtColor(card_scaled, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    r_mask = cv2.inRange(hsv, lower_r, upper_r)

    #Apply the mask

    red = cv2.countNonZero(r_mask)
    r_per = float(red) / (card_scaled.shape[0] * card_scaled.shape[1])

    color = "purp"
    if r_per > p_per and r_per > g_per:
        color = "red"
    if g_per > p_per and g_per > r_per:
        color = "green"
    if p_per > r_per and p_per > g_per:
        color = "purp"

    if STOP_ON_IMAGE_5:
        g_res = cv2.bitwise_and(card_scaled, card_scaled, mask=g_mask)
        cv2.imshow("g_res",g_res)
        print("green_%: ",g_per)

        p_res = cv2.bitwise_and(card_scaled, card_scaled, mask=p_mask)
        cv2.imshow("p_res",p_res)
        print("purple_%: ",p_per)

        r_res = cv2.bitwise_and(card_scaled, card_scaled, mask=r_mask)
        cv2.imshow("r_res",r_res)
        print("red_%: ",r_per)

        print("    ",color)
        print("--")

        cv2.waitKey(0)

    #Get color swatch of interior
    if num == 0 or num == 2:
        x = 91
        w = 18
    elif num == 1:
        x = 120
        w = 18
    y = 37
    h = 24
    color_swatch = card_scaled[y:y + h, x:x + w]

    n_w = 200
    n_h = 98
    n_y = 0
    n_x = 0
    neut_swatch = card_scaled[n_y:n_y + n_h, n_x:n_x + n_w]

    #hsv the swatches:
    color_hsv = cv2.cvtColor(color_swatch, cv2.COLOR_BGR2HSV)
    neut_hsv = cv2.cvtColor(neut_swatch, cv2.COLOR_BGR2HSV)

    color_hsv = np.array(color_hsv[0]).T
    neut_hsv = np.array(neut_hsv[0]).T

    color_h = np.median(color_hsv[0])
    neut_h = np.median(neut_hsv[0])

    color_s = np.median(color_hsv[1])
    neut_s = np.median(neut_hsv[1])

    color_v = np.median(color_hsv[2])
    neut_v = np.median(neut_hsv[2])

    delt_v = color_v - neut_v
    delta_s = color_s - neut_s

    #Red thresholds:
    if color == "red":
        delt_s_1 = 10
        delt_s_2 = 90

        if (delta_s <= delt_s_2 and delta_s >= delt_s_1):
            inter = "striped"
        elif delta_s > delt_s_2:
            inter = "solid"
        else:
            inter = "blank"

    #Green thresholds:
    if color == "green":
        delt_v_1 = -7
        delt_v_2 = -50

        if (delt_v <= delt_v_1 and delt_v >= delt_v_2):
            inter = "striped"
        elif delt_v < delt_v_2:
            inter = "solid"
        else:
            inter = "blank"


    #Purple thresholds:
    if color == "purp":
        delt_v_1 = -20
        delt_v_2 = -60

        if (delt_v <= delt_v_1 and delt_v >= delt_v_2):
            inter = "striped"
        elif delt_v < delt_v_2:
            inter = "solid"
        else:
            inter = "blank"


    if STOP_ON_IMAGE_5b:
        print("hsv")
        print(color_h - neut_h)
        print(color_s - neut_s)
        print(color_v - neut_v)
        print(inter)

        card_disp = copy.deepcopy(card_scaled)
        cv2.rectangle(card_disp, (x,y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(card_disp, (n_x,n_y), (n_x+n_w, n_y + n_h), (255, 0, 0), 2)
        cv2.imshow("card_disp",card_disp)
        cv2.imshow("color_swatch",color_swatch)
        cv2.imshow("neut_swatch",neut_swatch)
        cv2.waitKey(0)

    return color,inter

def determine_num_from_loc(loc):
    num_vote = [0, 0, 0]

    for pt in zip(*loc[::-1]):
        if pt[1] < 10:
            if pt[0] >= 2 and pt[0] <= 14:
                num_vote[2] += 1
            if pt[0] >= 65 and pt[0] <= 73:
                num_vote[0] += 1
            if pt[0] >= 120 and pt[0] <= 140:
                num_vote[2] += 1
            if pt[0] >= 30 and pt[0] <= 49:
                num_vote[1] += 1
            if pt[0] >= 89 and pt[0] <= 108:
                num_vote[1] += 1

    num_winner = num_vote.index(max(num_vote))
    return num_winner,sum(num_vote)

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-coordinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.asarray([tl, tr, br, bl], dtype=pts.dtype)

def subimage(image, center, theta, width, height):

   '''
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   '''

   # Uncomment for theta in radians
   #theta *= 180/np.pi

   shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

   matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
   image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

   x = int( center[0] - width/2  )
   y = int( center[1] - height/2 )

   image = image[ y:y+height, x:x+width ]

   return image

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)

def hough_transform(roi):
    # Hough Transform#

    roi_grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #roi_grey = cv2.threshold(roi_grey, 40, 255, cv2.THRESH_BINARY)[1]
    # Line finding using the Probabilistic Hough Transform
    edge = cv2.Canny(roi_grey, 50, 200, 5)
    lines = cv2.HoughLinesP(edge, 1, np.pi/360, threshold=40, minLineLength=50, maxLineGap=100)

    xs = []
    ys = []

    for l in lines:
        xs.append(l[0][0] + l[0][2])
        ys.append(l[0][1] + l[0][3])

    #leftmost x
    l_x = xs.index(min(xs))
    l_x = lines[l_x][0]

    l_x_A = (l_x[0],l_x[1])
    l_x_B = (l_x[2], l_x[3])

    #rightmost x
    r_x = xs.index(max(xs))
    r_x = lines[r_x][0]
    r_x_A = (r_x[0], r_x[1])
    r_x_B = (r_x[2], r_x[3])

    #topmost y
    t_y = ys.index(min(ys))
    t_y = lines[t_y][0]
    t_y_A = (t_y[0], t_y[1])
    t_y_B = (t_y[2], t_y[3])

    #bottommost y
    b_y = ys.index(max(ys))
    b_y = lines[b_y][0]
    b_y_A = (b_y[0], b_y[1])
    b_y_B = (b_y[2], b_y[3])

    if STOP_ON_IMAGE_2d:
        drawing = copy.deepcopy(roi)  # np.zeros((roi.shape[0], roi.shape[1], 1), dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(drawing, (x1, y1), (x2, y2), (0, 0, 255), 1)
        print(len(lines))
        cv2.imshow("drawing", drawing)

        cv2.imshow("roi", roi)

        cv2.imshow("edge", edge)
        cv2.waitKey(0)

    #topleft coincident point
    try:
        tl_pt = line_intersection((l_x_A, l_x_B), (t_y_A, t_y_B))

        #toprightmost coincident point
        tr_pt = line_intersection((r_x_A, r_x_B), (t_y_A, t_y_B))

        #bottomleftmost coincident point
        bl_pt = line_intersection((l_x_A, l_x_B), (b_y_A, b_y_B))

        #bottomrightmost coincident point
        br_pt = line_intersection((r_x_A, r_x_B), (b_y_A, b_y_B))
    except:
        raise Exception("Failed line intersection")
    # Show result
    if STOP_ON_IMAGE_2b:
        drawing = copy.deepcopy(roi)  # np.zeros((roi.shape[0], roi.shape[1], 1), dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(drawing, (x1, y1), (x2, y2), (0, 0, 255), 1)

        #draw points for corners
        cv2.circle(drawing, tl_pt,3,(0,255,0), -1)
        cv2.circle(drawing, bl_pt,3,(0,255,0), -1)
        cv2.circle(drawing, tr_pt,3,(0,255,0), -1)
        cv2.circle(drawing, br_pt,3,(0,255,0), -1)

        cv2.imshow("drawing", drawing)
        cv2.waitKey(0)

    pts = [tl_pt,tr_pt,br_pt,bl_pt]

    return pts

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	#rect = order_points(pts)
    rect = np.asarray([pts[0], pts[1], pts[2], pts[3]], dtype=np.float32)
    (tl, tr, br, bl) = rect
    #(pts[0], pts[1], pts[2], pts[3]) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    try: warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    except: raise Exception("whoa wild warp")
	# return the warped image
    return warped

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0
    return img_crop

def segment_image_into_cards_hough(ref_color):
    rois = []
    cards = []
    positions = []

    rad_thresh_1 = 50 #100
    rad_thresh_2 = 100 #400


    ref_bnw = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
    ref_thresh = cv2.threshold(ref_bnw, 80, 255, cv2.THRESH_BINARY_INV)[1]

    if STOP_ON_IMAGE_1:
        # store image shape
        scale_percent = 40  # percent of original size
        width = int(ref_thresh.shape[1] * scale_percent / 100)
        height = int(ref_thresh.shape[0] * scale_percent / 100)
        dim = (width, height)

        rs_ref_thresh = cv2.resize(ref_thresh, dim, interpolation=cv2.INTER_AREA)
        rs_ref_bnw = cv2.resize(ref_bnw, dim, interpolation=cv2.INTER_AREA)
        rs_ref_color = cv2.resize(ref_color, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow('ref_thresh', rs_ref_thresh)
        cv2.imshow('ref_bnw', rs_ref_bnw)
        cv2.imshow('ref_color', rs_ref_color)

        cv2.waitKey(0)

    contours, _ = cv2.findContours(ref_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if STOP_ON_IMAGE_2: drawing = np.zeros((ref_thresh.shape[0], ref_thresh.shape[1], 3), dtype=np.uint8)

    for i, c in enumerate(contours):
        epsilon = 0.1 * cv2.arcLength(c, True)
        contour = cv2.approxPolyDP(c, epsilon, True)
        center, rad = cv2.minEnclosingCircle(contour) #maybe take out but using for now

        if rad > rad_thresh_1 and rad < rad_thresh_2:
            rect = cv2.minAreaRect(c)

            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # crop
            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)

            # Center of rectangle in source image
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            # Size of the upright rectangle bounding the rotated rectangle
            size = (x2 - x1, y2 - y1)

            roi_cropped = cv2.getRectSubPix(ref_color, size, center)

            rois.append(roi_cropped)
            try:
                #roi_uint = cv2.UMat.get(roi_cropped)
                roi_pts = hough_transform(roi_cropped)
                card = four_point_transform(roi_cropped, roi_pts)
            except:
                continue
            if STOP_ON_IMAGE_2c:
                cv2.imshow("card", card)
                cv2.waitKey(0)

            cards.append(card)
            positions.append(center)

    return cards, positions

def determine_symbol_from_card(card):
    #card = cv2.UMat.get(card)
    card_scaled = cv2.resize(card, (228, 113))
    card_scaled = card_scaled[5:103, 14:214]  # trim the TINIEST big off so there are less edges

    card_grey = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    card_grey = cv2.resize(card_grey, (228, 113))
    card_grey = card_grey[5:103, 14:214]  # trim the TINIEST big off so there are less edges

    kernel_size = 3

    card_thresh = cv2.GaussianBlur(card_grey, (3, 3), 0, 0);
    card_thresh = cv2.Laplacian(card_thresh, cv2.CV_8U, ksize=kernel_size)

    # apply morphs to clean it up
    close_amount = 5
    #roi_thresh = cv2.threshold(roi_thresh, 20, 255, cv2.THRESH_BINARY)[1]
    card_thresh = cv2.morphologyEx(card_thresh, cv2.MORPH_CLOSE, np.ones((close_amount, close_amount)))

    ## Match to squiggle shape. Made it red if it is
    template = cv2.imread('./templates/squig.jpg')
    template_grey = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_grey.shape[::-1]

    res = cv2.matchTemplate(card_thresh, template_grey, cv2.TM_CCOEFF_NORMED)
    threshold = 0.285
    threshold = 0.25
    loc = np.where(res >= threshold)
    card_template = card_scaled
    for pt in zip(*loc[::-1]): cv2.rectangle(card_template, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    squig_num,squig_votes = determine_num_from_loc(loc)

    ## Match to oval shape. Made it green if it is
    template = cv2.imread('./templates/oval.jpg')
    template_grey = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_grey.shape[::-1]

    res = cv2.matchTemplate(card_thresh, template_grey, cv2.TM_CCOEFF_NORMED)
    threshold = 0.3
    loc = np.where(res >= threshold)
    card_template = card_scaled
    for pt in zip(*loc[::-1]): cv2.rectangle(card_template, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    oval_num,oval_votes = determine_num_from_loc(loc)

    ## Match to diamond shape. Made it blue if it is
    template = cv2.imread('./templates/diamond.jpg')
    template_grey = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_grey.shape[::-1]

    res = cv2.matchTemplate(card_thresh, template_grey, cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    loc = np.where(res >= threshold)
    card_template = card_scaled
    for pt in zip(*loc[::-1]): cv2.rectangle(card_template, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    diam_num, diam_votes = determine_num_from_loc(loc)



    #determine which shape won and how many
    num = 1
    shape = "squig"
    if oval_votes > squig_votes and oval_votes > diam_votes:
        shape = "oval"
        num = oval_num
    if squig_votes > oval_votes and squig_votes > diam_votes:
        shape = "squig"
        num = squig_num
    if diam_votes > squig_votes and diam_votes > oval_votes:
        shape = "diam"
        num = diam_num




    if STOP_ON_IMAGE_4:
        # cv2.imshow("drawing", drawing)
        print("Shape: ", shape)
        print("Num: ", num + 1)
        cv2.imshow('roi_thresh', card_thresh)
        cv2.imshow('ref_template', card_template)
        cv2.waitKey(0)

    return shape,num


def render(ref_color,fname):
    start_time = time.time()
    cards, centers = segment_image_into_cards_hough(ref_color)
    board = []
    for i in range(len(cards)):
        card = cards[i]
        shape, num = determine_symbol_from_card(card)
        color, inter = determine_color_from_card(card,num)

        obj_card = Card(color_text = color, shape_text = shape,numb_text = num,inter_text = inter)
        obj_card.coord = centers[i]
        obj_card.pic = cards[i]

        #set color
        if color == "red": obj_card.color = 0
        if color == "green": obj_card.color = 1
        if color == "purp": obj_card.color = 2

        #set shape
        if shape == "diam": obj_card.shape = 0
        if shape == "squig": obj_card.shape = 1
        if shape == "oval": obj_card.shape = 2

        #set inter
        if inter == "blank": obj_card.inter = 0
        if inter == "striped": obj_card.inter = 1
        if inter == "solid": obj_card.inter = 2

        #set numb
        if num == 0: obj_card.numb = 0
        if num == 1: obj_card.numb = 1
        if num == 2: obj_card.numb = 2

        board.append(obj_card)

        if STOP_ON_IMAGE_6:
            cv2.imshow("card",card)
            print("Card is:")
            print("     ",shape)
            print("     ",str(num+1))
            print("     ",color)
            print("     ",inter)
            cv2.waitKey(0)

    if STOP_ON_IMAGE_7: #show initial image with labelling on top
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        lineType = int(2)
        x_offset = -100
        y_offset = 150

        ref_label = copy.deepcopy(ref_color)

        for card in board:
            text = str(card.numb + 1) + " " +  str(card.shape_text) + " " +  str(card.inter_text)

            # fontColor
            if card.color_text == "red": fontColor = (0, 0, 255)
            if card.color_text == "green": fontColor = (131, 170, 83)
            if card.color_text == "purp": fontColor = (133, 100, 112)

            cv2.putText(ref_label, text,
                        (int(card.coord[0]+x_offset),int(card.coord[1]+y_offset)),
                        font,
                        fontScale,
                        fontColor,
                        lineType)

        # store image shape
        scale_percent = 70  # percent of original size
        width = int(ref_label.shape[1] * scale_percent / 100)
        height = int(ref_label.shape[0] * scale_percent / 100)
        dim = (width, height)

        ref_label = cv2.resize(ref_label, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow("ref_label",ref_label)
        cv2.waitKey(0)

    sets = game.check_board(board)

    if STOP_ON_IMAGE_8:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        lineType = int(1.5)
        x_offset = -50
        y_offset = 35

        ref_final = copy.deepcopy(ref_color)

        for set in sets:

            c_0 = set[0]
            c_1 = set[1]
            c_2 = set[2]

            #color = get_color(s)
            color_0 = (int(c_0.coord[0] / 50) * 20 % 120 + 125)
            color_1 = (int(c_2.coord[0] / 50) * 20 % 120 + 125)
            color_2 = (int(c_1.coord[0] / 50) * 20 % 120 + 125)
            color = (color_0, color_1, color_2)

            cv2.line(ref_final, (int(c_0.coord[0]), int(c_0.coord[1])), (int(c_1.coord[0]), int(c_1.coord[1])), (0,0,0), 4)
            cv2.line(ref_final, (int(c_2.coord[0]), int(c_2.coord[1])), (int(c_0.coord[0]), int(c_0.coord[1])), (0,0,0), 4)
            cv2.line(ref_final, (int(c_1.coord[0]), int(c_1.coord[1])), (int(c_2.coord[0]), int(c_2.coord[1])), (0,0,0), 4)
            cv2.line(ref_final, (int(c_0.coord[0]), int(c_0.coord[1])), (int(c_1.coord[0]), int(c_1.coord[1])), color, 2)
            cv2.line(ref_final, (int(c_1.coord[0]), int(c_1.coord[1])), (int(c_2.coord[0]), int(c_2.coord[1])), color, 2)
            cv2.line(ref_final, (int(c_2.coord[0]), int(c_2.coord[1])), (int(c_0.coord[0]), int(c_0.coord[1])), color, 2)

        for card in board:
            text = str(card.numb + 1) + " " + str(card.shape_text) + " " + str(card.inter_text)

            # fontColor
            if card.color_text == "red": fontColor = (0, 0, 255)
            if card.color_text == "green": fontColor = (107, 168, 0)
            if card.color_text == "purp": fontColor = (214, 111, 150)

            cv2.putText(ref_final, text,
                        (int(card.coord[0] + x_offset), int(card.coord[1] + y_offset)),
                        font,
                        fontScale,
                        (0,0,0),
                        lineType * 3)

            cv2.putText(ref_final, text,
                        (int(card.coord[0] + x_offset), int(card.coord[1] + y_offset)),
                        font,
                        fontScale,
                        fontColor,
                        lineType)


        # calculate the 50 percent of original dimensions
        width = int(ref_final.shape[1] * 2)
        height = int(ref_final.shape[0] * 2)

        # dsize
        dsize = (width, height)

        # resize image
        ref_final = cv2.resize(ref_final, dsize)

        cv2.imshow("ref_final", ref_final)

        cv2.waitKey(1)

    if STOP_ON_IMAGE_9:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        lineType = int(2)
        x_offset = -100
        y_offset = 55

        ref_final = copy.deepcopy(ref_color)

        for card in board:
            text = str(card.numb + 1) + " " + str(card.shape_text) + " " + str(card.inter_text)

            # fontColor
            if card.color_text == "red": fontColor = (0, 0, 255)
            if card.color_text == "green": fontColor = (131, 170, 83)
            if card.color_text == "purp": fontColor = (133, 100, 112)

            cv2.putText(ref_final, text,
                        (int(card.coord[0] + x_offset), int(card.coord[1] + y_offset)),
                        font,
                        fontScale,
                        (0,0,0),
                        lineType * 3)

            cv2.putText(ref_final, text,
                        (int(card.coord[0] + x_offset), int(card.coord[1] + y_offset)),
                        font,
                        fontScale,
                        fontColor,
                        lineType)


        for set in sets:

            c_0 = set[0]
            c_1 = set[1]
            c_2 = set[2]

            #color = get_color(s)
            color_0 = (int(c_0.coord[0] / 50) * 20 % 120 + 125)
            color_1 = (int(c_2.coord[0] / 50) * 20 % 120 + 125)
            color_2 = (int(c_1.coord[0] / 50) * 20 % 120 + 125)
            color = (color_0, color_1, color_2)

            cv2.line(ref_final, (int(c_0.coord[0]), int(c_0.coord[1])), (int(c_1.coord[0]), int(c_1.coord[1])), (0,0,0), 6)
            cv2.line(ref_final, (int(c_2.coord[0]), int(c_2.coord[1])), (int(c_0.coord[0]), int(c_0.coord[1])), (0,0,0), 6)
            cv2.line(ref_final, (int(c_1.coord[0]), int(c_1.coord[1])), (int(c_2.coord[0]), int(c_2.coord[1])), (0,0,0), 6)
            cv2.line(ref_final, (int(c_0.coord[0]), int(c_0.coord[1])), (int(c_1.coord[0]), int(c_1.coord[1])), color, 3)
            cv2.line(ref_final, (int(c_1.coord[0]), int(c_1.coord[1])), (int(c_2.coord[0]), int(c_2.coord[1])), color, 3)
            cv2.line(ref_final, (int(c_2.coord[0]), int(c_2.coord[1])), (int(c_0.coord[0]), int(c_0.coord[1])), color, 3)

        # store image shape
        scale_percent = 70  # percent of original size
        width = int(ref_final.shape[1] * scale_percent / 100)
        height = int(ref_final.shape[0] * scale_percent / 100)
        dim = (width, height)

        ref_final = cv2.resize(ref_final, dim, interpolation=cv2.INTER_AREA)

        end_name = fname[11:]
        #tylerb to renable
        #cv2.imwrite("C:\set_movie_frames/"+end_name,ref_final)


    print("--- %s seconds ---" % (time.time() - start_time))
    #return ref_final

def main():

    if CAM_ENABLED:

        cap = cv2.VideoCapture(0)
        #cv2.namedWindow("test")
        i = 0
        #capture a frame and throw it away since the first frame is garbage anyways
        ret, frame = cap.read()
        time.sleep(1)
        show_ans = False
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off
        while True:
            ret, frame = cap.read()

            if not ret:
                print("failed to grab frame")
                break

            fname = "test"


            #cv2.imshow("test", frame)
            #cv2.waitKey(0)
            if not show_ans:
                # calculate the 50 percent of original dimensions
                width = int(frame.shape[1] * 2)
                height = int(frame.shape[0] * 2)

                # dsize
                dsize = (width, height)

                # resize image
                ref_final = cv2.resize(frame, dsize)

                cv2.imshow("ref_final", ref_final)
                if cv2.waitKey(1) == ord('s'):
                    show_ans = True
            else:
                render(frame, fname)

            i += 1
    else:
        directory = "C:/set_ocr"
        for filename in os.listdir(directory):
            if filename.endswith(".bmp"):
                try:
                    fname = "C:\set_ocr/" + filename
                    ref_color = cv2.imread(fname)
                    render(ref_color,fname)
                except:
                    img = cv2.imread("C:\set_ocr/" + filename)
                    #cv2.imshow("img",img)
                    #cv2.waitKey(0)

                    #end_name = filename[11:]
                    #print(filename)
                    cv2.imwrite("C:/set_movie_frames/" + filename, img)
            else:
                continue



#filename ="C:\set_ocr/f000557.bmp"
#render(filename)

main()