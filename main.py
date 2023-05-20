# %%
import os
import time
import numpy as np
import cv2 as cv2
from scipy import ndimage as ndi
from skimage import feature
import seaborn as sns
import joblib
import imghdr

def should_flip(img):
    a = img.sum(axis=0)
    a = a / a.max()
    if (a[:int(len(a)//8)].sum() < a[int(len(a)*7//8):].sum()):
        # flip the image
        return True
    return False


def segm_9(img):
    # Convert BGR to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # Split HSV image into separate channels
    _, _, B_channel = cv2.split(hsv_img)
    _, trr2 = cv2.threshold(
        B_channel, 1, 1.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    trr2 = ndi.binary_fill_holes(trr2).astype(np.uint8)

    contours, _ = cv2.findContours(
        trr2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    maxcontour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(maxcontour)
    crop = img[y:y+h, x:x+w]
    masked_data = cv2.bitwise_and(crop, crop, mask=trr2[y:y+h, x:x+w])
    masked_data = cv2.resize(masked_data, (128*2, 64*2))
    if should_flip(trr2):
        masked_data = cv2.flip(masked_data, 1)

    return cv2.cvtColor(masked_data, cv2.COLOR_BGR2GRAY)


def preprocess(img_rgb):
    img_rgb = cv2.resize(img_rgb, (461, 260))
    holesimg = segm_9(img_rgb)
    return holesimg


out_labels_file = r'results.txt'
out_time_file = r'time.txt'
path_to_data = "data"

clf = joblib.load(r'SVM-POLY.pkl')
clf_3_4 = joblib.load(r'SVM-POLY_2_3_4.pkl')
# path = "test_data/"
# labels_file = r'labels.txt'


def DoItAll():
    labels, features = [], []
    image_names = []
    times = []
    for imagePath in os.listdir(path_to_data):
        p = os.path.join(path_to_data, imagePath)
        if(imghdr.what(p) == None):
            continue
        time_start = time.time()
        try:
            img_rgb = cv2.imread(p)
            time_start = time.time()
            image = preprocess(img_rgb)
        except:
            # print("error in: " + p)
            times.append(time.time() - time_start)
            labels.append(0)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        (H) = feature.hog(image, orientations=9,  pixels_per_cell=(16, 16),
                          cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True, channel_axis=-1)  # ,visualize=True
        ll = clf.predict([H])[0]
        # print(ll, type(ll))

        if ll == '3' or ll == '4' or ll == '2':
            ll = clf_3_4.predict([H])[0]
        # print('label: ', ll, 'image: ', imagePath)
        time_end = time.time()
        times.append(time_end - time_start)

        labels.append(ll)
        image_names.append(imagePath)

    # sort labels array according to the names
    labels = [x for _, x in sorted(
        zip(image_names, labels), key=lambda pair: int(pair[0].split('.')[0]))]

    with open(out_labels_file, 'w') as f:
        for item in labels:
            f.write("%s\n" % item)

    with open(out_time_file, 'w') as f:
        for item in times:
            f.write("%.3f\n" % item)

# %%
if __name__ == "__main__":
    DoItAll()
