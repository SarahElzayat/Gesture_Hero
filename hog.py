# %%
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
# import cv2
import cv2 as cv2
# import cv2.cv2 as cv2

from scipy import ndimage as ndi
import imageio
from os import walk
from pyefd import elliptic_fourier_descriptors
from skimage import feature
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import skew
from skimage.filters import sobel
sns.set()
np.random.seed(42)


# %%
base = './'


def plt_t(title, img, cmap=None):
    plt.imshow(img, cmap=cmap)
    plt.title(title) 
    plt.show()

 
def segm_1(img_rgb):
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YCR_CB)
    ycrcbmin = np.array((0, 133, 77))
    ycrcbmax = np.array((255, 173, 127))
    skin_ycrcb = cv2.inRange(img_ycrcb, ycrcbmin, ycrcbmax)
    kernel = np.ones((5, 5), np.uint8)
    img_erode = cv2.erode(skin_ycrcb, kernel, iterations=2)
    holesimg = (ndi.binary_fill_holes(img_erode)*255).astype(np.uint8)
    return holesimg


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
    maske = trr2[y:y+h, x:x+w]
    masked_data = cv2.bitwise_and(crop, crop, mask=maske)
    masked_data = cv2.resize(masked_data, (128*2, 64*2))
    if should_flip(trr2):
        masked_data = cv2.flip(masked_data, 1)
        maske = cv2.flip(maske,1)
    return cv2.cvtColor(masked_data, cv2.COLOR_BGR2GRAY), maske*255


def make_to_left(img):
    a = img.sum(axis=0)
    a = a / a.max()
    if (a[:int(len(a)//8)].sum() < a[int(len(a)*7//8):].sum()):
        # flip the image
        return cv2.flip(img, 1)
    return img


def preprocess(direc):
    try:
        img_rgb = cv2.imread(direc)
        img_rgb = cv2.resize(img_rgb, (461, 260))
    except:
        print("cant read image")
        return None
    # if(is_lightened(img_rgb)):
    #   holesimg = segm_5(img_rgb)
    # else:
    #   holesimg = segm_1(img_rgb)

    # img_bin = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # holesimg =  cv2.bitwise_and(img_bin, holesimg)
    # holesimg = segm_1(img_rgb)
    # holesimg = segm_3(img_rgb)
    # holesimg = segm_4(img_rgb)
    holesimg, mask = segm_9(img_rgb)
    return holesimg, mask


def ImageSegmentation():
    path_IS = r"./Image-Segmentation"
    path_mask = r"./Image-Segmentation-masks"
    if not os.path.exists(path_IS):
        os.makedirs(path_IS)
    if not os.path.exists(path_mask):
        os.makedirs(path_mask)
    lstFiles = []  # nombre de imagenes
    path = r"./dataset"

    for (path, _, archivos) in walk(path):
        for arch in archivos:
            (nomArch, ext) = os.path.splitext(arch)
            if (ext == ".JPG"):
                lstFiles.append(nomArch + ext)
                direc = path + "/" + nomArch + ext
                name = nomArch + ext
                print(path + "/" + nomArch + ext)

                holesimg, mask = preprocess(direc)
                if (holesimg is None):
                    continue

                imageio.imwrite(os.path.join(path_IS, name), holesimg)
                imageio.imwrite(os.path.join(path_mask, name), mask)


ImageSegmentation()
# plt.show()
# img = cv2.imread(r"dataset\men\1\1_men (1).JPG")
# img_p = preprocess(r"dataset\men\1\1_men (2).JPG")
# plt_t('',img_p,cmap='gray')

# %%
# The Fourier elliptical features are extracted from each of the images and we proceed to save them in a. txt file.
base = "./"


def HOG_oper(img_binary):
    (H) = feature.hog(img_binary, orientations=9, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", channel_axis=-1)
    # PCA(0.97).fit_transform(H.reshape(1, -1))
    pca = joblib.load(r"./Feature-Extraction/pca.pkl")
    components = pca.transform(H.reshape(1, -1))
    # joblib.dump(pca, r"./Feature-Extraction/pca.pkl")
    return components


def HOG_PCA():
    data_HOG = pd.read_csv(
        r'./Feature-Extraction/Histogram-of-Oriented-Gradients.txt', sep=',', header=None)
    file2 = open(
        r"./Feature-Extraction/Histogram-of-Oriented-Gradients-PCA.txt", "w")
    name_HOG = data_HOG.iloc[:, 0]
    value_HOG = data_HOG.iloc[:, 1:-1]
    tag_HOG = data_HOG.iloc[:, -1]  # 0,1,2,3,4,5
    print("PCA")
    pca = PCA(0.97).fit(value_HOG)
    joblib.dump(pca, r"./Feature-Extraction/pca.pkl")

    components = pca.transform(value_HOG)
    print(components.shape)
    for row in tqdm(range(len(components))):
        file2.write(name_HOG[row])
        for colm in range(len(components[row])):
            file2.write(",%.4f" % components[row][colm])
        file2.write(",%s" % tag_HOG[row] + "\n")
    file2.close()


def HOG():
    print("HOG\n")
    file = open(r"./Feature-Extraction/Histogram-of-Oriented-Gradients.txt", "w")
    lstFiles = []  # nombre de imagenes
    path = r"./Image-Segmentation"
    for (path, _, archivos) in walk(path):
        for arch in archivos:
            (nomArch, ext) = os.path.splitext(arch)
            if (ext == ".JPG"):
                lstFiles.append(nomArch + ext)
                direc = path + "/" + nomArch + ext
                name = nomArch + ext
                # print(nomArch + ext)
                img_binary = cv2.imread(direc)

                (H) = feature.hog(img_binary, orientations=9,  pixels_per_cell=(16, 16),
                                  cells_per_block=(3, 3), block_norm='L2-Hys', feature_vector=True, channel_axis=-1)  # ,visualize=True
                # plt_t('hog', himg)
                # hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))     ,hogImage
                # hogImage = hogImage.astype("uint8")

                # plt.imshow("HOG Image", hogImage)
                file.write(name)
                for item in range(len(H)):
                    file.write(",%.3f" % H[item])
                file.write("," + name[0] + "\n")
    file.close()


HOG()
HOG_PCA()

# The Histogram of Oriented Gradients features are extracted from each of the images and we proceed to save them in a. txt file
# %%
p5 = r'.\Image-Segmentation\5_woman (91).JPG'
img = cv2.imread(p5)
img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img_bw = cv2.threshold(img_bw, 127, 255, 0)
(H) = feature.hog(img_bw, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                  block_norm="L1")
# plt_t('HOG',hi)

contours, hier = cv2.findContours(
    img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
maxcontour = max(contours, key=cv2.contourArea)
cv2.drawContours(img, [maxcontour], 0, (255, 9, 9))
plt_t('cont', img)
# crop the bounding rect of the contour
x, y, w, h = cv2.boundingRect(maxcontour)
crop = img[y:y+h, x:x+w]
plt_t('crop', crop)
crop = cv2.resize(crop, (128, 64))
# rotate the crop
crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
plt_t('crop', crop)


# %%


# %%

base = "./"

# As Tecer method of classification we use Neural Networks


def TRY_classifiers(txt, test):

    data = pd.read_csv(base + 'Feature-Extraction/' +
                       txt+'.txt', sep=',', header=None)
    data = shuffle(data, random_state=0)

    s = data.shape  # tamaño de dataframe
    col = []
    # data.columns = ["a", "b", "c", "etc."]

    for x in range(0, s[1]):
        if x == 0:
            col.append("NAME")
        elif x == s[1]-1:
            col.append("TAG")
        else:
            col.append("VALOR-"+str(x))

    # se asigna el vector con los nombres de las columnas creado previamente y se las asignamos a la tabla
    data.columns = col

    # print(data.groupby(['TAG'])['TAG'].count())
    vals_to_replace = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
                       0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}

    data['TAG'] = data['TAG'].map(vals_to_replace)

    # print(data.tail())

    no_col = ['NAME', 'TAG']
    # obtener todas las columnas
    Name_value = [x for x in col if x not in no_col]
    # se obtienen solo los coefficientes
    value = data[Name_value]

    tags = data[col[-1]]  # columna de tags

    X_train, X_test, Y_train, Y_test = train_test_split(
        value, tags, test_size=test, random_state=0)

    print('dim' + str(X_train.shape))
    # try different classifiers
    classfiers = [RandomForestClassifier(), GaussianNB(), svm.SVC(kernel='rbf'), svm.SVC(kernel='poly'), KNeighborsClassifier(
    ), Pipeline(steps=[('scaler', StandardScaler()), ('SVC', SGDClassifier(loss="hinge", penalty="l2"))])]
    names = ["Random Forest", "Naive Bayes",
             "SVM-RBF", "SVM-POLY", "KNN", "SGDClassifier"]
    for i, clf in enumerate(classfiers):
        clf.fit(X_train.values, Y_train.values)
        joblib.dump(clf, base + names[i] + '.pkl')
        print("Classifier: "+str(names[i]))
        print("Accuracy: "+str(clf.score(X_test.values, Y_test.values)))
        conf_mat = confusion_matrix(Y_test, clf.predict(X_test.values))
        plt.figure(figsize=(5, 5))
        plt.title("Confusion matrix for "+str(names[i])+" classifier    "+txt)
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap=plt.cm.copper)


# porcentaje_test=[0.30,0.25,0.20]
# NN("Elliptic-Fourier",porcentaje_test[1]
# TRY_classifiers("hinge",0.2)
TRY_classifiers("Histogram-of-Oriented-Gradients", 0.2)
# TRY_classifiers("Elliptic-Fourier", 0.2)
# TRY_classifiers("Histogram-of-Oriented-Gradients-PCA",0.2)
# RandomForest("Elliptic-Fourier",0.2)
# RandomForest("pca",0.2)
