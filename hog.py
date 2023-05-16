# %%
# import os
# os.environ['KAGGLE_CONFIG_DIR'] = "/content/kaggle"
# !kaggle datasets download -d evernext10/hand-gesture-of-the-colombian-sign-language
# !unzip '/content/kaggle/hand-gesture-of-the-colombian-sign-language.zip' -d '/content/kaggle/dataset'
# !rm -rf /content/kaggle/dataset/men/A

# %%
# %pip install pyefd


# %%
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
from sklearn.model_selection import train_test_split,GridSearchCV  
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns;
import collections
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import skew
from skimage import segmentation
from skimage.filters import sobel
sns.set()
np.random.seed(42)



# %% [markdown]
# There is 0 csv file in the current version of the dataset:
# 

# %% [markdown]
# The next hidden code cells define functions for plotting data. Click on the "Code" button in the published kernel to reveal the hidden code.

# %%
def plt_t(title, img, cmap=None):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()

# %%

# %%capture
# Pre-processing of the images is done
base = './'
def segm_1(img_rgb):
  img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YCR_CB)  
  ycrcbmin = np.array((0, 133, 77))
  ycrcbmax = np.array((255, 173, 127))
  skin_ycrcb = cv2.inRange(img_ycrcb, ycrcbmin, ycrcbmax)
  kernel = np.ones((5, 5), np.uint8)
  img_erode = cv2.erode(skin_ycrcb, kernel, iterations=2)  
  holesimg = (ndi.binary_fill_holes(img_erode)*255).astype(np.uint8) 
  return holesimg

def segm_2(img_rgb):
  lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
  # bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
  _, _, r = cv2.split(lab)
  r = cv2.normalize(r, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
  elevation_map = sobel(r)
  markers = np.zeros_like(r)
  markers[r < 30] = 1
  markers[r > 150] = 2
  segmentation_coins = cv2.watershed(img_rgb, markers)
  return segmentation_coins


def segm_3(img_rgb):
  blur = cv2.GaussianBlur(img_rgb, (3, 3), 0)

  # Convert image to LAB color space
  lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
  # bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
  # lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
  lab[:, :, 0] = clahe.apply(lab[:, :, 0])

  # Convert image back to BGR color space
  bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
  # it was 50, 190
  canny = cv2.Canny(bgr, 10, 170)
  # opening = cv2.morphologyEx(canny, cv2.MORPH_OPEN, (3, 3))
  return canny

def is_lightened(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    quantiles = np.quantile(v, [0.5])
    # print(quantiles)
    return quantiles[0] > 230

def segm_4(img):
    # Convert image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Apply CLAHE to enhance contrast in LAB lightness channel
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(18,18))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert image back to BGR color space
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Display original and preprocessed images side by side
    # apply cannny edge detection
    edges = cv2.Canny(bgr, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 8))
    opening = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    from skimage import morphology
    holesimg = ndi.binary_fill_holes(opening).astype(np.int8)
    bin = holesimg>0.5
    a = morphology.remove_small_objects(bin, 1000)
    return (a*255).astype(np.uint8)

def segm_5(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mk = img_hsv > np.array([0, 0, 230])
    mk = mk.astype(np.float32)
    mask = (mk *0.5 + 0.5)
    masked = (mask* img_hsv)
    img_partly_darken = cv2.cvtColor(masked, cv2.COLOR_HSV2RGB)
    green_mask = (img_partly_darken[:, :, 0] > img_partly_darken[:, :, 1]).astype(np.float32)
    kernel = np.ones((5, 5), np.uint8)
    img_erode = cv2.erode(green_mask, kernel, iterations=2)  
    holesimg = (ndi.binary_fill_holes(img_erode)*255).astype(np.uint8) 
    return holesimg
  
def segm_6(img):
#     blur the img then turn it into gray scale
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    return gray



def segm_7(img):
    # Convert BGR to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # Split HSV image into separate channels
    _, _, B_channel = cv2.split(hsv_img)
    _, trr2 = cv2.threshold(
        B_channel, 1, 1.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    trr2 = ndi.binary_fill_holes(trr2).astype(np.uint8)
    masked_data = cv2.bitwise_and(img, img, mask=trr2)
    # after_color_segm = segm_1(masked_data)
    # kernel = np.ones((5, 5), np.uint8)
    # after_color_segm = cv2.morphologyEx((after_color_segm*255).astype(np.uint8), cv2.MORPH_OPEN, kernel,iterations=4)
    return cv2.cvtColor(masked_data, cv2.COLOR_BGR2GRAY)
  
def segm_8(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img_HSV = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))

    return global_mask


def segm_9(img):
    # Convert BGR to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # Split HSV image into separate channels
    _, _, B_channel = cv2.split(hsv_img)
    _, trr2 = cv2.threshold(
        B_channel, 1, 1.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    trr2 = ndi.binary_fill_holes(trr2).astype(np.uint8)
    
    contours ,_= cv2.findContours(trr2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    maxcontour = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(maxcontour)
    crop = img[y:y+h,x:x+w]
    masked_data = cv2.bitwise_and(crop, crop,mask = trr2[y:y+h,x:x+w])
    masked_data = cv2.resize(masked_data,(128*2,64*2))
    return cv2.cvtColor(masked_data, cv2.COLOR_BGR2GRAY)

def make_to_left(img):
  a = img.sum(axis=0)
  a = a / a.max()
  if(a[:int(len(a)//8)].sum() < a[int(len(a)*7//8):].sum()):
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
  holesimg = segm_9(img_rgb)
  holesimg = make_to_left(holesimg)
  return holesimg


def ImageSegmentation():
    path_IS = r"./Image-Segmentation"
    if not os.path.exists(path_IS):
        os.makedirs(path_IS)
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
                
                holesimg = preprocess(direc)
                if(holesimg is None):
                  continue
                
                imageio.imwrite(os.path.join(path_IS, name), holesimg)
                
ImageSegmentation()
# plt.show()
# img = cv2.imread(r"dataset\men\1\1_men (1).JPG")
# img_p = preprocess(r"dataset\men\1\1_men (2).JPG")
# plt_t('',img_p,cmap='gray')

# %%
# The Fourier elliptical features are extracted from each of the images and we proceed to save them in a. txt file.
base = "./"
def EF_oper(img_rgb):
    img_binary = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_binary, 127, 255, 0)
    contours, _ = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxcontour = max(contours, key=cv2.contourArea)
    # cv2.drawContours(img_rgb,[maxcontour],0,(255,9,9))
    # plt_t('cont',img_rgb)
    coeffs = []
    # Find the coefficients of all contours
    coeffs.append(elliptic_fourier_descriptors(np.squeeze(maxcontour), order=13))
    # print("coeff",coeffs)
    coeffs2 = []
    for row in coeffs:
        for elem in row:
            coeffs2.append(elem)
    coeffs = []
    for row in coeffs2:
        for elem in row:
            coeffs.append(elem)
    return np.array(coeffs)


def EllipticFourier():
    print("EF\n")
    path_EF = base + r"Feature-Extraction"
    if not os.path.exists(path_EF):
        os.makedirs(path_EF)

    file = open(base + r"Feature-Extraction/Elliptic-Fourier.txt", "w")
    # file = open(r"C:\Users\Ever\Desktop\Elliptic-Fourier.txt", "w")
    lstFiles = []  # nombre de imagenes
    path = r"./Image-Segmentation3"
    for (path, _, archivos) in walk(path):
        for arch in archivos:
            (nomArch, ext) = os.path.splitext(arch)
            if (ext == ".JPG"):
                lstFiles.append(nomArch + ext)
                direc = path + "/" + nomArch + ext
                name = nomArch + ext
                print(nomArch + ext)
                img_binary = cv2.imread(direc)
                coeffs = EF_oper(img_binary)
                file.write(name)
                for item in range(len(coeffs)):
                    file.write(",%.4f" % coeffs[item])
                file.write("," + name[0] + "\n")

    file.close()
EllipticFourier()
# img_binary = cv2.imread(r'Image-Segmentation2\0_men (4).JPG')
# coeffs = EF_oper(img_binary)
# %%
def HOG_oper(img_binary):
    (H) = feature.hog(img_binary, orientations=9, pixels_per_cell=(16,16),
                                  cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",channel_axis=-1) 
    pca = joblib.load(r"./Feature-Extraction/pca.pkl") #PCA(0.97).fit_transform(H.reshape(1, -1))
    components = pca.transform(H.reshape(1, -1))
    # joblib.dump(pca, r"./Feature-Extraction/pca.pkl")
    return components
from tqdm import tqdm
def HOG_PCA():
    data_HOG = pd.read_csv(r'./Feature-Extraction/Histogram-of-Oriented-Gradients.txt', sep=',', header=None)
    file2 = open(r"./Feature-Extraction/Histogram-of-Oriented-Gradients-PCA.txt", "w")
    name_HOG = data_HOG.iloc[:, 0]
    value_HOG = data_HOG.iloc[:, 1:-1]
    tag_HOG = data_HOG.iloc[:, -1] # 0,1,2,3,4,5
    print("PCA")
    pca = PCA(0.97).fit(value_HOG)
    joblib.dump(pca, r"./Feature-Extraction/pca.pkl")
    
    components = pca.transform(value_HOG)
    print(components.shape)
    for row in tqdm(range(len(components))):
        file2.write(name_HOG[row])
        for colm in range(len(components[row])):
            file2.write(",%.4f" %components[row][colm])
        file2.write(",%s" %tag_HOG[row] + "\n")
    file2.close()

def HOG():
    print("HOG\n")
    file  = open(r"./Feature-Extraction/Histogram-of-Oriented-Gradients.txt", "w")
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
                                  cells_per_block=(3, 3) ,block_norm='L2-Hys',feature_vector=True,channel_axis=-1)  # ,visualize=True
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
p5 = r'.\Image-Segmentation3\3_men (17).JPG'
img = cv2.imread(p5)
img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img_bw = cv2.threshold(img_bw, 127, 255, 0)
(H) = feature.hog(img_bw, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2, 2), 
                  block_norm="L1")
# plt_t('HOG',hi)

contours ,hier= cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
maxcontour = max(contours, key=cv2.contourArea)
cv2.drawContours(img,[maxcontour],0,(255,9,9))
plt_t('cont',img)
# crop the bounding rect of the contour
x,y,w,h = cv2.boundingRect(maxcontour)
crop = img[y:y+h,x:x+w]
plt_t('crop',crop)
crop = cv2.resize(crop,(128,64))
# rotate the crop
crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
plt_t('crop',crop)


# %%


# %%
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
base = "./"

#As Tecer method of classification we use Neural Networks
def TRY_classifiers(txt,test):

   

    data = pd.read_csv(base +'Feature-Extraction/'+txt+'.txt',sep=',',header=None)
    data=shuffle(data, random_state=0)

    s=data.shape# tamaño de dataframe
    col=[]
    #data.columns = ["a", "b", "c", "etc."]

    for x in range(0, s[1]):
        if x==0:
            col.append("NAME")
        elif x ==s[1]-1:
            col.append("TAG")
        else:
            col.append("VALOR-"+str(x))

    #se asigna el vector con los nombres de las columnas creado previamente y se las asignamos a la tabla
    data.columns = col

    ##print(data.groupby(['TAG'])['TAG'].count())
    vals_to_replace = { '0':'0', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5',
                         0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5'}

    data['TAG'] = data['TAG'].map(vals_to_replace)

    #print(data.tail())

    no_col=['NAME','TAG']
    #obtener todas las columnas
    Name_value = [x for x in col if x not in no_col]
    #se obtienen solo los coefficientes
    value=data[Name_value]

    tags=data[col[-1]] #columna de tags

    X_train, X_test, Y_train, Y_test = train_test_split(value,tags,test_size=test, random_state=0)

    # try different classifiers
    classfiers = [RandomForestClassifier(), GaussianNB(),svm.SVC(kernel='rbf'), svm.SVC(kernel='poly'),KNeighborsClassifier(),Pipeline(steps=[('scaler', StandardScaler()), ('SVC', SGDClassifier(loss="hinge", penalty="l2"))])]
    names = ["Random Forest", "Naive Bayes","SVM-RBF","SVM-POLY","KNN","SGDClassifier"]
    for clf in classfiers:
        clf.fit(X_train, Y_train)
        print("Classifier: "+str(names[classfiers.index(clf)]))
        print("Accuracy: "+str(clf.score(X_test,Y_test)))
        conf_mat = confusion_matrix(Y_test, clf.predict(X_test))
        plt.figure(figsize=(5,5))
        plt.title("Confusion matrix for "+str(names[classfiers.index(clf)])+" classifier    "+txt)
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap=plt.cm.copper)
    

# porcentaje_test=[0.30,0.25,0.20]
# NN("Elliptic-Fourier",porcentaje_test[1]
# TRY_classifiers("hinge",0.2)
TRY_classifiers("Histogram-of-Oriented-Gradients",0.2)
TRY_classifiers("Histogram-of-Oriented-Gradients-PCA",0.2)
# RandomForest("Elliptic-Fourier",0.2)
# RandomForest("pca",0.2)



# %%