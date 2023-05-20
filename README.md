
<h1 align="center" id="title">Gesture Hero</h1>

![1684570250027](image/README/1684570250027.png)

<p id="description">Gesture Hero is a hand gesture classification system that's build to differentiate between hand gestures representing numbers from 0-5.</p>

<h2>🛠️ Installation Steps:</h2>

<p>1. Installing required packages</p>

```
pip install -r requirements.txt
```

<p>2. To run for a specific dataset</p>

```
python ./main.py
```

<h2>💻 Built with</h2>

Technologies used in the project:


<div align="center">
	<code><img height="50" src="https://user-images.githubusercontent.com/25181517/183914128-3fc88b4a-4ac1-40e6-9443-9a30182379b7.png" alt="Jupyter Notebook" title="Jupyter Notebook"/></code>
	<code><img height="50" src="https://user-images.githubusercontent.com/25181517/183423507-c056a6f9-1ba8-4312-a350-19bcbc5a8697.png" alt="Python" title="Python"/></code>
	<code><img height="50" src="https://opencv.org/wp-content/uploads/2020/07/OpenCV_logo_black-2.png" alt="OpenCV" title="OpenCV"/></code>
	<code><img height="50" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png" alt="Scikit" title="sklearn"/></code>
</div>

* Python
* scikit-learn
* OpenCV
* skimage
  span

<h2>📝 Project description</h2>
<p> Gesture Hero is a hand gesture classification system that's build to differentiate between hand gestures representing numbers from 0-5. Gesture Hero is a machine learning based tool, trianed on a dataset with almost 2000 pictures, using an excellent preprocessing, HOG features descriptor and SVM for an accurate answer to your problem.

<h2> 🧩 Project pipleline

![1684571380892](image/README/1684571380892.png)

<h2> 🎞️ Preprocessing </h2>

<p> Gesture Hero's strongest point is its preprocessing. By applying classical image processing techniques to preprocess the data for removing shadows, enhancing colors, clipping the area surrounding the hand, rotating so all hands would be pointing in the same direction and resizing for efficiency. </p>
<p>The main preprocessing approach taking is ignoring/ eliminating the channels with misleading information, mainly the ones that represent the illumination.
<p>Segmentation is done using the YCRB channels using a basic thresholding and ignoring the Y(illumination channel), then cropping the photo to only contain the hand by finding the maximum contours surrounding the hand. 
<p> To unify the orientations of hands, the preprocessed image is passed to a function to flip it so that the fingers are pointing to the left, based on the histogram of the image, the more dense half represents the palm while the other represents the fingers.

| Raw image                                      | Preprocessed                                   |
| ---------------------------------------------- | ---------------------------------------------- |
| ![1684569675443](image/README/1684569675443.png) | ![1684569885945](image/README/1684569885945.png) |
| ![1684569753377](image/README/1684569753377.png) | ![1684569906290](image/README/1684569906290.png) |
| ![1684570066053](image/README/1684570066053.png) | ![1684570109112](image/README/1684570109112.png) |

<h2>🪶 Feature extraction </h2>

<p> The main feature descriptor used is the Histogram of Gradients (HOG), since it’s robust to variations in appearance, computationally efficient, its discriminative power as it’s very efficient in capturing the distinguishing features of an object, and lastly its compatibility with machine learning algorithms.

| Preprocessed image                             | Visualized HOG                                 |
| ---------------------------------------------- | ---------------------------------------------- |
| ![1684570608379](image/README/1684570608379.png) | ![1684570558000](image/README/1684570558000.png) |
| ![1684570563967](image/README/1684570563967.png) | ![1684570580475](image/README/1684570580475.png) |

<h2>💪🏻 Model training and performance analysis </h2>

<p> The chosen model is a support vector machine (SVM) with an RBF kernel, trianed on the HOG extracted features.

<p> Resulting confusion matrix is as follows

![1684570926835](image/README/1684570926835.png)


<p> Noticing that there's a great confusion between 2,3 and 4 gestures, a 2 layer classification method is used to improve the performance.

<p> The fisrt layer is an SVM model trained on the whole dataset, if the result label is a 2, 3 or 4, the image is then passed to another SVM model that's only trained on the 2,3,4 dataset. This has significantly improved the accuracy of the classificatin.

![1684571178913](image/README/1684571178913.png)

<h2> 👥 Collaborators </h2>

| [Sarah Elzayat](https://github.com/SarahElzayat) | [Ahmed ata abdallah](https://github.com/Ahmed-ata112) | **[Doaa Magdy](https://github.com/doaamagdy2024)** | **[Rufaida Kassem](https://github.com/Rufaida-Kassem)** |
| --------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------- |
