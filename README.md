# TapDanceDecoder 
## About: 
The goal of this project is to decode audio recordings of tap dancing into the (written) names of the observed steps. <br/>Simply said: Feet to text. 
## Motivation: 
Tap dance steps and choreography have traditionally been passed down (orally) directly from mentor to student, hence the lack of a standardized step nomenclature and archive of written choreography. However, we are left with valuable tap artifacts in the form of audio and video recordings. Even to the trained ear, it can be difficult, sometimes near impossible, to decipher exact steps based solely on audio due to variations in technique, shoes, floor, and recording equipment.   
## Goal:
Inspired by advances in speech recognition, music genre classification, and a lifelong passion for dance, this project aims to shake up the tap dance scene by demystifying these elaborate works of art with the help of machine learning. While this is a lofty goal, we will start simple with two of the most basic steps: shuffle and ball-change. Given an audio clip, our model will (hopefully) be able to predict which step has been recorded.
## Approach:
Unfortunately, while there is no shortage of tap dance sound effects, it is rare to find labelled audio segements of specific steps. Therefore, we will be collecting audio samples of the two steps (shuffle and ball-change) from tap dancing volunteers. These samples will make up our training, validation, and test sets. For further test samples, we will be using publicly available examples from YouTube (tap dancing tutorial) videos.

## Road Map:
- Data collection from volunteers, YouTube of two steps (2 classes)
- Feature extraction for input data (MFCC, Zero crossing rate, energy, tempo)
- Initial model: Logistic regression and Random Forest models with sklearn
- Improvements: Feedforward neural network with torch
- Maybe: Convolutional neural network to decode steps using raw signal data
- If I'm lucky, incorporate a direct "tap translation" over live microphone
