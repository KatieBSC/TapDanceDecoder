# TapDanceDecoder 
## About: 
The goal of this project is to decode audio recordings of tap dancing into the (written) names of the observed steps. 
## Motivation: 
At the ripe age of 3, I began my love affair with tap dancing. In the decades that followed, interests in science, mathematics, and creative problem solving were awakened; however, my passion for tap never waned. Tap history and choreograpy, unlike the other aformentioned disciplines, has traditionally been passed down (orally) directly from mentor to student, hence the lack of a standardized step nomenclature and archive of written choreography. However, we are left with valuable tap artifacts in the form of audio and video recordings. Even to the trained ear, it can be difficult, sometimes near impossible, to decipher exact steps based solely on audio due to variations in technique, shoes, floor, and recording equipment.   
## Goal:
Inspired by advances in speech recognition, music genre classification, and morse code audio analysis, this project aims to shake up the tap dance scene by demystifying these elaborate works of art with the help of machine learning. While this is a lofty goal, we will start simple with two of the most basic steps: shuffle and ball-change. Given an audio clip, our model will (hopefully) be able to predict which step has been recorded.
## Approach:
Unfortunately, while there is no shortage of tap dance sound effects, it is rare to find labelled audio segements of specific steps. Therefore, we will be collecting audio samples of the two steps (shuffle and ball-change) from tap dancing volunteers. These samples will make up our training and validation sets. For the test samples, we will be using publicly available examples from YouTube videos.

## Initial Road Map:
- Data collection from volunteers, YouTube
- Feature extraction for input data (MFCC)
- Implement logistic regression for the classification of the two steps. 
- Construct LSTM (RNN) network to decode combinations (sequences) of variable length.
- Finally, there is an option to incorporate a direct "tap translation" with use of the computer's microphone. 
  <br/>(I am aware that this is a tall order for 4 weeks, but a girl's gotta dream!)
