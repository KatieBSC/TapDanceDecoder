# TapDanceDecoder 
## About: 
The goal of this project is to decode audio recordings of tap dancing into the (written) names of the observed steps. 
## Motivation (or why I learned to stop worrying and love the beat): 
At the ripe age of 3, I began my love affair with tap dancing. In the decades that followed, interests in science, mathematics, and creative problem solving were awakened; however, my passion for tap never waned. Tap history and choreograpy, unlike the other aformentioned disciplines, has traditionally been passed down (orally) directly from mentor to student, hence the lack of a standardized step nomenclature and archive of written choreography. However, we are left with valuable tap artifacts in the form of audio and video recordings. Even to the trained ear, it can be difficult, sometimes near impossible, to decipher exact steps based solely on audio due to variations in technique, shoes, floor, and recording equipment.   
## Goal:
Inspired by advances in speech recognition, music genre classification, and morse code audio analysis, this project aims to shake up the tap dance scene by demystifying these elaborate works of art with the help of machine learning. While this is a lofty goal, we will start simple with two of the most basic steps: shuffle and ball-change. Given an audio clip, our model will (hopefully) be able to predict which step has been recorded.
## Approach:
Unfortunately, while there is no shortage of tap dance sound effects, it is rare to find labelled audio segements of specific steps. Therefore, we will be using publicly available examples of the two steps (shuffle and ball-change) from YouTube videos. Combinations, or sequences, of the steps will be generated from the extracted observations for model training and testing. In addition, noise and tempo/speed changes will be incorporated in the datasets to increase complexity and mirror real-world challenges. 
## Initial Road Map:
- Data collection from YouTube
- Feature extraction for input data
- Implement logistic regression for the classification of the two steps. 
- Construct LSTM (RNN) network to decode combinations (sequences) of variable length.
- Finally, there is an option to incorporate a direct "tap translation" with use of the computer's microphone. 
  <br/>(I am aware that this is a tall order for 4 weeks, but a girl's gotta dream!)
