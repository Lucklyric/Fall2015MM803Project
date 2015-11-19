options.vid = webcam();
% Load the newly-trained detector
options.detectorfull = vision.CascadeObjectDetector('1256617233-1-haarcascade_hand.xml');
options.detectorpist = vision.CascadeObjectDetector('aGest.xml');
options.detectorFace = vision.CascadeObjectDetector('FrontalFaceCART');
% Test classifier on video
testClassifier(options)


