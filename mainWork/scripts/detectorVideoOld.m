%% Setup
% Create objects for detecting faces, tracking points, acquiring and
% displaying video frames.
clear;
% Create the face detector object.
pistDetector = vision.CascadeObjectDetector('../trainedxml/fist.xml');



v = VideoReader('../testVideo/fistTest3-Lab.avi');
vOutput = VideoWriter('../testVideo/fistTest3-Lab-Old-Output.avi','Uncompressed AVI');
vOutput.FrameRate = 10;


% Initialize the tracker histogram using the Hue channel pixels from the
% nose.


videoFrame = readFrame(v);
frameSize = size(videoFrame);

% Create the video player object. 
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);

%% Detection and Tracking
% Capture and process video frames from the webcam in a loop to detect and
% track a face. The loop will run for 400 frames or until the video player
% window is closed.

runLoop = true;
numPts = 0;
frameCount = 0;
hasNose = false;
open(vOutput);
tic;
while hasFrame(v) && runLoop
    if frameCount > 0
            videoFrame = readFrame(v);
    end
    % Get the next frame.
    frameCount = frameCount+1;
    grayImage = rgb2gray(videoFrame);% Take green channel.
    
    
    handbbox = pistDetector.step(grayImage);

    videoFrame = insertObjectAnnotation(videoFrame, 'rectangle',handbbox, 'hand');
    % Display the annotated video frame using the video player object.
    step(videoPlayer, videoFrame);


    writeVideo(vOutput,videoFrame);

    % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);
end
executeTime = toc/frameCount;

% Clean up.
release(videoPlayer);
release(pistDetector);
close(vOutput);
