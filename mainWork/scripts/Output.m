%% Setup
% Create objects for detecting faces, tracking points, acquiring and
% displaying video frames.
clear;
% Create the face detector object.
pistDetector = vision.CascadeObjectDetector('../trainedxml/fist.xml');



v = VideoReader('../testVideo/Demo/fistTest3-Lab-Output.avi');
v2 = VideoReader('../testVideo/Demo/fistTest1-Skin-Output.avi');

vOutput = VideoWriter('../testVideo/Demo/Demo-New.avi','Uncompressed AVI');
vOutput.FrameRate = 10;


% Initialize the tracker histogram using the Hue channel pixels from the
% nose.


videoFrame = readFrame(v);
videoFrame2 = readFrame(v2);
frameSize = size(videoFrame);

% Create the video player object. 
videoPlayer = vision.VideoPlayer('Position', [100 100 [2*frameSize(2), frameSize(1)]+30]);
vOutput.FrameRate = 10;

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
            videoFrame2 = readFrame(v2);
    end
    % Get the next frame.
    frameCount = frameCount+1;
    newFrame = [videoFrame,videoFrame2];
    % Display the annotated video frame using the video player object.
    step(videoPlayer, newFrame);


    writeVideo(vOutput,newFrame);

    % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);
end
executeTime = toc/frameCount;

% Clean up.
release(videoPlayer);
release(pistDetector);
close(vOutput);
