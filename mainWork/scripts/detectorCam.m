%% Setup
% Create objects for detecting faces, tracking points, acquiring and
% displaying video frames.
clear;
% Create the face detector object.
faceDetector = vision.CascadeObjectDetector();
pistDetector = vision.CascadeObjectDetector('../trainedxml/fist.xml');
noseDetector = vision.CascadeObjectDetector('Nose', 'UseROI', true);
%noseBBox     = step(noseDetector, videoFrame, bbox(1,:));

% Create the point tracker object.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

v = VideoWriter('../testVideo/fistTest3-Lab.avi','Uncompressed AVI');
% vOutput = VideoWriter('../testVideo/fistTest2-Lab-Output.avi');
% Initialize the tracker histogram using the Hue channel pixels from the
% nose.

% Create the webcam object.
cam = webcam();

% Capture one frame to get its size.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

% Create the video player object. 
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);
videoPlayerFilter = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);

%% Detection and Tracking
% Capture and process video frames from the webcam in a loop to detect and
% track a face. The loop will run for 400 frames or until the video player
% window is closed.

runLoop = true;
numPts = 0;
frameCount = 0;
hasNose = false;
open(v);
%open(vOutput);
while runLoop 
    
    % Get the next frame.
    videoFrame = snapshot(cam);
    writeVideo(v,videoFrame);
    grayImage = videoFrame(:, :, 2); % Take green channel.
    [hueChannel,~,~] = rgb2hsv(videoFrame);
    %binaryImage = grayImage < 128;
    %testImage = grayImage.*uint8(binaryImage);
    testImage = grayImage;
    videoFrameGray = testImage;
    frameCount = frameCount + 1;
    if ~hasNose
        facebbox = faceDetector.step(videoFrameGray);
        if ~isempty(facebbox)
            nosebbox = noseDetector.step(videoFrameGray,facebbox(1,:));
            if ~isempty(nosebbox)
                [fh hw] = size(nosebbox);
                x = nosebbox(1,1);
                y = nosebbox(1,2);
                bw = nosebbox(1,3);
                noseG = hueChannel(y:y+bw,x:x+bw,:);
                meanG = mean(mean(noseG));
                videoFrame = insertObjectAnnotation(videoFrame, 'rectangle',nosebbox, 'nose');
                hasNose = true;
            end
        end
    else
        binaryImage = hueChannel < meanG+1*meanG & hueChannel > meanG-1*meanG;
        %binaryImage = medfilt2(binaryImage,[10,10]);
        I = gpuArray(binaryImage);
        K = gather(medfilt2(I,[7,7]));
        testImage = grayImage.*uint8(K);
    end
    
    handbbox = pistDetector.step(testImage);
   % videoFrame = insertObjectAnnotation(videoFrame, 'rectangle',handbbox, 'hand');
    
    if ~hasNose 
        step(videoPlayer, videoFrame);
        step(videoPlayerFilter,testImage);

    % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);
        continue;
    end
    if numPts < 10
        % Detection mode.
        bbox = pistDetector.step(testImage);
        if ~isempty(bbox)

            % Find corner points inside the detected region.
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));
            
            % Re-initialize the point tracker.
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            if numPts > 0
                initialize(pointTracker, xyPoints, videoFrameGray);
            end 
            % Save a copy of the points.
            oldPoints = xyPoints;
            
            % Convert the rectangle represented as [x, y, w, h] into an
            % M-by-2 matrix of [x,y] coordinates of the four corners. This
            % is needed to be able to transform the bounding box to display
            % the orientation of the face.
            bboxPoints = bbox2points(bbox(1, :));  
            
            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4] 
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);
            
            % Display a bounding box around the detected face.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            
            % Display detected corners.
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end
        
    else
        
        
        
        
        % Tracking mode.
        % Insert a bounding box around the object being tracked
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);
                
        numPts = size(visiblePoints, 1);       
        
        if numPts >= 10
            % Estimate the geometric transformation between the old points
            % and the new points.
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);            
            
            % Apply the transformation to the bounding box.
            bboxPoints = transformPointsForward(xform, bboxPoints);
            
            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4] 
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);            
            
            % Display a bounding box around the face being tracked.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            
            % Display tracked points.
            %videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');
            
            newxyPoints = [];
            if ~isempty(handbbox)
                [numOfBboxes,~] = size(handbbox);
                points = {};
                for i=1:numOfBboxes
                    if isempty(points)
                        points = detectMinEigenFeatures(testImage, 'ROI', handbbox(i, :));
                        
                    else
                        points = vertcat(points,detectMinEigenFeatures(videoFrameGray, 'ROI', handbbox(i, :)));
                        
                    end
                end
                if ~isempty(points)
                    Blob = inpolygon(points.Location(:,1),points.Location(:,2),bboxPoints(:,1),bboxPoints(:,2));
                    newPoints = points(find(Blob==1),:);
                    newxyPoints = round(newPoints.Location);
                end
            end
            
            
            % Reset the points.
            oldPoints = [visiblePoints;newxyPoints];
            oldPoints = unique(  round(oldPoints), 'rows');
            setPoints(pointTracker, oldPoints);
        end
    end
        
    % Display the annotated video frame using the video player object.
    step(videoPlayer, videoFrame);
    step(videoPlayerFilter,testImage);
    % writeVideo(vOutput,videoFrame);
    % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);
end

% Clean up.
clear cam;
close(v);
%close(vOutput);
release(videoPlayer);
release(videoPlayerFilter);

release(pointTracker);
release(faceDetector);
