% Create the face detector object.
faceDetector = vision.CascadeObjectDetector();

% Create pist detector object.
pistDetector = vision.CascadeObjectDetector('falm.xml');

% Create the point tracker object.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Create the webcam object.
cam = webcam();

% Capture one frame to get its size.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

% Create the video player object.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);


runLoop = true;
numPts = 0;
frameCount = 0;

while runLoop 

    % Get the next frame.
    videoFrame = snapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;
    handbbox = pistDetector.step(videoFrameGray);
    videoFrame = insertObjectAnnotation(videoFrame, 'rectangle',handbbox, 'hand');


    if numPts < 10
        % Detection mode.
        

        if ~isempty(handbbox)
            % Find corner points inside the detected region.
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', handbbox(1, :));

            % Re-initialize the point tracker.
            xyPoints = round(points.Location);
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);

            % Save a copy of the points.
            oldPoints = xyPoints;

            % Convert the rectangle represented as [x, y, w, h] into an
            % M-by-2 matrix of [x,y] coordinates of the four corners. This
            % is needed to be able to transform the bounding box to display
            % the orientation of the face.
            bboxPoints = bbox2points(handbbox(1, :));

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Display a bounding box around the detected face.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Display detected corners.
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end

    else
%         % Tracking mode.
%         [xyPoints, isFound] = step(pointTracker, videoFrameGray);
%         visiblePoints = xyPoints(isFound, :);
%         oldInliers = oldPoints(isFound, :);
% 
%         numPts = size(visiblePoints, 1);
% 
%         if numPts >= 10
%             % Estimate the geometric transformation between the old points
%             % and the new points.
%             [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
%                 oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
% 
%             % Apply the transformation to the bounding box.
%             bboxPoints = transformPointsForward(xform, bboxPoints);
% 
%             % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
%             % format required by insertShape.
%             bboxPolygon = reshape(bboxPoints', 1, []);
% 
%             % Display a bounding box around the face being tracked.
%             videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
% 
%             % Display tracked points.
%             videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');
% 
%             
%             % Update ROI points with new bouding box
%             newxyPoints = [];
%             if ~isempty(handbbox)
%                 [numOfBboxes,~] = size(handbbox);
%                 points = {};
%                 for i=1:numOfBboxes
%                     if isempty(points)
%                         points = detectMinEigenFeatures(videoFrameGray, 'ROI', handbbox(i, :));
%                         
%                     else
%                         points = vertcat(points,detectMinEigenFeatures(videoFrameGray, 'ROI', handbbox(i, :)));
%                         
%                     end
%                 end
%                 if ~isempty(points)
%                     Blob = inpolygon(points.Location(:,1),points.Location(:,2),bboxPoints(:,1),bboxPoints(:,2));
%                     newPoints = points(find(Blob==1),:);
%                     newxyPoints = round(newPoints.Location);
%                 end
%             end
%             
%             % Reset the points.
%             oldPoints = [visiblePoints;newxyPoints];
%             oldPoints = unique(  round(oldPoints), 'rows');
%             setPoints(pointTracker, oldPoints);
%         end

    end

    % Display the annotated video frame using the video player object.
    step(videoPlayer, videoFrame);

    % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);
end

% Clean up.
clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);