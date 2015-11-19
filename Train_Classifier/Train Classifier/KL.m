% Create the face detector object.
faceDetector = vision.CascadeObjectDetector();
detectorpist = vision.CascadeObjectDetector();
% Create the point tracker object.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
% Create a tracker object.
shiftTracker = vision.HistogramBasedTracker;
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
    %[videoFrameHue,~,~] = rgb2hsv(videoFrame);
    videoFrameHue = videoFrameGray;
    frameCount = frameCount + 1;
    handbboxes = detectorpist.step(videoFrame);
    facebboxes = detectorpist.step(videoFrame);
    [numOfBboxes,~] = size(handbboxes);

    % Extract the skin area

    if numPts < 10
       % Detection mode.
       %bboxes = detectorpist.step(videoFrame);
       %bbox = round(getPosition(imrect));
        if ~isempty(handbboxes)
            % Find corner points inside the detected region.
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', handbboxes(1, :));
            initializeObject(shiftTracker, videoFrameHue, handbboxes(1, :));
            % Re-initialize the point tracker.
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);

            % Save a copy of the points.
            oldPoints = xyPoints;

            % Convert the rectangle represented as [x, y, w, h] into an
            % M-by-2 matrix of [x,y] coordinates of the four corners. This
            % is needed to be able to transform the bounding box to display
            % the orientation of the face.
            bboxPoints = bbox2points(handbboxes(1, :));

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Display a bounding box around the detected face.
            %videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            videoFrame =  insertObjectAnnotation(videoFrame, 'rectangle', handbboxes(1,:), 'New hand');

            % Display detected corners.
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end

    else
        
        % Tracking mode.
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);
        
        numPts = size(visiblePoints, 1);
        
        if numPts >= 10
            
            % CAMShift
            % Track using the Hue channel data
%             shiftbbox = step(shiftTracker, videoFrameHue);
%             
%             % Insert a bounding box around the object being tracked
%             videoFrame = insertObjectAnnotation(videoFrame,'rectangle',shiftbbox,'CAMShift');
            % Estimate the geometric transformation between the old points
            % and the new points.
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 3);
            
            % Apply the transformation to the bounding box.
            bboxPoints = transformPointsForward(xform, bboxPoints);

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);
            newBbox = [bboxPolygon(1),bboxPolygon(2),bboxPolygon(3)-bboxPolygon(1),bboxPolygon(4)-bboxPolygon(3)];

            % Display a bounding box around the face being tracked.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            videoFrame =  insertObjectAnnotation(videoFrame, 'rectangle',handbboxes, 'hand');
            % Display tracked points.
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');
            
            newxyPoints = [];
%             if isempty(points)
%                 points= {};
%             end
%             if ~isempty(handbboxes)
%                 % Find corner points inside the detected region.
%                 for i=1:numOfBboxes
%                     points = vertcat(points,detectMinEigenFeatures(videoFrameGray, 'ROI', handbboxes(i, :)));
%                 end
%                 
%                 if ~isempty(points)
%                     % Re-initialize the point tracker.
%                     Blob = inpolygon(points.Location(:,1),points.Location(:,2),bboxPoints(:,1),bboxPoints(:,2));
%                     newPoints = points(find(Blob==1),:);
%                 end
%                 
%                 if ~isempty(newPoints)
%                    % disp(length(newPoints));
%                     tmpxyPoints = newPoints.Location;
%                     newxyPoints = zeros(0,2);
% %                     for i=1:length(tmpxyPoints)
% %                         close = 0;
% %                         for j=1:length(visiblePoints)
% %                             pDist = pdist([tmpxyPoints(i,:);visiblePoints(j,:)],'euclidean');
% %                             if pDist < 3
% %                                 close = 1;
% %                                 break;
% %                             end
% %                         end
% %                         if close == 0
% %                             newxyPoints(end+1,:) = tmpxyPoints(i,:);
% %                         end
% %                     end
%                 end
%                     
%             end
            % Reset the points.
            %oldPoints = [visiblePoints;newxyPoints];
            %oldPoints = unique(  round(oldPoints), 'rows');
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end
        
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