#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>
#define DETECTHANDS 1

#ifdef _EiC
#define WIN32
#endif

//~~~~~~~~ Begin declaration for Camshift variables! Careful! These are Global (C!) ~~~~~~~~~~~//

IplImage *image = 0, *hsv = 0, *hue = 0, *mask = 0, *backproject = 0, *histimg = 0;
CvHistogram *hist = 0;
int flag_firstface = 0;
int backproject_mode = 0;
int select_object = 0;
int track_object = 0;
int show_hist = 1;
CvPoint origin;
CvRect selection, removal;
CvRect track_window;
CvBox2D track_box;
CvConnectedComp track_comp;
int hdims = 16;
float hranges_arr[] = { 0,180 };
float* hranges = hranges_arr;
int vmin = 10, vmax = 256, smin = 30;

CvRect track_window2;
CvBox2D track_box2;
CvConnectedComp track_comp2;

//~~~~~~~ END~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

static CvMemStorage* storage = 0;

static CvSeq* faces = 0;

static CvHaarClassifierCascade* cascade = 0;

const char* cascade_name = "haarcascade3.xml";

CvScalar hsv2rgb(float hue) {
	int rgb[3], p, sector;
	static const int sector_data[][3] = { { 0,2,1 },{ 1,2,0 },{ 1,0,2 },{ 2,0,1 },{ 2,1,0 },{ 0,1,2 } };
	hue *= 0.033333333333333333333333333333333f;
	sector = cvFloor(hue);
	p = cvRound(255 * (hue - sector));
	p ^= sector & 1 ? 255 : 0;

	rgb[sector_data[sector][0]] = 255;
	rgb[sector_data[sector][1]] = 0;
	rgb[sector_data[sector][2]] = p;

	return cvScalar(rgb[2], rgb[1], rgb[0], 0);
}

bool detectFaces(IplImage*);

void initializeCamVariables(IplImage*);

void trackHand(IplImage*);
//
//int main(int argc, char* argv[]) {
//	CvCapture* capture = 0;
//	IplImage *frame, *frame_copy = 0;
//	const char* input_name;
//	if (argc != 2) {
//		printf("No input video specified. Will use laptop camera\n");
//		capture = cvCaptureFromCAM(0);
//	}
//	else {
//		input_name = argv[1];
//		capture = cvCaptureFromAVI(input_name);
//		if (capture == NULL) {
//			printf("Cannot open specified input video, Exiting..");
//			exit(-1);
//		}
//	}
//
//	cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);
//
//	if (!cascade) {
//		fprintf(stderr, "ERROR: Could not load classifier cascade\n");
//		fprintf(stderr, "Usage: facedetect --cascade=\"<cascade_path>\" [filename|camera_index]\n");
//		return -1;
//	}
//
//	storage = cvCreateMemStorage(0);
//
//	cvNamedWindow("Histogram", 1);
//	cvNamedWindow("CamShiftDemo", 1);
//	//cvNamedWindow( "Temp", 1 );
//	//cvSetMouseCallback( "CamShiftDemo", on_mouse, 0 );
//
//	//cvCreateTrackbar( "Vmin", "CamShiftDemo", &vmin, 256, 0 );
//	//cvCreateTrackbar( "Vmax", "CamShiftDemo", &vmax, 256, 0 );
//	//cvCreateTrackbar( "Smin", "CamShiftDemo", &smin, 256, 0 );
//
//	for (;;) {
//		//Initialize loop. Keep grabbing frames till you detect a face
//		if (!cvGrabFrame(capture))
//			break;
//		frame = cvRetrieveFrame(capture);
//		if (!frame)
//			break;
//		if (!frame_copy)
//			frame_copy = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, frame->nChannels);
//
//		printf("LOG: Frame copy done\n");
//		if (frame->origin == IPL_ORIGIN_TL)
//			cvCopy(frame, frame_copy, 0);
//		else
//			cvFlip(frame, frame_copy, 0);
//
//		printf("LOG: Frame set right\n");
//		//A face is present in the frame.
//		//CvSeq* faces has the faces updated. Use faces -> count to get a count of number of faces in the frame.
//		//Then call Camshift tracker helper. 
//		//printf("Faces detected\n");
//
//		if (DETECTHANDS) {
//			printf("LOG: In detect hands\n");
//			if (flag_firstface < 1) {
//				detectFaces(frame_copy);
//				printf("LOG: First call to detect faces\n");
//
//				track_object = -1;
//				CvPoint centre;
//				double scale = 1.0;
//				//Lets get the co-ordinates of this face now.
//				CvRect* r = (CvRect*)cvGetSeqElem(faces, 0);
//				if (faces->total > 0) {
//					flag_firstface = 1;
//					printf("LOG: Face 0 recovered\n");
//					centre.x = cvRound((r->x + r->width*0.5)*scale);
//					centre.y = cvRound((r->y + r->height*0.5)*scale);
//					printf("LOG: Centre of face 0 being calculated\n");
//					selection = cvRect(r->x, r->y, 0, 0);
//					selection.x = centre.x;
//					selection.y = centre.y;
//					selection.width = r->width;
//					selection.height = r->height;
//					printf("LOG: Selection region set\n");
//				}
//			}
//			if (!image) {
//				initializeCamVariables(frame_copy);
//				printf("LOG: Camshift variables initialized\n");
//			}
//			cvCopy(frame_copy, image, 0);
//			cvCvtColor(image, hsv, CV_BGR2HSV);
//			printf("LOG: HSV copy of the current frame created\n");
//			if (track_object) {
//				if (flag_firstface > 0) {
//					trackHand(frame_copy);
//					printf("LOG: Hand tracking for one frame done\n");
//				}
//				else {
//					printf("LOG: No face detected. Skipping this frame...\n");
//				}
//			}
//
//			/*if( flag_firstface && selection.width > 0 && selection.height > 0 ) {
//			cvSetImageROI( image, selection );
//			cvXorS( image, cvScalarAll(255), image, 0 );
//			cvResetImageROI( image );
//			}*/
//			cvShowImage("CamShiftDemo", image);
//			cvShowImage("Histogram", histimg);
//		}
//
//
//		if (cvWaitKey(10) >= 0)
//			break;
//	}
//	cvReleaseImage(&frame);
//	cvReleaseImage(&frame_copy);
//	cvReleaseCapture(&capture);
//
//	cvDestroyWindow("CamShiftDemo");
//	cvDestroyWindow("Histogram");
//	//cvDestroyWindow("Temp");
//	printf("Done here!\n");
//	return 0;
//}

bool detectFaces(IplImage* img) {
	static CvScalar colors[] = {
		{ { 0,0,255 } },
		{ { 0,128,255 } },
		{ { 0,255,255 } },
		{ { 0,255,0 } },
		{ { 255,128,0 } },
		{ { 255,255,0 } },
		{ { 255,0,0 } },
		{ { 255,0,255 } }
	};

	double scale = 1.3;
	IplImage* gray = cvCreateImage(cvSize(img->width, img->height), 8, 1);
	IplImage* small_img = cvCreateImage(cvSize(cvRound(img->width / scale), cvRound(img->height / scale)), 8, 1);
	int i;
	cvCvtColor(img, gray, CV_BGR2GRAY);
	cvResize(gray, small_img, CV_INTER_LINEAR);
	cvEqualizeHist(small_img, small_img);
	cvClearMemStorage(storage);
	if (cascade) {
		double t = (double)cvGetTickCount();
		faces = cvHaarDetectObjects(small_img, cascade, storage, 1.1, 2, 0/*CV_HAAR_DO_CANNY_PRUNING*/, cvSize(30, 30));
		t = (double)cvGetTickCount() - t;
		//printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
		/*printf("The number of faces = %d\n", faces -> total);
		for( i = 0; i < (faces ? faces->total : 0); i++ ) {
		CvRect* r = (CvRect*)cvGetSeqElem( faces, i );
		CvPoint center;
		int radius;
		printf("The Co-ordinates of face (x, y) = (%d , %d) with width = %d, height = %d\n", r -> x, r -> y, r -> width, r -> height);
		center.x = cvRound((r->x + r->width*0.5)*scale);
		center.y = cvRound((r->y + r->height*0.5)*scale);
		radius = cvRound((r->width + r->height)*0.25*scale);
		cvCircle( img, center, radius, colors[i%8], 3, 8, 0 );
		}*/
		//cvShowImage( "result", img );
		cvReleaseImage(&gray);
		cvReleaseImage(&small_img);
	}
	else {
		printf("Error in detectFaces(): Could not load cascade, exiting\n");
		exit(-1);
	}
	if (faces->total > 0) {
		return true;
	}
	else
		return false;
}

bool trackHands(IplImage* img) {
	return true;
}

void initializeCamVariables(IplImage* frame) {
	/* allocate all the buffers */
	image = cvCreateImage(cvGetSize(frame), 8, 3);
	image->origin = frame->origin;
	hsv = cvCreateImage(cvGetSize(frame), 8, 3);
	hue = cvCreateImage(cvGetSize(frame), 8, 1);
	mask = cvCreateImage(cvGetSize(frame), 8, 1);
	backproject = cvCreateImage(cvGetSize(frame), 8, 1);
	hist = cvCreateHist(1, &hdims, CV_HIST_ARRAY, &hranges, 1);
	histimg = cvCreateImage(cvSize(320, 200), 8, 3);
	cvZero(histimg);
}

void trackHand(IplImage* frame) {
	int _vmin = vmin, _vmax = vmax;
	int i, bin_w, c;
	cvInRangeS(hsv, cvScalar(0, smin, MIN(_vmin, _vmax), 0), cvScalar(180, 256, MAX(_vmin, _vmax), 0), mask);

	cvSplit(hsv, hue, 0, 0, 0);

	if (track_object < 0) {
		float max_val = 0.f;
		cvSetImageROI(hue, selection);
		cvSetImageROI(mask, selection);
		cvCalcHist(&hue, hist, 0, mask);
		cvGetMinMaxHistValue(hist, 0, &max_val, 0, 0);
		cvConvertScale(hist->bins, hist->bins, max_val ? 255. / max_val : 0., 0);
		cvResetImageROI(hue);
		cvResetImageROI(mask);

		track_window = selection;
		track_window2 = cvRect(selection.x, selection.y, 0, 0);
		track_window2.x = selection.x + 40;
		track_window2.y = selection.y - 50;
		track_window2.width = selection.width;
		track_window2.height = selection.height;

		track_object = 1;
		cvZero(histimg);


		bin_w = histimg->width / hdims;
		for (i = 0; i < hdims; i++) {
			int val = cvRound(cvGetReal1D(hist->bins, i)*histimg->height / 255);
			CvScalar color = hsv2rgb(i*180.f / hdims);
			cvRectangle(histimg, cvPoint(i*bin_w, histimg->height), cvPoint((i + 1)*bin_w, histimg->height - val), color, -1, 8, 0);
		}
	}

	if (flag_firstface > 1) {
		detectFaces(frame);
		CvPoint centre;
		int k = 0;
		if (faces->total > 0) {
			//for(k = 0 ; k > faces -> total; k++) {	
			CvRect* r = (CvRect*)cvGetSeqElem(faces, k);

			centre.x = cvRound((r->x + r->width*0.5) * 1);
			centre.y = cvRound((r->y + r->height*0.5)*0.75);

			removal = cvRect(r->x, r->y, 0, 0);
			removal.x = centre.x;
			removal.y = centre.y;
			removal.width = r->width + 50;
			removal.height = r->height + 100;
			cvSetImageROI(mask, removal);
			cvAndS(mask, cvScalarAll(0), mask, 0);
			cvResetImageROI(mask);
			//cvShowImage( "Temp", mask );
			//trackHand(frame_copy);
			//}
		}
	}
	else {
		if (flag_firstface == 1)
			flag_firstface = 2;
	}
	cvCalcBackProject(&hue, backproject, hist);
	cvAnd(backproject, mask, backproject, 0);
	cvCamShift(backproject, track_window, cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1), &track_comp, &track_box);

	cvCamShift(backproject, track_window2, cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1), &track_comp2, &track_box2);
	track_window = track_comp.rect;
	track_window2 = track_comp2.rect;
	if (track_window2.x == track_window.x && track_window2.y == track_window.y) {
		track_window2.x = 0;
		track_window2.y = 0;
		printf("Both are tracking the same hand!!\n");
	}
	if (backproject_mode)
		cvCvtColor(backproject, image, CV_GRAY2BGR);
	if (!image->origin)
		track_box.angle = -track_box.angle;
	cvEllipseBox(image, track_box, CV_RGB(255, 0, 0), 3, CV_AA, 0);
	cvEllipseBox(image, track_box2, CV_RGB(25, 89, 9), 3, CV_AA, 0);

}
