

#include <stdio.h>
#include <cv.h>
#include <highgui.h>

#define distance 7
#define CONTOUR_LENGTH 250

IplImage *image; 
CvMemStorage* storage = cvCreateMemStorage(0); 
CvSeq* contours;
CvBox2D contourcenter;
CvPoint armcenter;
CvMemStorage* palmstorage = cvCreateMemStorage(0);
CvMemStorage* fingerstorage = cvCreateMemStorage(0);
CvSeq* palm = cvCreateSeq(CV_SEQ_ELTYPE_POINT,sizeof(CvSeq),sizeof(CvPoint),palmstorage);
CvSeq* fingerseq = cvCreateSeq(CV_SEQ_ELTYPE_POINT,sizeof(CvSeq),sizeof(CvPoint),fingerstorage);
IplImage *output; 

void fingertip();
 
int main()
{
    char filename[]="xyed.jpg"; 
    
    image = cvLoadImage(filename,CV_LOAD_IMAGE_GRAYSCALE); 
    output = cvCreateImage(cvGetSize(image),IPL_DEPTH_8U,3);
    
    if(!image){
        printf("Error: Couldn't open the image file.\n");
    }
    
    cvThreshold(image, image, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    cvErode(image,image,0,1);
    cvDilate(image,image,0,1); 
    //cvShowImage("HelloWorld",image);
    cvFindContours( image, storage, &contours, sizeof(CvContour),CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cvPoint(0,0) ); 
    while(contours && contours->total <= CONTOUR_LENGTH){  
      contours = contours->h_next;                                                                
    } 
    cvDrawContours( output, contours, CV_RGB(100,100,100), CV_RGB(0,255,0), 1, 2, CV_AA, cvPoint(0,0) ); 
    
    if(contours) {
	  contourcenter =  cvMinAreaRect2(contours,0);
      armcenter.x = cvRound(contourcenter.center.x);
      armcenter.y = cvRound(contourcenter.center.y);
      cvCircle(output,armcenter,10,CV_RGB(255,255,255),-1,8,0);
      fingertip();                                                                                              
    }
    
    cvShowImage("HelloWorld",output); 
    cvWaitKey(0); 
    cvDestroyWindow("HelloWorld"); 
    cvReleaseImage(&image); 
    cvReleaseImage(&output);
    return EXIT_SUCCESS;
}

void fingertip()
{    
    int dotproduct,i;
	float length1,length2,angle,minangle,length;
	CvPoint vector1,vector2,min,minp1,minp2;
	CvPoint fingertip[20];
	CvPoint *p1,*p2,*p;
	int tiplocation[20];
	int count = 0;
	bool signal = false;

     for(i=0;i<contours->total;i++)
     {
         p1 = (CvPoint*)cvGetSeqElem(contours,i);
		 p = (CvPoint*)cvGetSeqElem(contours,(i+distance)%contours->total);
		 p2 = (CvPoint*)cvGetSeqElem(contours,(i+(2*distance))%contours->total);
		 vector1.x = p->x - p1->x;
		 vector1.y = p->y - p1->y;
		 vector2.x = p->x - p2->x;
		 vector2.y = p->y - p2->y;
		 dotproduct = (vector1.x*vector2.x) + (vector1.y*vector2.y); 
		 length1 = sqrtf((vector1.x*vector1.x)+(vector1.y*vector1.y));
		 length2 = sqrtf((vector2.x*vector2.x)+(vector2.y*vector2.y));
		 angle = fabs(dotproduct/(length1*length2));    
         

         if(angle < 0.2)
         {
            cvCircle(output,*p,4,CV_RGB(0,150,150),-1,8,0);
            
            if(!signal)
            {
				signal = true;
				min.x = p->x;
				min.y = p->y;
				minp1.x = p1->x;
				minp1.y = p1->y;
				minp2.x = p2->x;
				minp2.y = p2->y;
				minangle = angle;
			}      
			else
            {
				if(angle <= minangle)
                {
					
					min.x = p->x;
					min.y = p->y;
					minp1.x = p1->x;
					minp1.y = p1->y;
					minp2.x = p2->x;
					minp2.y = p2->y;
					minangle = angle;
				}
			}
						
	     }
	     
	     
	     else
	     {
             if(signal)
             {
                signal = false;
                CvPoint l1,l2,l3;
				l1.x = min.x - armcenter.x;
				l1.y = min.y - armcenter.y;
				
				l2.x = minp1.x - armcenter.x;
				l2.y = minp1.y - armcenter.y;

				l3.x = minp2.x - armcenter.x;
				l3.y = minp2.y - armcenter.y;
				
				length = sqrtf((l1.x*l1.x)+(l1.y*l1.y));
				length1 = sqrtf((l2.x*l2.x)+(l2.y*l2.y));
				length2 = sqrtf((l3.x*l3.x)+(l3.y*l3.y));    
                
                if(length > length1 && length > length2)
                {
					cvCircle(output,min,6,CV_RGB(255,0,255),-1,8,0);
					fingertip[count] = min;
				    tiplocation[count] = i+20;
				    count = count + 1;
				}
				
				else if(length < length1 && length < length2)
                {
					cvCircle(image,min,8,CV_RGB(255,0,0),-1,8,0);
					//cvCircle(virtualhand,min,8,CV_RGB(255,255,255),-1,8,0);
					cvSeqPush(palm,&min);
					//fingertip[count] = min;
				    //tiplocation[count] = i+20;
				    //count = count + 1;
				}
				
                      
             }
         }//else end
		
		
      }//for end	
       
	  for(i=0;i<count;i++)
      {
		if( (tiplocation[i] - tiplocation[i-1]) > 40)
        {
		  if( fingertip[i].x >= 630  || fingertip[i].y >= 470 )
		  {
             cvCircle(image,fingertip[i],6,CV_RGB(50,200,250),-1,8,0);                           
          }
          
          else
          {
          cvCircle(output,fingertip[i],6,CV_RGB(0,255,0),-1,8,0);
		//cvCircle(virtualhand,fingertip[i],6,CV_RGB(0,255,0),-1,8,0);
		//cvLine(virtualhand,fingertip[i],armcenter,CV_RGB(255,0,0),3,CV_AA,0);
		  cvSeqPush(fingerseq,&fingertip[i]);
          }
		}
	  }
	    //cvClearSeq(fingerseq);    
}
