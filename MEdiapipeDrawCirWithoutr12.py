'''
- This is the main program used in the conference paper entitled:
"Deep Learning-Based Real-Time Hand Landmark Recognition with MediaPipe for R12 Robot Control".
'''

import mediapipe as mp
import cv2
import numpy as np
import math
import json
from time import sleep
import __future__
import os
import shell
import arm
import time

# Latest working version of the drawing circle with 
# mediqpipe hand landmark 
#Authors. Ghoggali N. and Bensaadalah Moh.
#25/07/2023

global Robotconnected 
global i
Robotconnected = True
i=1
'''
r12shell = shell.ArmShell(arm.Arm())
#r12shell.cmdloop( ['connect','ROBOFORTH','START', 'CALIBRATE', 'READY','CARTESIAN'])
#r12shell.cmdloop( ['connect','ROBOFORTH','START', 'CALIBRATE','HOME','CARTESIAN'])
r12shell.cmdloop( ['connect','ROBOFORTH','START','HOME','CARTESIAN'])
time.sleep(2)
#r12shell.cmdloop( ['TELL GRIP WITHDRAW'])
#comdcircle = ['20000 SPEED !\n']
#r12shell.cmdloop(comdcircle)   
time.sleep(1)
'''

def SetZaxis():
   r12shell.cmdloop( ['READY'])  
   #comdcircle = [str(int(50))+' '+str(int(0))+' '+str(0)+' MOVE\n']
   #r12shell.cmdloop(comdcircle)  
  
   comdcircle = [str(int(0))+' '+str(int(0))+' '+str(-1445)+' MOVE\n']
   r12shell.cmdloop(comdcircle)  

def InitRect():
   r12shell.cmdloop( ['READY'])  
   comdcircle = [str(int(-500))+' '+str(int(0))+' '+str(0)+' MOVE\n']
   r12shell.cmdloop(comdcircle)  
   
   comdcircle = [str(int(0))+' '+str(int(2000))+' '+str(0)+' MOVE\n']
   r12shell.cmdloop(comdcircle) 
   
   #comdcircle = [str(int(0))+' '+str(int(0))+' '+str(-1445)+' MOVE\n']
   #r12shell.cmdloop(comdcircle)

   comdcircle = [str(int(0))+' '+str(int(0))+' '+str(-1445)+' MOVE\n']
   r12shell.cmdloop(comdcircle)  
   
def DrawLine(xii, yii, x, y):
   r12shell.cmdloop( ['READY'])
   
   comdcircle = [str(int(xii))+' '+str(int(0))+' '+str(0)+' MOVE\n']
   r12shell.cmdloop(comdcircle) 
   
   comdcircle = [str(int(0))+' '+str(int(yii))+' '+str(0)+' MOVE\n']
   r12shell.cmdloop(comdcircle) 
   
   comdcircle = [str(int(0))+' '+str(int(0))+' '+str(-1445)+' MOVE\n']
   r12shell.cmdloop(comdcircle) 
   
   comdcircle = [str(int(x))+' '+str(int(y))+' '+str(0)+' MOVE\n']
   r12shell.cmdloop(comdcircle) 
   
   r12shell.cmdloop( ['HOME'])


def RunCmmd(comdcircle):
    r12shell.cmdloop([comdcircle])
    #coord=r12shell.cmdloop(['WHERE'])

def InitCircle(radius): # Set up the parameters for drawing a circle
   center = [0.0, 0.0, 0.0]  # Center coordinates of the circle
   #radius = 5.0  # Radius of the circle
   #speed = 50  # Speed of the robot's movement (units depend on the robot's configuration)

   # Calculate the number of points to approximate the circle
   #num_points = 360  # You can adjust this for a smoother or more segmented circle
   num_points = 72  # You can adjust this for a smoother or more segmented circle

   # Calculate the angle increment between each point
   angle_increment = 360.0 / num_points

   # Move to the starting position
   #robot.move_to(center[0] + radius, center[1], center[2], center[3], center[4])

   # Move in a circular path to draw the circle
   SetZaxis()   
   for i in range(num_points):
       # Calculate the angle for the current point
       #print('i=', i, '\n')
       angle = i * angle_increment

       # Calculate the coordinates of the current point on the circle
       x = center[0] + radius * math.cos(math.radians(angle))
       y = center[1] + radius * math.sin(math.radians(angle))
       #z = center[2]  # Assuming the robot operates in a 2D plane
       z = 0
       #print('x=', x*10, '      y=', y, '      z=', z, '\n')
    
       #comdcircle = str(int(x*10))+' '+str(int(y*10))+' '+str(int(z))+' MOVE\n'
       comdcircle = str(int(x))+' '+str(int(y))+' '+str(int(z))+' MOVE\n'  
       print('radius=', radius, '\n')
       RunCmmd(comdcircle)
   
   r12shell.cmdloop( ['HOME'])     

def DrawRectangle(xii,yii,x,y):
   InitRect()
   print('\n--------------------------------------- rectfunction----------------------\n')
   xstep = (x - xii +100)*2
   ystep= (y - yii +100)*2
   print(f"xii: {xii}  xstep: {xstep}  x: {x}\n" )
   print(f"yii: {yii}  ystep: {ystep}  y: {y}\n" )
   #SetZaxis()
   comdcircle = str(int(xstep))+' '+str(int(0))+' '+str(int(0))+' MOVE\n'
   RunCmmd(comdcircle)
   comdcircle = str(int(0))+' '+str(int(-ystep))+' '+str(int(0))+' MOVE\n'
   RunCmmd(comdcircle)
   comdcircle = str(int(-xstep))+' '+str(int(0))+' '+str(int(0))+' MOVE\n'
   RunCmmd(comdcircle)
   comdcircle = str(int(0))+' '+str(int(ystep))+' '+str(int(0))+' MOVE\n'  
   RunCmmd(comdcircle)
   
   r12shell.cmdloop( ['HOME'])     
     


    
#----------------------------------------------------------------------------------------------
#contants
ml = 150
max_x, max_y = 250+ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0,0

#get tools function
def getTool(x):
	if x < 50 + ml:
		return "line"

	elif x<100 + ml:
		return "rectangle"

	elif x < 150 + ml:
		return"draw"

	elif x<200 + ml:
		return "circle"

	else:
		return "erase"

def index_raised(yi, y9):
	if (y9 - yi) > 40:
		return True

	return False



hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils


# drawing tools
tools = cv2.imread("tools.png")
tools = tools.astype('uint8')

mask = np.ones((480, 640))*255
mask = mask.astype('uint8')

cap = cv2.VideoCapture(0)
while True:
	_, frm = cap.read()
	frm = cv2.flip(frm, 1)

	rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

	op = hand_landmark.process(rgb)

	if op.multi_hand_landmarks:
		for i in op.multi_hand_landmarks:
			#print('i=', i, '\n')
			draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
			x, y = int(i.landmark[8].x*640), int(i.landmark[8].y*480)
			#print('\nInitial coordinates of the x=',x ,'\n')
			#print('\nInitial coordinates of the y=',y ,'\n')

			if x < max_x and y < max_y and x > ml:
				if time_init:
					ctime = time.time()
					time_init = False
				ptime = time.time()

				cv2.circle(frm, (x, y), rad, (0,255,255), 2)
				rad -= 1

				if (ptime - ctime) > 0.8:
					curr_tool = getTool(x)
					print("your current tool set to : ", curr_tool)
					time_init = True
					rad = 40

			else:
				time_init = True
				rad = 40
			print('curr_tool =', curr_tool, '\n')
			
			if curr_tool == "draw":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
					prevx, prevy = x, y

				else:
					prevx = x
					prevy = y



			elif curr_tool == "line":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.line(frm, (xii, yii), (x, y), (50,152,255), thick)

				else:
					if var_inits:
						cv2.line(mask, (xii, yii), (x, y), 0, thick)
						var_inits = False
						#DrawLine(xii, yii, x, y)

			elif curr_tool == "rectangle":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.rectangle(frm, (xii+20, yii+20), (x+20, y+20), (0,255,255), thick)
					print('if rectangle xc=', xii, '\n')
					print('yc=', yii, '\n')
					print('x=', x, '\n')
					print('y=', y, '\n')

				else:
					if var_inits:
						print('elserectangle=','\n')
						print('yc=', yii, '\n')
						cv2.rectangle(mask, (xii+20, yii+20), (x+20, y+20), 0, thick)
						var_inits = False
						'''
						print('else coordinates of rectangle\n')
						print('xii=', xii, '\n')
						print('yii=', yii, '\n')
						print('x=', x, '\n')
						print('y=', y, '\n')
						'''
						curr_tool = "select tool"
						#DrawRectangle(xii,yii,x,y)
					        

			elif curr_tool == "circle":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)
				

				if index_raised(yi, y9):
					print('if index_raised(yi, y9)',index_raised(yi, y9),'\n')   
				        
					if not(var_inits):
						#print('inot(var_inits)',not(var_inits),'\n')
						#print('if index_raised(yi, y9)',index_raised(yi, y9),'\n')
						xii, yii = x, y
						var_inits = True
						
						#print('xc=', xii, '\n')
						#print('yc=', yii, '\n')
						#print('x=', x, '\n')
						#print('y=', y, '\n')
						print('radius=', int(((xii-x)**2 + (yii-y)**2)**0.5), '\n')

					#cv2.circle(frm, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (255,255,0), thick) 
					#cv2.circle(frm, (320, 200), int(((xii-x)**2 + (yii-y)**2)**0.5), (255,255,0), thick) 
					print('\n drawing done    with -------- if  \n')
					
				else:
					print('else index_raised(yi, y9)',index_raised(yi, y9),'\n')
					print('\n drawing done    with -------- else \n')
					if var_inits:
						cv2.circle(mask, (320, 200), int(((xii-x)**2 + (yii-y)**2)**0.5), (0,255,0), thick)
						var_inits = False
						radius=int(((xii-x)**2 + (yii-y)**2)**0.5)
						radius-=100
					#InitCircle(radius)	
					curr_tool = "select tool"

			elif curr_tool == "erase":
				'''
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.circle(frm, (x, y), 30, (0,0,0), -1)
					cv2.circle(mask, (x, y), 30, 255, -1)
                                '''
				pass


	op = cv2.bitwise_and(frm, frm, mask=mask)
	frm[:, :, 1] = op[:, :, 1]
	frm[:, :, 2] = op[:, :, 2]

	frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

	cv2.putText(frm, curr_tool, (270+ml,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow("paint app", frm)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break
 
