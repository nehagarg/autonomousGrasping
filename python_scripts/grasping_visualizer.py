import matplotlib.pyplot as plt
import re


state_line = "Gripper at: 7.33333,5 Opening left: 2 Opening Right:4 Circle radius:2 for object id 1"
state_line = "Gripper at: 1.66667,4 Opening left: 2 Opening Right:3 Circle radius:4 for object id 1"
state_line = "Gripper at: -4,-14.5826 Opening left: 1 Opening Right:2 Circle radius:5 for object id 0"
state_line = "Gripper at: 0,-14.899 Opening left: 1 Opening Right:1 Circle radius:5 for object id 0"
state_line = "Gripper at: 15,15 Opening left: 5 Opening Right:5 Circle radius:5 for object id 0"
state_line = "Gripper at: 15,15 Opening left: 5 Opening Right:4 Circle radius:5 for object id 0"
state_line = "Gripper at: -6,-14 Opening left: 0 Opening Right:3 Circle radius:5 for object id 0"

state_line = "Gripper at: -15,-15 Opening left: 5 Opening Right:5 Circle radius:5 for object id 0"

state_line = "Gripper at: -3,-14.899 Opening left: 5 Opening Right:5 Circle radius:5 for object id 0"
state_line = "Gripper at: 1.393,-13.4616 Opening left: 5 Opening Right:2.216 Circle radius:5 for object id 0"
state_line = "Gripper at: 1.394,-13.4616 Opening left: 5 Opening Right:2.216 Circle radius:5 for object id 0"
state_line = "Gripper at: 0.874,-4 Opening left: 4.874 Opening Right:3.126 Circle radius:4 for object id 1"
state_line = "Gripper at: 0.874,-4 Opening left: 5 Opening Right:5 Circle radius:4 for object id 1"
state_line = "Gripper at: -4.58258,2 w.r.t object Opening left: 0 Opening Right:0 Circle radius:5 for object id 0 World coordinates:3.60942,17 Change:0,0"
state_line = "Gripper at: -0.272,-11.664 w.r.t object Opening left: 5 Opening Right:5 Circle radius:5 for object id 0 World coordinates:7.92,3.336 Change:0,1.024"
state_line = "Gripper at: -2.047,-14.0356 w.r.t object Opening left: 5 Opening Right:5 Circle radius:5 for object id 0 World coordinates:6.145,0.964446 Change:0.001,0"
state_line = "Gripper at: -4.44089e-16,-11.9876 w.r.t object Opening left: 5 Opening Right:5 Circle radius:5 for object id 0 World coordinates:8.192,3.01245 Change:0,2.048"
state_line = "Gripper at: -0.0663248,-12.0635 w.r.t object Opening left: 5 Opening Right:5 Circle radius:5 for object id 0 World coordinates:7.71145,2.93648 Change:0.512,0"
state_line = "Gripper at: 0.61,-12.988 w.r.t object Opening left: 5 Opening Right:5 Circle radius:5 for object id 0 World coordinates:8.802,2.01204 Change:0,0.256"
state_line = "Gripper at: 0.000729306,-10.0894 w.r.t object Opening left: 5 Opening Right:5 Circle radius:5 for object id 0 World coordinates:9.00073,3.9106 Change:0,0.004"
state_line = "Gripper at: -0.786,3.56267 w.r.t object Opening left: 5 Opening Right:5 Circle radius:1 for object id 4 World coordinates:9.214,16.896 Change:16.384,0"
state_line = "Gripper at: -4.5,-0.5 w.r.t object Opening left: 0 Opening Right:5 Circle radius:0.5 for object id 0 World coordinates:5.5,14.5 Change:0,0"

#state_line = "Gripper at: -9.344,-15.596 w.r.t object Opening left: 0 Opening Right:0 Circle radius:0.5 for object id 0 World coordinates:0.656,-0.596 Change:0,0"
numeric_const_pattern = r"""
[-+]? # optional sign
(?:
   (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
        |
         (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
     )
     # followed by optional exponent part if desired
     (?: [Ee] [+-]? \d+ ) ?
     """

rx = re.compile(numeric_const_pattern, re.VERBOSE)
values = rx.findall(state_line)
print values
if(len(values) >= 6):
    gripper_x = float(values[0])
    gripper_y = float(values[1])
    gripper_l = float(values[2])
    gripper_r = float(values[3])
    circle_radius = float(values[4])
    circle1=plt.Circle((0,0),circle_radius,color='r', clip_on=False)
    #circle2=plt.Circle((.5,.5),2,color='b')
    #circle3=plt.Circle((1,1),2,color='g',clip_on=False)
    
    #gripper
    y1 = [gripper_y, gripper_y]
    x1 = [gripper_x - gripper_l, gripper_x+gripper_r]
    plt.plot(x1, y1, 'go-', label='line 1', linewidth=2)
    y1 = [gripper_y, gripper_y+10]
    x1 = [gripper_x - gripper_l, gripper_x-gripper_l]
    plt.plot(x1, y1, 'go-', label='line 1', linewidth=2)
    y1 = [gripper_y, gripper_y+10]
    x1 = [gripper_x + gripper_r, gripper_x+gripper_r]
    plt.plot(x1, y1, 'go-', label='line 1', linewidth=2)
    
    #gripper center
    y1 = [gripper_y]
    x1 = [gripper_x]
    plt.plot(x1, y1, 'bo-', label='line 1', linewidth=2)
    
    #gripper sensors
    for i in range(0, 11) :
        y1 = [gripper_y + i]
        x1 = [gripper_x - gripper_l]
        plt.plot(x1, y1, 'yo-', label='line 1', linewidth=2)
        y1 = [gripper_y + i]
        x1 = [gripper_x + gripper_r]
        plt.plot(x1, y1, 'yo-', label='line 1', linewidth=2)

    range_xy = 15
    y1 = [-range_xy, -range_xy]
    x1 = [-range_xy, range_xy]
    plt.plot(x1, y1, 'bo-', label='line 1', linewidth=1)
    y1 = [range_xy, range_xy]
    x1 = [-range_xy, range_xy]
    plt.plot(x1, y1, 'bo-', label='line 1', linewidth=1)
    y1 = [-range_xy, range_xy]
    x1 = [-range_xy, -range_xy]
    plt.plot(x1, y1, 'bo-', label='line 1', linewidth=1)
    y1 = [-range_xy, range_xy]
    x1 = [range_xy, range_xy]
    plt.plot(x1, y1, 'bo-', label='line 1', linewidth=1)
    
    plt.xlim(-20,20)
    plt.ylim(-15,25)

    #y1 = [1,5,7,3]
    #x1 = range(1,5)
    #y2 = [3,5,10,3,6,8]
    #x2 = range(4,len(y2)+4)
    
    #plt.plot(x2, y2, 'rs--',  label='line 2')
    #plt.legend()
    #plt.show()

    fig = plt.gcf()
    #fig.set_size_inches(8,8)
    fig.gca().add_artist(circle1)
    #fig.gca().add_artist(circle2)
    #fig.gca().add_artist(circle3)
    fig.show()

