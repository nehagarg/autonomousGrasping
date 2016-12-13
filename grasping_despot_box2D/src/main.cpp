#include <grasping_tui.h>
#include "grasping_box2D.h"

using namespace despot;
/*
class Box2DTUI : public TUI {
public:
    Box2DTUI()  {
        
    }
    
    DSPOMDP* InitializeModel(option::Option* options) {
     DSPOMDP* model;
    if (options[E_DATA_FILE])
            {
                if (options[E_NUMBER]) {
                        int number = atoi(options[E_NUMBER].arg);
                        
                        model = new GraspingV4(options[E_DATA_FILE].arg, number);
                }
                else
                {
                    model = new GraspingV4(options[E_DATA_FILE].arg, -1);
                }
            }
            else
    
     //{
                if (options[E_NUMBER]) {
                        int number = atoi(options[E_NUMBER].arg);
                        std::cout << "Number is " <<  number;
                        model = new GraspingBox2D( number);
                }
                else
                {
                    model = new GraspingBox2D(-1);
                }
     //       }
     
    return model;
  }
  
  void InitializeDefaultParameters() {
  }

};
*/
class Box2DTUI : public TUI {
public:
    Box2DTUI()  {
        
    }
    DSPOMDP* InitializeModel(option::Option* options) {
     DSPOMDP* model;
    /*if (options[E_DATA_FILE])
            {
                if (options[E_NUMBER]) {
                        int number = atoi(options[E_NUMBER].arg);
                        
                        model = new GraspingV4(options[E_DATA_FILE].arg, number);
                }
                else
                {
                    model = new GraspingV4(options[E_DATA_FILE].arg, -1);
                }
            }
            else
    */
     //{
                if (options[E_NUMBER]) {
                        int number = atoi(options[E_NUMBER].arg);
                        std::cout << "Number is " <<  number;
                        model = new GraspingBox2D( number);
                }
                else
                {
                    model = new GraspingBox2D(-1);
                }
     //       }
     
    return model;
  }

};
int main(int argc, char* argv[]) {
  return Box2DTUI().run(argc, argv);
}
