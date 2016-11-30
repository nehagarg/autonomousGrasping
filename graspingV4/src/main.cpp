#include "grasping_tui.h"
#include "grasping_v4.h"



  
    DSPOMDP* TUI::InitializeModel(option::Option* options) {
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
                        model = new GraspingV4( number);
                }
                else
                {
                    model = new GraspingV4(-1);
                }
     //       }
     
    return model;
  }
int main(int argc, char* argv[]) {
  return TUI().run(argc, argv);
}
