#include "ext_tui.h"
#include "grasping_v4.h"



class GraspingV4TUI: public TUI {
public:
  GraspingV4TUI() {
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
        int number = -1;
        if (options[E_NUMBER]) {
            number = atoi(options[E_NUMBER].arg);
            std::cout << "Number is " << number;
        }

        if (options[E_PARAMS_FILE]) {

            std::cout << "Config file is " << options[E_PARAMS_FILE].arg << std::endl;
            model = new GraspingV4(number, options[E_PARAMS_FILE].arg);
        } else {
            model = new GraspingV4(number);
        }
         //       }

    return model;
  }
};

int main(int argc, char* argv[]) {
  return GraspingV4TUI().run(argc, argv);
}
