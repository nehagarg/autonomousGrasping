#include "ext_tui.h"
#include "pocman.h"



class PocmanTUI: public TUI {
public:
  PocmanTUI() {
  }
  
    DSPOMDP* InitializeModel(option::Option* options) {
    DSPOMDP* model;
    int number = -1;
        if (options[E_NUMBER]) {
            number = atoi(options[E_NUMBER].arg);
            std::cout << "Number is " << number;
        }

        if (options[E_PARAMS_FILE]) {

            std::cout << "Config file is " << options[E_PARAMS_FILE].arg << std::endl;

                        model = new FullPocman( number, options[E_PARAMS_FILE].arg);
                }
        else
        {
            model = new FullPocman(number);
        }
    return model;
  }

  void InitializeDefaultParameters() {
     Globals::config.num_scenarios = 100;
  }

};

int main(int argc, char* argv[]) {
  return PocmanTUI().run(argc, argv);
}
