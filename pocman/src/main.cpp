#include "ext_tui.h"
#include "pocman.h"



class PocmanTUI: public TUI {
public:
  PocmanTUI() {
  }
  
    DSPOMDP* InitializeModel(option::Option* options) {
    DSPOMDP* model;
        if (options[E_NUMBER]) {
                        int number = atoi(options[E_NUMBER].arg);
                        std::cout << "Number is " <<  number;
                        model = new FullPocman( number);
                }
        else
        {
            model = new FullPocman();
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
