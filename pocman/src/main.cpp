#include "grasping_tui.h"
#include "pocman.h"



class PocmanTUI: public TUI {
public:
  PocmanTUI() {
  }
  
    DSPOMDP* InitializeModel(option::Option* options) {
    DSPOMDP* model = new FullPocman();
    return model;
  }

  void InitializeDefaultParameters() {
     Globals::config.num_scenarios = 100;
  }

};

int main(int argc, char* argv[]) {
  return PocmanTUI().run(argc, argv);
}
