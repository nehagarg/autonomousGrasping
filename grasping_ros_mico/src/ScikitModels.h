/* 
 * File:   ScikitModels.h
 * Author: neha
 *
 * Created on November 17, 2017, 7:17 PM
 */

#ifndef SCIKITMODELS_H
#define	SCIKITMODELS_H


class ScikitModels {
public:
    ScikitModels(std::string model_type, std::string yaml_file);
    ScikitModels(const ScikitModels& orig);
    virtual ~ScikitModels();
    virtual void loadModel(std::string yaml_file) = 0;
    virtual std::vector<double>  predict(std::vector<double> x) = 0;
private:
    std::string model_type;
    
};

class MultiScikitModels {
public:
    MultiScikitModels(std::string model_dir,  std::string classifier_type, int action, int num_predictions);
    virtual ~MultiScikitModels() {
    }
    virtual  std::vector<double>  predict(std::vector<double> x);
private:
    std::vector<ScikitModels*> models;
};

class DecisionTreeScikitModel : public ScikitModels {
public:
    DecisionTreeScikitModel(std::string model_type, std::string yaml_file) : ScikitModels(model_type, yaml_file){};
    virtual ~DecisionTreeScikitModel() {};

    void loadModel(std::string yaml_file);

    std::vector<double>  predict(std::vector<double> x);


private:
    int n_nodes ;
    int max_depth;
    std::vector<int> children_left ;
    std::vector<int>children_right;
    std::vector<int>feature ;
    std::vector<double>threshold;
    std::vector<std::vector<double> >  value;
    
    int apply(std::vector<double> x);
    
    
};

class LinearScikitModel : public ScikitModels {
public:
    LinearScikitModel(std::string model_type, std::string yaml_file): ScikitModels(model_type, yaml_file){};
    virtual ~LinearScikitModel() {};
    
    void loadModel(std::string yaml_file);

    std::vector<double> predict(std::vector<double> x);



private:
    std::vector<double>coeff;
    double intercept;
};


#endif	/* SCIKITMODELS_H */

