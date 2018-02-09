#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list, const TLabels& labels, const string& prediction_file) {
    // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());
        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

#define kletok 8
#define segments 32 //angle segmentst
#define color_blocks 8 //num of blocks for coloring
#define pi 3.14159265

vector<float> make_hist(pair<vector<vector<float>>, vector<vector<float>>> sob) //mod arc
//make loooong hist for mod and arc vectors of image
{
    vector<float> res;
    vector<float> hist(segments, 0.0);
    uint kletka_height = sob.first.size() / kletok, 
        kletka_width = sob.first[0].size() / kletok;
    for (uint v = 0; v < sob.first.size(); v = v + kletka_height) 
    {
        for (uint h = 0; h < sob.first[0].size(); h = h + kletka_width) 
        {
            hist.assign(segments, 0.0);
            for (uint i = 0; i < kletka_height; i++) 
            {
                for (uint j = 0; j < kletka_width; j++) 
                {
                    if (i + v < sob.first.size() && j + h < sob.first[0].size()) 
                    {
                        hist[static_cast<int>(floor(sob.second[i + v][j + h] / 2 / pi * segments))] += 
                            sob.first[i + v][j + h];
                    }
                }
            }
            float sum = 0;
            for (uint k = 0; k < hist.size(); k++) //normalizing
            {
                sum += hist[k]*hist[k];
            }
            sum = sqrt(sum);
            if (sum > 0) 
            {
                for (uint k = 0; k < hist.size(); k++) 
                {
                    hist[k] /= sum;
                }
            }
            res.insert(res.end(), hist.begin(), hist.end());
        }
    }
    return res;
}

pair<vector<vector<float>>, vector<vector<float>>> //LOL WTF MLG STYLE
    make_gradiented(vector<vector<int>> image) //make 2 vectors with modul and arctg(from 0 to2pi)
{
    vector<vector<float>> sob_mod(image.size() - 2, vector<float>(image[0].size() - 2));
    vector<vector<float>> sob_arc(image.size() - 2, vector<float>(image[0].size() - 2));
    for (uint i = 1; i < sob_mod.size() + 1; i++) 
    {
        for (uint j = 1; j < sob_mod[1].size() + 1; j++) 
        {
            int vert = image[i][j + 1] - image[i][j - 1];
            int hor = image[i + 1][j] - image[i - 1][j];
            sob_mod[i - 1][j - 1] = sqrt(static_cast<double> (hor * hor + vert * vert));
            float arc = (hor == 0 ? pi / 2 : atan(static_cast<double> (vert) / static_cast<double> (hor)));
            if (hor == 0 && vert < 0) 
            {
                arc = -arc + 0.000001;
            }
            arc = arc + pi / 2;
            if (hor < 0) 
            {
                arc = arc + pi;
            }
            sob_arc[i - 1][j - 1] = arc;
        }
    }
    return make_pair(sob_mod, sob_arc);
}

vector<vector<int>> make_gray(BMP *im) //makes grayscale image int vector<vector> from BMP image
{
    int wid = im->TellWidth(), hei = im->TellHeight();
    vector<vector<int>> res(hei, vector<int>(wid));
    for (int j = 0; j < hei; j++) {
        for (int i = 0; i < wid; i++) {
            res[j][i] = static_cast<int> (floor(0.299 * (*im)(i,j)->Red + 0.587 * (*im)(i,j)->Green + 0.114 * (*im)(i,j)->Blue));
        }
    }
    return res;
}



void add_color(vector<float> &feat, BMP *im) //add color features
{
    uint block_height = (*im).TellHeight() / color_blocks;
    uint block_width = (*im).TellWidth() / color_blocks;
    float square_norm = block_height * block_width * 255;
    for (uint j = 0; j < color_blocks; j++)
    {
        for (uint i = 0; i < color_blocks; i++) 
        {
            vector<float> average (3, 0);
            for (uint h = 0; h < block_height; h++) 
            {
                for (uint w = 0; w < block_width; w++)
                {
                    average[0] += (*im)(i * block_width + w, j * block_height + h)->Red;
                    average[1] += (*im)(i * block_width + w, j * block_height + h)->Green;
                    average[2] += (*im)(i * block_width + w, j * block_height + h)->Blue;
                }
            }
            average[0] /= square_norm;
            average[1] /= square_norm;
            average[2] /= square_norm;
            feat.insert(feat.end(), average.begin(), average.end());
        }
    }

    return;
}

void add_locbin(vector<float> &feat, const vector<vector<int>> gray) //vector<int> &feat,
{
    uint kletka_height = (gray.size() - 2) / kletok, 
        kletka_width = (gray[0].size() - 2) / kletok;
    uint vert_size = gray.size(),
        hor_size = gray[0].size();
    vector<float> hist(256, 0.0f);
    for (uint v = 1; v < vert_size - 1; v = v + kletka_height) 
    {
        for (uint h = 1; h < hor_size - 1; h = h + kletka_width) 
        {
            hist.assign(256, 0);
            for (uint i = 0; i < kletka_height; i++) 
            {
                for (uint j = 0; j < kletka_width; j++) 
                {   
                    uint temp = 0;
                    if (i + v < vert_size - 1 && j + h < hor_size - 1) 
                    {
                        temp |= (gray[i+v][j+h] <= gray[i+v-1][j+h-1]) << 0;
                        temp |= (gray[i+v][j+h] <= gray[i+v-1][j+h]) << 1;
                        temp |= (gray[i+v][j+h] <= gray[i+v-1][j+h+1]) << 2;
                        temp |= (gray[i+v][j+h] <= gray[i+v][j+h+1])  << 3;
                        temp |= (gray[i+v][j+h] <= gray[i+v+1][j+h+1]) << 4;
                        temp |= (gray[i+v][j+h] <= gray[i+v+1][j+h]) << 5;
                        temp |= (gray[i+v][j+h] <= gray[i+v+1][j+h-1]) << 6;
                        temp |= (gray[i+v][j+h] <= gray[i+v][j+h-1]) << 7;
                        hist[temp] += 1;
                    }
                }
                float sum = 0;
                for (uint k = 0; k < hist.size(); k++) 
                {
                    sum += hist[k] * hist[k];
                }
                sum = (sqrt(static_cast<double> (sum)));
                for (uint k = 0; k < hist.size(); k++) 
                {
                    hist[k] = hist[k] / sum;
                }
                feat.insert(feat.end(), hist.begin(), hist.end());
            }
        }
    }
    return;
}

// Extract features from dataset.
// You should implement this function by yourself :=(
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        auto gray = make_gray(data_set[image_idx].first); //this wtf
        auto res = make_gradiented(gray);
        vector<float> one_image_features = make_hist(res);
        add_locbin(one_image_features, gray);
        add_color(one_image_features, data_set[image_idx].first);
        features->push_back(make_pair(one_image_features, data_set[image_idx].second));
    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}

//Copyright QEQER
//E.Yatsko 2017
//3 grade MSU CMC SP