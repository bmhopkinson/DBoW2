/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <experimental/filesystem>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"


using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(vector<vector<cv::Mat > > &features);
void loadFeaturesSURF(vector<vector<std::vector<float> > > &features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void changeStructureSURF(const std::vector<float> &plain, std::vector<std::vector<float>> &out, int L);
void testVocCreation(const vector<vector<cv::Mat > > &features);
void testVocSURFCreation(const vector<vector<std::vector<float>  > > &features);
void testDatabase(const vector<vector<cv::Mat > > &features);
void testDatabaseSURF(const vector<vector<std::vector<float> > > &features);
cv::Mat PreProcessImg(cv::Mat &img);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
 int NIMAGES ;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// ----------------------------------------------------------------------------

int main()
{
  //vector<vector<cv::Mat > > features;
  vector<vector<std::vector<float> > > features;
  loadFeaturesSURF(features);

  testVocSURFCreation(features);

  wait();

  testDatabaseSURF(features);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features)
{
    int NIMAGE_EST = 100;
  features.clear();
  features.reserve(NIMAGE_EST);

  cv::Ptr<cv::ORB> detector = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;
  std::string img_path = "../data/images_vocab";
 // for(int i = 0; i < NIMAGES; ++i)
 for(const auto & entry : std::experimental::filesystem::directory_iterator(img_path))
  {
     std::cout << entry.path().string() << std::endl;

    cv::Mat image = cv::imread(entry.path().string() , 0);
    image = PreProcessImg(image);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    detector->detectAndCompute(image, mask, keypoints, descriptors);

    features.push_back(vector<cv::Mat >());
    changeStructure(descriptors, features.back());
    NIMAGES++;

  }
}

// ----------------------------------------------------------------------------

void loadFeaturesSURF(vector<vector<std::vector<float> > > &features){
    int NIMAGE_EST = 100;
    features.clear();
    features.reserve(NIMAGE_EST);

    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(20, 4, 2);

    cout << "Extracting SURF features..." << endl;
    std::string img_path = "../data/images_vocab";
    // for(int i = 0; i < NIMAGES; ++i)
    for(const auto & entry : std::experimental::filesystem::directory_iterator(img_path))
    {
        std::cout << entry.path().string() << std::endl;

        cv::Mat image = cv::imread(entry.path().string() , 0);
        image = PreProcessImg(image);
        cv::Mat mask;
        vector<cv::KeyPoint> keypoints;
        std::vector<float> descriptors;

        detector->detectAndCompute(image, mask, keypoints, descriptors);

        features.push_back(vector<std::vector<float> >());
        changeStructureSURF(descriptors, features.back(), detector->descriptorSize());
        NIMAGES++;

    }


}

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

void changeStructureSURF(const std::vector<float> &plain, std::vector<std::vector<float>> &out, int L){
    out.resize(plain.size() / L);
    unsigned int j = 0;
    for(unsigned int i = 0; i <plain.size(); i+=L, ++j){
        out[j].resize(L);
        std::copy(plain.begin()+i, plain.begin()+i+L, out[j].begin());
    }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<cv::Mat > > &features)
{
  // branching factor and depth levels 
  const int k = 7;
  const int L = 5;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L1_NORM;

  OrbVocabulary voc(k, L, weight, scoring);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for(int i = 0; i < NIMAGES; i++)
  {
    voc.transform(features[i], v1);
  //  std::cout << v1 << std::endl;
    for(int j = 0; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);
      
      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  voc.saveToBinaryFile("ORBVocab_alt.bin");
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testVocSURFCreation(const vector<vector<std::vector<float> > > &features){
    // branching factor and depth levels
    const int k = 6;
    const int L = 5;
    const WeightingType weight = TF_IDF;
    const ScoringType scoring = L1_NORM;

    Surf64Vocabulary voc(k, L, weight, scoring);

    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;

    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;
    for(int i = 0; i < NIMAGES; i++)
    {
        voc.transform(features[i], v1);
        //  std::cout << v1 << std::endl;
        for(int j = 0; j < NIMAGES; j++)
        {
            voc.transform(features[j], v2);

            double score = voc.score(v1, v2);
            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
        }
    }

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save("small_surf_voc.yml.gz");
   // voc.saveToBinaryFile("SURFVocab.bin");
    cout << "Done" << endl;

}

void testDatabase(const vector<vector<cv::Mat > > &features)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  OrbVocabulary voc("small_voc.yml.gz");
  
  OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  cout << "Retrieving database once again..." << endl;
  OrbDatabase db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}


void testDatabaseSURF(const vector<vector<std::vector<float> > > &features)
{
    cout << "Creating a small database..." << endl;

    // load the vocabulary from disk
    Surf64Vocabulary voc("small_surf_voc.yml.gz");

    Surf64Database db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for(int i = 0; i < NIMAGES; i++)
    {
        db.add(features[i]);
    }

    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;

    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    for(int i = 0; i < NIMAGES; i++)
    {
        db.query(features[i], ret, 4);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
    }

    cout << endl;

    // we can save the database. The created file includes the vocabulary
    // and the entries added
    cout << "Saving database..." << endl;
    db.save("small_db.yml.gz");
    cout << "... done!" << endl;

    // once saved, we can load it again
    cout << "Retrieving database once again..." << endl;
    Surf64Database db2("small_db.yml.gz");
    cout << "... done! This is: " << endl << db2 << endl;
}

cv::Mat PreProcessImg(cv::Mat &img){

    int width = img.size().width;
    float fscale = 1.000;
    if(width > 2000){
        fscale = static_cast<float>(width)/2000.0;
    }
    cv::resize(img, img, cv::Size(), fscale, fscale);
    //cv::cvtColor(img, img ,CV_RGB2GRAY);


    return img;
}

// ----------------------------------------------------------------------------


