#ifndef COLOR_2_NUMBER_G_
#define COLOR_2_NUMBER_G_

#include <string>

#define C2N_RED 0
#define C2N_GREEN 1
#define C2N_BLUE 2

#define MAX_COLORS 128


using namespace std;

//Data type definitions
class ColorSetEntry {
  public:
    unsigned int color_num;   //Number assigned to this color entry
    string name;              //Common Name
    unsigned int color_hex;   //HEX value for the color
    unsigned int color_rgb[3];//RGB triple of the color

    double distance_hex(unsigned int target);
    double distance_rgb(unsigned int target[3]);
};

class ColorSet {
  public:
    int num_colors;          // total colors available
    string set_name;         // common name of set
    ColorSetEntry colors[MAX_COLORS];  // all colors contained by set

    ColorSet(string file_name);

    int color2num_rgb(unsigned int target[3]);
    int color2num_hex(unsigned int target);
    string num2name(int id);
};

#endif
