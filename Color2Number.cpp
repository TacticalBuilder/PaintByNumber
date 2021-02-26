// Functions for

#include "Color2Number.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
using namespace std;

double ColorSetEntry::distance_hex (unsigned int target) {
  unsigned int t_iso_red, t_iso_blue, t_iso_green; //target isolated colors
  unsigned int e_iso_red, e_iso_blue, e_iso_green; //entry isolated colors
  int sqr_dist, dist_r, dist_g, dist_b;
  //cout << "Comparing: " <<  target << " - " << color_hex << endl;
  t_iso_red = (0xFF0000 & target) >> 16;
  e_iso_red = (0xFF0000 & color_hex) >> 16;

  //cout << "Red iso " << t_iso_red << " - " << e_iso_red << endl;

  t_iso_green = (0xFF00 & target) >> 8;
  e_iso_green = (0xFF00 & color_hex) >> 8;

  //cout << "Green iso " << t_iso_green << " - " << e_iso_green << endl;

  t_iso_blue = 0xFF & target;
  e_iso_blue = 0xFF & color_hex;

  //cout << "Blue iso " << t_iso_blue << " - " << e_iso_blue << endl;

  dist_r = e_iso_red - t_iso_red;
  dist_g = e_iso_green - t_iso_green;
  dist_b = e_iso_blue - t_iso_blue;

  sqr_dist = (dist_r * dist_r) + (dist_b * dist_b) + (dist_g * dist_g);

  return sqrt(sqr_dist);
}


double ColorSetEntry::distance_rgb (unsigned int target[3]) {
  int dist_r = color_rgb[C2N_RED] - target[C2N_RED];
  int dist_g = color_rgb[C2N_GREEN] - target[C2N_GREEN];
  int dist_b = color_rgb[C2N_BLUE] - target[C2N_BLUE];

  int sqr_dist = (dist_r * dist_r) + (dist_b * dist_b) + (dist_g * dist_g);

  return sqrt(sqr_dist);
}


ColorSet::ColorSet (string file_name) {
  string line, header;
  vector<ColorSetEntry> new_colors;
  ifstream color_file(file_name);
  int cnt = 0;

  if(!color_file.is_open()) { return; } //catch no file. IDK what we want.

  getline(color_file, header);
  while(getline(color_file, line)) {
    //cout << line << endl;
    stringstream line_str(line);
    string substr;

    getline(line_str, substr, ',');  //number
    //cout << "\tnumber:" << substr << endl;
    colors[cnt].color_num = stoul(substr);

    getline(line_str, substr, ',');  //hex
    //cout << "\thex:" << substr << endl;
    colors[cnt].color_hex = stoul(substr.substr(2), nullptr, 16);

    getline(line_str, substr, ',');  //RGB red
    //cout << "\tred:" << substr;
    //cout << "\tfinal:" << substr.substr(2) << endl;
    colors[cnt].color_rgb[C2N_RED] = stoul(substr.substr(2));

    getline(line_str, substr, ',');  //RGB green
    //cout << "\tgreen:" << substr << endl;
    colors[cnt].color_rgb[C2N_GREEN] = stoul(substr);

    getline(line_str, substr, ',');  //RGB blue
    //cout << "\tblue:" << substr << endl;
    colors[cnt].color_rgb[C2N_BLUE] = stoul(substr);

    getline(line_str, substr, ',');  //common name
    //cout << "\tname:" << substr << endl;
    colors[cnt].name = substr;

    cnt++;
  }
  set_name = file_name;
  num_colors = cnt;
}

//may want to rewrite for loop so that all distances are calc'd simultaneously
//comparisons are not worth parallelizing Id think
int ColorSet::color2num_hex (unsigned int target) {
  int color_id = colors[0].color_num;
  unsigned int dist = colors[0].distance_hex(target);
  unsigned int temp_dist;

  for(int i = 0; i < num_colors; ++i) {
    temp_dist = colors[i].distance_hex(target);
    cout << "Test Dist: " << temp_dist << " color: " << colors[i].name << endl;
    if(temp_dist < dist) {
      dist = temp_dist;
      color_id = colors[i].color_num;
    }
  }

  return color_id;
}


int ColorSet::color2num_rgb (unsigned int target[3]) {
  int color_id = colors[0].color_num;
  unsigned int dist = colors[0].distance_rgb(target);
  unsigned int temp_dist;

  for(int i = 0; i < num_colors; ++i) {
    temp_dist = colors[i].distance_rgb(target);
    cout << "Test Dist: " << temp_dist << " color: " << colors[i].name << endl;
    if(temp_dist < dist) {
      dist = temp_dist;
      color_id = colors[i].color_num;
    }
  }

  return color_id;
}


string ColorSet::num2name (int id) {
  for(int i = 0; i < num_colors; i++) {
    if(colors[i].color_num == id) {
      return colors[i].name;
    }
  }

  string default_str("none");
  return default_str;
  //return colors[id].name //only if colors are entered in order
}


int main() {
  string file = "crayola_22pk.txt";
  ColorSet my_set(file);
  cout << my_set.set_name << "\t" << my_set.num_colors << endl;

  unsigned int test_col1 = 0xC561ED; //Heliotrope
  unsigned int test_col2 = 0xED6199; //Brilliant Rose
  unsigned int test_col3[] = {170, 251, 162}; //Mint Green
  unsigned int test_col4[] = {251, 247, 162}; //Texas

  //cout << "Distance: " << my_set.colors[1].distance_hex(test_col1) << endl;
  //int r1 = my_set.color2num_hex(test_col1);
  //int r2 = my_set.color2num_hex(test_col2);
  int r3 = my_set.color2num_rgb(test_col3);
  //int r4 = my_set.color2num_rgb(test_col4);

  //cout << "Test 1 [Heliotrope] - " << r1 << ">" << my_set.num2name(r1) << endl;
  //cout << "Test 2 [Brill. Rose] - " << r2 << ">" << my_set.num2name(r2) << endl;
  cout << "Test 3 [Mint Green] - " << r3 << ">" << my_set.num2name(r3) << endl;
  //cout << "Test 4 [Texas] - " << r4 << ">" << my_set.num2name(r4) << endl;
  return 0;
}
