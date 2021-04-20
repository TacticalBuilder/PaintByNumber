# Color Pack Object
"""
This object handles the 'crayons' that will be targetted by the program
Input is a csv of the colors, name, number, and RGB / HEX
sample line from correctly formatted input:

# labels image - this operation is exceptionally fast. Not worth parallelizing
def applyNumberLabels

# this operation is moderately expensive and scales with number of shapes
def betterColorToNumber_gpu
def betterColorToNumber
    |--ShapePoints[]

# Encapsulation of Color2Number operations and colorspace conversions
|--ColorPack
    |--ColorEntry[]
"""
import math
import cv2 as cv
import numpy as np
#import paint_by_numbers_pipeline as pbn

RGB_RED = 0
RGB_BLUE = 1
RGB_GREEN = 2

LAB_L = 0
LAB_A = 1
LAB_B = 2

COMPONENT_THRESH = 25  # defines the minimum size of a valid shape
PT_SURVAILENCE = 5     # defines the number of candidate points to inspect

# Organizes the points of a single conffnected components for candidate selection
class ShapePoints:
    def __init__(self):
        self.all_x = list()
        self.all_y = list()
        self.num_pts = 0

# GPU version of image color to number assignment
def betterColorToNumber_gpu(ref_img, shapeMask, crayons, num_comps):
    assert (COMPONENT_THRESH > 2 * PT_SURVAILENCE), "Min shape threshold too small to select reasonable sample pts."

    # GPU function definition
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule

    mod = SourceModule("""
    __global__ void color_to_number_gpu(int *ref_l, int *ref_a, int *ref_b, int *remask, int *mask_seek, int *s_mask, int *meta, int *f_cray, int *lab_targs)
    {
        // meta = rows, cols, num crayons, threshold, Number Points to select

        int t_idx = (blockIdx.x * blockDim.x) + threadIdx.x;  // 1 thread per shape
        int idx_pxls = mask_seek[t_idx+1] - mask_seek[t_idx]; // numbr pxls associated to shape
        int idx_offset = mask_seek[t_idx];                    // base of candidates in remask array

        const int num_crayons = meta[2];       //number of colors to test
        const int min_pxl_thresh = meta[3];    //minimum threshold of pxls to label
        const int sz_candidate_ls = meta[4];   //number of pixels polled for labeling
        int num_comps = meta[5];               //number of shapes (for testing overflow threads)

        int best_area = 0;
        int best_dist = 256000;      // massive starting comparison distance
        int cand_h, cand_w;
        int c_pt, dist, cand_area, best_dist_idx;
                                     // up, down, left, right
        int radius[] = {10,3,3,10};  // approximation of acceptable padding for label
        int buffer[] = {1,1,1,1};    // measured padding for candidate

        // check for overallocation of threads to problem
        if (t_idx >= num_comps) {
            return;
        }

        // check for tiny shapes to disregard (skip analysis if that is case)
        if ( idx_pxls < min_pxl_thresh ) {
            // flatten output is idx, stride 3
            lab_targs[t_idx*3] = 0;
            lab_targs[(t_idx*3)+1] = 0;
            lab_targs[(t_idx*3)+2] = -1;

        // run shape analysis

        } else {
            // survey for candidate label anchors
            for (int j = 0; j < sz_candidate_ls; j++) {
                c_pt = int( (j + 0.5) * idx_pxls / sz_candidate_ls ) + idx_offset;
                cand_h = remask[c_pt] / meta[1];
                cand_w = remask[c_pt] % meta[1];

                // test up direction
                for (int k = 0; k < radius[0]; k++) {
                    if ((cand_h - k) >= 0 ) {  //cap = 0
                        if (t_idx == s_mask[((cand_h - k) * meta[1]) + cand_w]) {
                            buffer[0] += 1;
                        }
                    }
                }

                // test down direction
                for (int k = 0; k < radius[1]; k++) {
                    if ((cand_h + k) < meta[0] ) { //cap = img_h
                        if (t_idx == s_mask[((cand_h + k) * meta[1]) + cand_w]) {
                            buffer[1] += 1;
                        }
                    }
                }

                //  test left direction
                for (int k = 0; k < radius[2]; k++) {
                    if ( (cand_w - k) >= 0 ) {  //cap = 0
                        if (t_idx == s_mask[(cand_h * meta[1]) + (cand_w - k)]) {
                            buffer[2] += 1;
                        }
                    }
                }

                //  test right direction
                for (int k = 0; k < radius[2]; k++) {
                    if ( (cand_w - k) < meta[1] ) {  // cap = img_w
                        if (t_idx == s_mask[(cand_h * meta[1]) + (cand_w - k)]) {
                            buffer[2] += 1;
                        }
                    }
                }

                cand_area = (buffer[0] + buffer[1]) * (buffer[2] + buffer[3]);
                if(cand_area > best_area) {
                    best_area = cand_area;
                    lab_targs[t_idx*3] = cand_w;
                    lab_targs[(t_idx*3)+1] = cand_h;
                }

                buffer[0] = 1; // reset buffer counter
                buffer[1] = 1;
                buffer[2] = 1;
                buffer[3] = 1;
            }

            // color 2 number selection process
            //    recreate 1D compressed pixel coordinate
            c_pt = (lab_targs[t_idx*3] * meta[0]) + lab_targs[(t_idx*3)+1];

            // LAB euclidean dist. comparison, uses dist^2 to avoid needing sqrt
            for (int colr = 0; colr < num_crayons; colr++) {
                dist = (ref_l[c_pt] - f_cray[colr*3]) * (ref_l[c_pt] - f_cray[colr*3]) + \
                       (ref_a[c_pt] - f_cray[(colr*3)+1]) * (ref_a[c_pt] - f_cray[(colr*3)+1]) + \
                       (ref_b[c_pt] - f_cray[(colr*3)+2]) * (ref_b[c_pt] - f_cray[(colr*3)+2]);

                // flattened LAB data relates number to flat_cray idx
                if (dist < best_dist) {
                    best_dist_idx = colr;
                    best_dist = dist;
                }
            }

            lab_targs[(t_idx*3) + 2] = best_dist_idx + 1;

        }
    }
    """)
    color_to_number_gpu = mod.get_function("color_to_number_gpu")

    # set up (convert existing data array)
    label_targets = np.zeros(num_comps * 3, dtype=np.int32)    # flattened array (stride = 3).
    ref_l = ref_img[:, :, 0].flatten().astype('uint8')
    ref_a = ref_img[:, :, 1].flatten().astype('uint8')
    ref_b = ref_img[:, :, 2].flatten().astype('uint8')
    s_mask = shapeMask.flatten().astype('uint32')
    flat_cray = crayons.flatten_bgr_data()
    #   rows, cols, num crayons, threshold, Number Points to select
    meta = np.array([int(shapeMask.shape[0]), int(shapeMask.shape[1]),
                     int(crayons.packSize()), int(COMPONENT_THRESH),
                     int(PT_SURVAILENCE), int(num_comps)], dtype=np.int32)
                     
    # set up (reshape shapeMask)
    remask = np.zeros(shapeMask.shape[0] * shapeMask.shape[1], dtype=np.int32)
    mask_seek = np.zeros(num_comps+1, dtype=np.int32)

    shapes = []
    for i in range(num_comps):
        shapes.append(ShapePoints())

    for i in range(shapeMask.shape[0]) :    # row
        for j in range(shapeMask.shape[1]) : # col
            shape_id = int(shapeMask[i][j])   # get region tag
            shapes[shape_id].all_x.append(i)  # assign row to x
            shapes[shape_id].all_y.append(j)  # assign col to y
            shapes[shape_id].num_pts += 1     # increment px count

    # set up (GPU friendly mask structure)
    rmask_idx = 0
    for i in range(num_comps):  # for each component
        for j in range(shapes[i].num_pts): # for each pxl held by component
            # remask[base+next] = rw_idx * rw_width + col_idx
            remask[rmask_idx+j] = int((shapes[i].all_x[j] * shapeMask.shape[1]) + shapes[i].all_y[j])
        mask_seek[i] = rmask_idx
        rmask_idx += shapes[i].num_pts
    mask_seek[len(mask_seek)-1] = rmask_idx

    #float *ref_l, float *ref_a, float *ref_b, int *remask, int *mask_seek, int *s_mask, int *meta, float *f_cray, int *lab_targs
    color_to_number_gpu(drv.In(ref_l), drv.In(ref_a), drv.In(ref_b), drv.In(remask), drv.In(mask_seek), drv.In(s_mask), drv.In(meta), drv.In(flat_cray), drv.Out(label_targets), block=(num_comps,1,1), grid=(1,1,1))
    label_returns = np.reshape(label_targets, (num_comps, 3))
    return label_returns


# Color to Number operation - returns a list of all labels (selected colors) and
#   the pixel of the image to anchor them to.
def betterColorToNumber(ref_img, shapeMask, crayons, num_comps):
    assert (COMPONENT_THRESH > 2 * PT_SURVAILENCE), "Min shape threshold too small to select reasonable sample pts."

    label_targets = np.zeros((num_comps, 3)) # output for image writing.
    shapes = [] # each component is given a blank list to append all related pixels to.
    for i in range(num_comps):
        shapes.append(ShapePoints())

    # mask transform
    for i in range(shapeMask.shape[0]) :    #row
        for j in range(shapeMask.shape[1]) : #col
            shape_id = int(shapeMask[i][j])   #get region tag
            shapes[shape_id].all_x.append(i)  #assign row to x
            shapes[shape_id].all_y.append(j)  #assign col to y
            shapes[shape_id].num_pts += 1     # increment px count

    # shape analysis (parallelize this)
    for i in range(num_comps):
        #check for dead components
        if shapes[i].num_pts < COMPONENT_THRESH:
            label_targets[i][0] = 0
            label_targets[i][1] = 0
            label_targets[i][2] = -1 # negative 1 denotes ignore shape
            continue

        # sample pixels for pinning label location
        samp_choices = list()
        for samp in range(PT_SURVAILENCE):
            # select representative pixels
            smpl_idex = int ( (samp + 0.5) * shapes[i].num_pts / PT_SURVAILENCE)
            test_x = shapes[i].all_x[ smpl_idex ]
            test_y = shapes[i].all_y[ smpl_idex ]


            # discern the buffer available at the pixel
            radius = [10, 3, 10, 3] # rep pxl is bottom right corner of label
            buffer = [1, 1, 1, 1]   # up / down / right / left

            for bit in range(radius[0]):  # test up direction
                if (test_x - bit) >= 0:
                    #ref_img[test_x - bit][test_y] = [0,255,0]
                    if shapeMask[test_x][test_y] == shapeMask[test_x - bit][test_y]:
                        buffer[0] += 1

            for bit in range(radius[1]):  # test down direction
                if (test_x + bit) < shapeMask.shape[0]:
                    #ref_img[test_x + bit][test_y] = [0,255,0]
                    if shapeMask[test_x][test_y] == shapeMask[test_x + bit][test_y]:
                        buffer[1] += 1

            for bit in range(radius[2]):   # test right direction
                if (test_y + bit) < shapeMask.shape[1]:
                    #ref_img[test_x][test_y + bit] = [0,255,0]
                    if shapeMask[test_x][test_y] == shapeMask[test_x][test_y+bit]:
                        buffer[2] += 1

            for bit in range(radius[3]):   # test left direction
                if (test_y - bit) >= 0:
                    #ref_img[test_x][test_y - bit] = [0,255,0]
                    if shapeMask[test_x][test_y] == shapeMask[test_x][test_y-bit]:
                        buffer[3] += 1

            buff_area = (buffer[0]+buffer[1]) * (buffer[2]+buffer[3])
            samp_choices.append([test_x, test_y, buff_area])

            # get most favorable location
            best_area = samp_choices[0][2]
            best_px = samp_choices[0][0]
            best_py = samp_choices[0][1]

            for j in range(1,len(samp_choices)):
                if samp_choices[j][2] >= best_area:
                    best_px = samp_choices[j][0]
                    best_py = samp_choices[j][1]


            # fit closest crayon
            samp_color = ref_img[test_x][test_y]
            #print(samp_color)
            #print('\t')

            # convert single pixel to LAB space
            cnvtr = ColorEntry((0,0,0), 0x0, 0, "dummy")
            samp_lab = cnvtr.rgb2lab((samp_color[2], samp_color[1], samp_color[0]))
            #print("C2N: Color sample: " + str(samp_lab))
            # identify closest crayon
            samp_num = crayons.color2number((samp_lab[0], samp_lab[1], samp_lab[2]))
            #print(samp_num)

            # write out the label info
            label_targets[i][0] = best_py  # col placement
            label_targets[i][1] = best_px  # row placement
            label_targets[i][2] = samp_num # number for label

    return label_targets

# Applying the label list as text on top of the unlabeled template
def applyNumberLabels(template, label_targets):
    for trgt_lab in label_targets:
        if trgt_lab[2] > 0:
            corner = (int(trgt_lab[0]), int(trgt_lab[1]))
            template = cv.putText(template, str(int(trgt_lab[2])), corner, cv.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,255), 1)

    return template


# DATA FOR SINGLE COLOR OBJECT
class ColorEntry:
    def __init__(self, rgb, hex, name, number):
        self.rgb = rgb
        self.hex = hex
        self.lab = self.rgb2lab(rgb)
        self.name = name
        self.number = number

    # returns LAB distance of this crayon from the reference value
    def distance(self, target):
        comp_val = target
        if type(target) is not tuple:
            comp_val = self.hex2rgb(target)
            comp_val = self.rgb2lab(comp_val)

        dist_r = self.lab[LAB_L] - comp_val[LAB_L]
        dist_g = self.lab[LAB_A] - comp_val[LAB_A]
        dist_b = self.lab[LAB_B] - comp_val[LAB_B]

        sqr_dist = (dist_r * dist_r) + (dist_b * dist_b) + (dist_g * dist_g)
        return math.sqrt(sqr_dist)


    # convert RGB pixel to LAB pixel
    def rgb2lab(self, t_rgb):
       num = 0
       RGB = [0, 0, 0]

       for value in t_rgb :
           value = float(value) / 255

           if value > 0.04045 :
               value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
           else :
               value = value / 12.92

           RGB[num] = value * 100
           num = num + 1

       XYZ = [0, 0, 0,]
       X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
       Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
       Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
       XYZ[ 0 ] = round( X, 4 )
       XYZ[ 1 ] = round( Y, 4 )
       XYZ[ 2 ] = round( Z, 4 )

       XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
       XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
       XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

       num = 0
       for value in XYZ :
           if value > 0.008856 :
               value = value ** ( 0.3333333333333333 )
           else :
               value = ( 7.787 * value ) + ( 16 / 116 )
           XYZ[num] = value
           num = num + 1

       Lab = [0, 0, 0]

       L = ( 116 * XYZ[ 1 ] ) - 16
       a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
       b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

       Lab [ 0 ] = round( L, 4 )
       Lab [ 1 ] = round( a, 4 )
       Lab [ 2 ] = round( b, 4 )

       return Lab


    # convert HEX code to RGB pixel
    def hex2rgb(self, t_hex):
        iso_r = (0xFF0000 & t_hex) >> 16
        iso_g = (0XFF00 & t_hex) >> 8
        iso_b = 0xFF & t_hex
        #print("HEX: " + hex(t_hex) + " R:"+hex(iso_r)+" G:"+hex(iso_g)+" B:"+hex(iso_b))
        return (iso_r, iso_g, iso_b)

    # convert RGB pixel to hex code
    def rgb2hex(self, t_rgb):
        hex_r = t_rgb[RGB_RED] << 16
        hex_g = (t_rgb[RGB_GREEN] << 8) & 0xFF00
        hex_b = t_rgb & 0xFF

        hex_val = hex_g | hex_b | hex_r
        return hex_val


# ENTIRE COLOR PACK
class ColorPack:
    color_set = []

    def __init__(self, fpath):
        entries = None
        with open(fpath) as f:
            entries = f.readlines()

        entries.pop(0)
        for entry in entries:
            #print(entry)
            info = entry.split(",")
            ce_num = int(info[0])
            ce_hex = int(info[1].strip().strip("#").strip(","), 16)
            ce_rgb = ColorEntry.hex2rgb(self, ce_hex)
            ce_name = info[5].strip()
            self.color_set.append(ColorEntry(ce_rgb, ce_hex, ce_name, ce_num))


    # retreive english name of color by number
    def num2name(self, num):
        for col_num in self.color_set:
            if col_num.number == num:
                return col_num.name
        return None


    # retreive number of english name color
    def name2num(self, name):
        for col_name in self.color_set:
            if col_name.name == name:
                return col_name.number
        return None


    # number of available colors in set
    def packSize(self):
        return len(self.color_set)

    # Flattens the crayons into a 1D (stride=3) array for GPU readin
    def flatten_lab_data(self):
        out_arr = np.zeros(self.packSize() *  3)

        for c in range(len(self.color_set)):
            out_arr[c*3] = self.color_set[c].lab[LAB_L]
            out_arr[(c*3)+1] = self.color_set[c].lab[LAB_A]
            out_arr[(c*3)+2] = self.color_set[c].lab[LAB_B]

        return out_arr

    # Flattens the crayons int a 1D (stride 3) array for GPU readin
    def flatten_bgr_data(self):
        out_arr = np.zeros(self.packSize() *  3)

        for c in range(len(self.color_set)):
            out_arr[c*3] = self.color_set[c].rgb[RGB_BLUE]
            out_arr[(c*3)+1] = self.color_set[c].rgb[RGB_GREEN]
            out_arr[(c*3)+2] = self.color_set[c].rgb[RGB_RED]

        return out_arr

    # Color To Number Selection Operation (.distance() handles col. space conversion)
    def color2number(self, target):
        # There's probably a threshold for which this is worth parallelizing
        # 8 pk is fast on CPU, 128 pk might be better in parallel
        best_dist = 442 # No value in RGB space will be farther than this
        best_num = 0

        for test in self.color_set:
            res = test.distance(target)

            if res < best_dist :
                best_dist = res
                best_num = test.number

        return best_num


# TEST BATTERY FOR COLORSPACE COMPARISONS
if __name__ == "__main__":
    print("TEST - Color to Number Pack functionality")
    colors = ColorPack("color_packs/crayola_22pk.txt")

    for color in colors.color_set:
        print(color.name + " 0x" + str(color.hex) + "LAB: " + str())

    """test_battery = [[0xFA8072, "Salmon"],
                    [0xFA8072, "Yellow Green"],
                    []]"""


    """print("VALIDATION: ")
    for cols in colors.color_set:
        print("Color Pack Entry: ", cols.number, cols.name, hex(cols.hex), cols.rgb, cols.lab)

    """
    for cols in colors.color_set:
        print("Color Pack Entry: ", cols.number, cols.name, hex(cols.hex), cols.rgb, cols.lab)
    salmon_res = colors.color2number(0xFA8072)
    yellowgreen_res = colors.color2number(0xADFF2F)
    darkgray_res = colors.color2number(0xA9A9A9)
    deepskyblue_res = colors.color2number(0x00BFFF)
    darkyellow_res = colors.color2number(0xf2c634)
    purple_res = colors.color2number(0xae2ab5)

    print("Test [Salmon #FA8072] -> #" + str(salmon_res) + " " + \
           colors.num2name(salmon_res) + " LAB deltaE (dist):" + \
           str(colors.color_set[salmon_res-1].distance(0xFA8072)))
    print("Test [Yellow Green #ADFF2F] -> #" + str(yellowgreen_res) + " " + \
           colors.num2name(yellowgreen_res) + " LAB deltaE (dist):" + \
           str(colors.color_set[yellowgreen_res-1].distance(0xADFF2F)))
    print("Test [Dark Gray #A9A9A9] -> #" + str(darkgray_res) + " " + \
           colors.num2name(darkgray_res) + " LAB deltaE (dist):" + \
           str(colors.color_set[darkgray_res-1].distance(0xA9A9A9)))
    print("Test [Sky Blue #00BFFF] -> #" + str(deepskyblue_res) + " " + \
           colors.num2name(deepskyblue_res) + " LAB deltaE (dist):" + \
           str(colors.color_set[deepskyblue_res-1].distance(0x00BFFF)))
    print("Test [Dark Yellow #f2c634] -> #" + str(darkyellow_res) + " " + \
           colors.num2name(darkyellow_res) + " LAB deltaE (dist):" + \
           str(colors.color_set[darkyellow_res-1].distance(0xf2c634)))
    print("Test [Purple #ae2ab5] -> #" + str(purple_res) + " " + \
           colors.num2name(purple_res) + " LAB deltaE (dist):" + \
           str(colors.color_set[purple_res-1].distance(0xae2ab5)))


    print("Test [Dark Yellow #f2c634] -> #" + str(darkyellow_res) + " " + \
           colors.num2name(darkyellow_res) + " LAB deltaE (dist):" + \
           str(colors.color_set[darkyellow_res-1].distance(0xf2c634)))

    print("Test [Purple #ae2ab5] -> #" + str(purple_res) + " " + \
           colors.num2name(purple_res) + " LAB deltaE (dist):" + \
           str(colors.color_set[purple_res-1].distance(0xae2ab5)))

    # print("Salmon -> pink d:", colors.color_set[15].distance(0xFA8072))
