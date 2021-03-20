# Color Pack Object
"""
This object handles the 'crayons' that will be targetted by the program
Input is a csv of the colors, name, number, and RGB / HEX

|--ColorPack
    |--ColorEntry[]
"""
import math

RGB_RED = 0
RGB_BLUE = 1
RGB_GREEN = 2

# DATA FOR SINGLE COLOR OBJECT
class ColorEntry:
    def __init__(self, rgb, hex, name, number):
        self.rgb = rgb
        self.hex = hex
        self.name = name
        self.number = number

    def distance(self, target):
        comp_val = target
        if target is not tuple:
            comp_val = self.hex2rgb(target)

        dist_r = self.rgb[RGB_RED] - comp_val[RGB_RED]
        dist_g = self.rgb[RGB_GREEN] - comp_val[RGB_GREEN]
        dist_b = self.rgb[RGB_BLUE] - comp_val[RGB_BLUE]

        sqr_dist = (dist_r * dist_r) + (dist_b * dist_b) + (dist_g * dist_g)
        return math.sqrt(sqr_dist)


    def hex2rgb(self, t_hex):
        iso_r = (0xFF0000 & t_hex) >> 16
        iso_g = (0XFF00 & t_hex) >> 8
        iso_b = 0xFF & t_hex
        #print("HEX: " + hex(t_hex) + " R:"+hex(iso_r)+" G:"+hex(iso_g)+" B:"+hex(iso_b))
        return (iso_r, iso_b, iso_g)


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


    def num2name(self, num):
        for col_num in self.color_set:
            if col_num.number == num:
                return col_num.name
        return None


    def name2num(self, name):
        for col_name in self.color_set:
            if col_name.name == name:
                return col_name.number
        return None


    def color2number(self, target):
        # There's probably a threshold for which this is worth parallelizing
        # 8 pk is fast on CPU, 128 pk might be better in parallel
        best_dist = 442 # No value in RGB space will be farther than this
        best_num = 0
        for test in self.color_set:
            res = test.distance(target)
            #print(target, " "+test.name+" d:" + str(res))
            if res < best_dist :
                best_dist = res
                best_num = test.number

        return best_num


if __name__ == "__main__":
    print("TEST - Color to Number Pack functionality")
    colors = ColorPack("color_packs/crayola_22pk.txt")

    print("VALIDATION: ")
    for cols in colors.color_set:
        print("Color Pack Entry: ", cols.number, cols.name, hex(cols.hex), cols.rgb)

    salmon_res = colors.color2number(0xFA8072)
    yellowgreen_res = colors.color2number(0xADFF2F)
    darkgray_res = colors.color2number(0xA9A9A9)
    deepskyblue_res = colors.color2number(0x00BFFF)

    print("Test [Salmon #FA8072] -> #" + str(salmon_res) + " " + \
           colors.num2name(salmon_res) + " RGB dist:" + \
           str(colors.color_set[salmon_res-1].distance(0xFA8072)))

    print("Test [Yellow Green #ADFF2F] -> #" + str(yellowgreen_res) + " " + \
           colors.num2name(yellowgreen_res) + " RGB dist:" + \
           str(colors.color_set[yellowgreen_res-1].distance(0xADFF2F)))

    print("Test [Dark Gray #A9A9A9] -> #" + str(darkgray_res) + " " + \
           colors.num2name(darkgray_res) + " RGB dist:" + \
           str(colors.color_set[darkgray_res-1].distance(0xA9A9A9)))

    print("Test [Sky Blue #00BFFF] -> #" + str(deepskyblue_res) + " " + \
           colors.num2name(deepskyblue_res) + " RGB dist:" + \
           str(colors.color_set[deepskyblue_res-1].distance(0x00BFFF)))

    #print("Salmon -> pink d:", colors.color_set[15].distance(0xFA8072))
