"""
Main pipeline for processing images.

Command options.
-Image="Image_string" (input image)
-Aspect="AxB"         (set a desired output size)
-Verbose              (turn on verbose mode)
-ColorPack="cp_string"(define location of the color pack)
-RA="R_idxXA_idx"     (sets resolution x aspect ratio [supersceded by Aspect])

New arg syntax -arg_name[="argv_str"] (sorry its not true linux syntax)
*if the flag simply sets a value (like turning verbosity on), then use -arg
*if the flag requires more data (like setting a file path), then use -arg=data
*if the flag requires several datapoints (like w x h), the use -arg="dat1[delim]dat2..."
"""

# ToDO: BKL - fix the namespaces so class calls are cleaner
import sys
import Preprocess
import ColorPack

class MetaData:
    #attributes set as default
    color_src = "color_packs/crayola_22pk.txt"
    img_src = None
    verbose = 0
    img_h = 0
    img_w = 0
    res = Preprocess.RES_1080p
    asp = Pre

    def __init__(self, args):
        args.pop(0) #remove the name of the py file
        for arg in args:
            arg_tok = arg.split("=")

            if (arg_tok[0] == "-Image"):
                self.img_src = arg_tok[1]

            elif (arg_tok[0] == "-Verbose"):
                self.verbose = 1

            elif (arg_tok[0] == "-ColorPack"):
                self.color_src = arg_tok[1]

            elif (arg_tok[0] == "-Aspect"):
                wh_tok = arg_tok[1].split("x")
                self.img_w = int(wh_tok[0])
                self.img_h = int(wh_tok[1])

            elif (arg_tok[0] == "-RA"):
                ra_tok = arg_tok[1].split("x")
                self.res = int(ra_tok[0])
                self.asp = int(ra_tok[1])

            else:
                print("What's this arg? -> " + arg)


# Or make into a main function or class
if __name__ == "__main__":
    print("Paint By Numbers Generator\n")

    # decode Cmd line Args
    md = MetaData(sys.argv)

    # intake the color pack
    col_pack = ColorPack.ColorPack(md.color_src)

    # set up the dimensions of the image based on provided dimensions
    if md.img_w == 0 or md.img_h == 0 :
        img_data = Preprocess.InputImage(md.res, md.asp)
    else:
        img_data = Preprocess.InputImage(0, 0, screen=(md.width, md.height))

    if md.img_src == None:
        print("You need to specify an image to get a template.")
        exit() #or return if repacked into a def

    # intake image and fit.
    img_data.set_image(md.img_src) # can be recalled to adjust interpolation

# INSERT THE CV STEPS HERE
    print("Ready to process.\n")
    print("Parameter Check:\nColor Package:" + md.color_src + \
          "\nImage File:" + str(md.img_src) + "\nImage W x H: " + \
          str(img_data.width) + " x " + str(img_data.height) + "\nVerbosity: " + \
          str(md.verbose))

    print("\nJust kidding ;) bye.")
