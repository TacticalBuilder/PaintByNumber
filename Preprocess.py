"""
The InputImage Object holds one image which has been sized to the desired
dimensions and resolution

The target image dimensions are automatically calculated from a resolution
   and aspect ratio, or the width and height selected directly

Resolution options:
    - RES_720p
    - RES_1080p
    - RES_4K

Aspect Ratio options:
     - ASP_4_3  - common for phone cameras
     - ASP_3_2  - common for DSLR cameras
     - ASP_1_1  - square images
     - ASP_16_9 - widescreen dimension
"""
import cv2 as cv
import math

# AVAILABLE RESOLUATIONS (Mpx  or common screeen images)
RES_720p = 1
RES_1080p = 2
RES_4K = 3

# AVAILABLE ASPECT RATIOS
ASP_4_3 = 1     # Common for Phone cameras
ASP_3_2 = 2     # Common for DSLR cameras
ASP_1_1 = 3     # Square images
ASP_16_9 = 4    # widescreen aspect ratio

class ImVal():
    resln = [0, 921600, 2073600, 8847360]
    ratio = [[0, 0], [4, 3], [3, 2], [1, 1], [16, 9]] # [W, H]

class InputImage:
    width = 0       # width of image to be fed in as input
    height  = 0     # height of image to be fed in as input
    img_fpath = None            # file path to source image
    img_dat = None              # image object

    def __init__(self, res, aspect, screen=None):
        #set the image directly if specified
        if screen is not None:
            if type(screen) is tuple:
                self.width = screen[0]
                self.height = screen[1]
            return None

        # calculate the image dimensions
        base = math.sqrt(ImVal.resln[res] / (ImVal.ratio[aspect][0] * ImVal.ratio[aspect][1]))
        self.width = int(ImVal.ratio[aspect][0] * base)
        self.height = int(ImVal.ratio[aspect][1] * base)


    def set_image(self, fpath, interp=cv.INTER_NEAREST):
        file_src = self.img_fpath
        if fpath is not None:
            file_src = fpath
            self.img_fpath = fpath

        src_img = cv.imread(file_src, cv.IMREAD_UNCHANGED)
        dim = (self.width, self.height)

        rsz = cv.resize(src_img, dim, interpolation=interp)
        self.img_dat = rsz


    def save_image(self, fpath=None):
        dest_img = self.img_fpath
        if fpath is None:
            dest_img = fpath
        cv.imwrite(dest_img, self.img_dat)



if __name__ == "__main__":
    # TESTING RESOLUTION MAPPING
    res_sz = RES_720p
    my_imgs = [InputImage(res_sz, ASP_4_3),
               InputImage(res_sz, ASP_1_1),
               InputImage(res_sz, ASP_3_2),
               InputImage(res_sz, ASP_16_9)]

    names = ["4:3", "1:1", "3:2", "16:9"]
    print("Test image preprocessor: RES SZ: " + str(ImVal.resln[res_sz]))
    for i in range(len(my_imgs)):
        wi = my_imgs[i].width
        hi = my_imgs[i].height
        diff = abs((wi*hi) - ImVal.resln[res_sz])
        err = (diff / ImVal.resln[res_sz]) * 100
        print("Aspect ratio ["+names[i]+"] W:"+str(wi)+" H:"+str(hi)+" T:"+str(wi*hi)+" E:"+str(err)+"%")

    #TESTING DOWNSCALING
    orig_img = cv.imread("silly_source.jpg")
    cv.imshow('test image', orig_img)
    cv.waitKey(0)

    fit_test = InputImage(RES_720p, ASP_16_9)
    fit_test.set_image("silly_source.jpg");

    print("Down-sampled image dim: ", fit_test.img_dat.shape)
    cv.imshow('test image', fit_test.img_dat)
    cv.waitKey(0)

    #TESTING UPSCALING
    orig_img = cv.imread("silly_source2.jpg")
    cv.imshow('test image', orig_img)
    cv.waitKey(0)

    fit_test = InputImage(RES_1080p, ASP_16_9)
    fit_test.set_image("silly_source2.jpg");

    print("Up-sampled image dim: ", fit_test.img_dat.shape)
    cv.imshow('test image', fit_test.img_dat)
    cv.waitKey(0)
    cv.destroyAllWindows()
