from .filesUtils import *
import math
from PIL import ImageChops
import PIL as pil
import torchvision.transforms as transforms
# from icecream import ic
#*--------------------------------------------------------------------------------------------------*#
#* FileName: imgUtils.py
#* Last Modified: 2023-12-17
#* This is the 2D image utils libs to process the img operation like crop, resize, get diff image etc.
#*--------------------------------------------------------------------------------------------------*#
ToTensor = transforms.ToTensor() # HxWxC -> CxHxW, [0,255] -> [0,1]
ToPILImage = transforms.ToPILImage() # CxHxW -> HxWxC, [0,1] -> [0,255]

def CHW2HWC(img:torch.Tensor):
    return img.permute(1,2,0)

def HWC2CHW(img:torch.Tensor):
    return img.permute(2,0,1)

def readImg(imgPath):
    img_pil = pil.Image.open(imgPath)
    return ToTensor(img_pil)

def saveImg(img:torch.Tensor,savePath):
    img_pil = ToPILImage(img)
    img_pil.save(savePath)
    
def viewImg(img:torch.Tensor):
    img_pil = ToPILImage(img)
    img_pil.show()
        
def viewImage(img,fromtype='img'):
    if fromtype == 'img':
        img.show()
    elif fromtype == 'numpy':
        img = pil.Image.fromarray(img)
        img.show()
    elif fromtype == 'tensor':
        img = img.cpu().numpy()
        img = pil.Image.fromarray(img)
        img.show()
    else:
        raise ValueError(f'fromtype {fromtype} is not supported')

def RGBA2RGB(img,whiteBG=False):
    if type(img) == str:
        img = readImg(img)
    elif type(img) == torch.Tensor:
        pass
    else:
        raise ValueError(f'input type {type(img)} is not supported')
    
    if whiteBG:
        imgH,imgW = img.shape[1],img.shape[2]
        img = img.view(4,-1).permute(1,0)
        img = img[:,:3]*img[:,-1:] + (1 - img[:,-1:])
        img = img.permute(1,0).view(3,imgH,imgW)
    else:
        img = img[:3,:,:]
    return img


def convert(folder,type1,type2):
    jpg = []
    png = []
    for r,d,f in os.walk(folder):
        for file in f:
            if type1 in file:
                jpg.append(os.path.join(r,file))
    for image in jpg:
        png.append(image.replace(type1,type2))
    for i in range(0,len(jpg)):
        cmd = 'convert '+jpg[i]+' '+png[i]
        subprocess.call(cmd,shell=True)

def ImgCrop(ImgDir,img_type,size,pos):
    """
    Description:
    Batch operation to crop all images (with img_type) in the ImgDir folder

    Args:
        ImgDir (str): the Image Dir path to be cropped
        img_type (str): specify the image type to be cropped
        size (list): The crop region size
        pos (list): The crop region position (left-up corner)
    Returns:
        None
    """
    i1 = []
    for r,d,f in os.walk(ImgDir):
        for file in f:
            if img_type in file:
                i1.append(os.path.join(r,file))
    for i in range(0,len(i1)):
        #print(i1[i])
        cmd = 'convert '+i1[i]+' -crop '+str(size[0])+'x'+str(size[1])+'+'+str(pos[0])+'+'+str(pos[1])+' '+i1[i]
        #print(cmd)
        subprocess.call(cmd,shell=True)

def ImgAutoTrim(ImgDir,log=True):
    from wand.image import Image
    img_paths = getFilePathsInDir(ImgDir)
    global_bbox = (np.inf,np.inf,0,0)
    for img_p in img_paths:
        if 'txt' in img_p:
            continue
        img = Image.open(img_p)
        bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
        diff = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        global_bbox = (min(global_bbox[0],bbox[0]),min(global_bbox[1],bbox[1]),max(global_bbox[2],bbox[2]),max(global_bbox[3],bbox[3]))
    pos = (global_bbox[0],global_bbox[1])
    size = (global_bbox[2]-global_bbox[0],global_bbox[3]-global_bbox[1])
    if log:
        with open(os.path.join(ImgDir,'bbox.txt'),'w') as f:
            f.write("bbox: "+str(global_bbox)+'\n')
            f.write("size: "+str(size)+'\n')
            f.write("pos: "+str(pos))
    ImgCrop(ImgDir,size,pos)


def ImgResize(ImgDir,type1,type2):
    """
    Description:
    Batch operation to resize all images (with img_type) in the ImgDir folder to a dimension of 256x256

    Args:
        ImgDir (str): the Image Dir path to be cropped
        img_type (str): specify the image type to be cropped
        size (list): The crop region size
        pos (list): The crop region position (left-up corner)
    Returns:
        None
    """
    jpg = []
    tga = []
    for r,d,f in os.walk(ImgDir):
        for file in f:
            if type1 in file:
                jpg.append(os.path.join(r,file))
    for image in jpg:
        tga.append(image.replace(type1,type2))
    for i in range(0,len(jpg)):
        cmd = 'convert '+jpg[i]+' -resize 256x256 '+tga[i]
        # print(cmd)
        # exit()
        subprocess.call(cmd,shell=True)
        
def getDiffImgFromDir(dirPath,GTFileName='GT.png'):
    imgExt = os.path.splitext(GTFileName)[-1]
    paths = getFilePathsInDir(dirPath,ext=imgExt)
    GTImgPaths = os.path.join(dirPath,GTFileName)
    for path in paths:
        if(path == GTFileName):continue
        diff_img = get_diff_img(GTImgPaths,path)
        diff_img.format='png'
        saveFileName = os.path.splitext(path)[0]+f'-diff.png'
        diff_img.save(filename=saveFileName)

def getDiffImgDir(dirPathEval,dirPathGT,DiffSaveDir):
    EvalPaths = sorted(getFilePathsInDir(dirPathEval))
    GTPaths = sorted(getFilePathsInDir(dirPathGT))
    for idx,evalPath in enumerate(EvalPaths):
        diff_img = get_diff_img(GTPaths[idx], evalPath)
        diff_img.format = 'png'
        saveFileName = os.path.splitext(evalPath)[0] + f'-diff.png'
        saveFileName = os.path.join(DiffSaveDir,parseFNfromP(saveFileName))
        diff_img.save(filename=saveFileName)


def get_deltae_color_a(step):
    if step < 0:
        step = 0
    if step > 255:
        step = 255
    
    if step <= 5:
        r = 0.0
        g = 0.0
        b = 0.0
    elif step <= 11:
        r = 0.75 - (step-5) * 0.75 / 6.0
        g = 0.375 - (step-5) * 0.375 / 6.0
        b = 1.0
    elif step <= 19:
        r = 0.0
        g = (step-11) / 8.0
        b = 1.0
    elif step <= 27:
        r = 0.0
        g = 1.0
        b = 1.0 - (step-19) / 8.0
    elif step <= 37:
        r = (step-27)/10.0
        g = 1.0
        b = 0.0
    elif step <= 47:
        r = 1.0
        g = 1.0 - (step-37) * 0.5 / 10.0
        b = 0.0
    elif step <= 255:
        r = 1.0
        g = 0.5 - (step-47) * 0.5 / 208.0
        b = 0.0
    
    return (r,g,b)

def get_diff_img(path1, path2):
    from wand.image import Image
    img1 = np.array(Image(filename=path1))
    img2 = np.array(Image(filename=path2))

    dim = img1.shape

    diff_img = np.zeros((dim[0],dim[1],3), dtype=np.uint8)

    for x in range(dim[0]):
        for y in range(dim[1]):
            if len(img1[x][y]) == 3:
                r1,g1,b1 = img1[x][y]
                r2,g2,b2 = img2[x][y]
            if len(img1[x][y]) == 4:
                r1,g1,b1,a1 = img1[x][y]
                r2,g2,b2,a2 = img2[x][y]

            r1 = float(r1)
            g1 = float(g1)
            b1 = float(b1)
            r2 = float(r2)
            g2 = float(g2)
            b2 = float(b2)

            step = math.sqrt((r1-r2)*(r1-r2)+(g1-g2)*(g1-g2)+(b1-b2)*(b1-b2))/7.5
            

            r,g,b = get_deltae_color_a(step)

            diff_img[x][y] = [int(r*255),int(g*255),int(b*255)]

            if r==0 and g==0 and b==0:
                diff_img[x][y] = [255,255,255]


    diff_img = Image.from_array(diff_img)
    return diff_img
