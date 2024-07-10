import PySimpleGUI as sg
import cv2
import numpy as np
import copy


EVENTLER = ['-SMT-','-BLUR-','-GBLUR-','-MBLUR-','-BBLUR-','-BTRESH-','-BITRESH-','-TTRESH-','-ZTRESH-','-ZITRESH-','-ERO-','-DIL-','-OPE-','-CLO-','-MG-','-TH-','-BH-','-EI-','-DI-','-SMT-F','-BLUR-F','-GBLUR-F','-MBLUR-F','-BBLUR-F','-BTRESH-F','-BITRESH-F','-TTRESH-F','-ZTRESH-F','-ZITRESH-F','-MO-','-MOB-','-MOC-']
image_elem_org = sg.Image(filename=None, key='-IMGO-')
image_elem = sg.Image(filename=None, key='-IMG-')
image_sub = sg.Image(filename=None, key='-IMGS-')
iter_test_col = sg.Frame('Iter Test',[
    [sg.Text('Erosion Iteration'),sg.Slider(range=(1,10),orientation='h',key='-EI-',enable_events=True)],
    [sg.Text('Dilation Iteration'),sg.Slider(range=(1,10),orientation='h',key='-DI-',enable_events=True)],
])
smoot_col = sg.Frame('Smoothing',[
    [sg.Text('2D Convolution'),sg.Slider(range=(0,10),orientation='h',key='-SMT-',enable_events=True)],
    [sg.Text('Image Blurring'),sg.Slider(range=(0,10),orientation='h',key='-BLUR-',enable_events=True)],
    [sg.Text('Gaussian Blurring'),sg.Slider(range=(0,10),orientation='h',key='-GBLUR-',enable_events=True)],
    [sg.Text('Median Blurring'),sg.Slider(range=(0,10),orientation='h',key='-MBLUR-',enable_events=True)],
    [sg.Text('Bilateral Filtering'),sg.Slider(range=(0,10),orientation='h',key='-BBLUR-',enable_events=True)],
])
adap_thres_col=sg.Frame('Adaptive Thresholding',[
    [sg.Text('Mean'),sg.Radio('ON','MEAN',key='-MO-',enable_events=True),sg.Radio('OFF','MEAN'),sg.Text('Block Size'),sg.Slider(range=(0,20),orientation='h',key='-MOB-',enable_events=True),sg.Text('Constant'),sg.Slider(range=(0,20),orientation='h',key='-MOC-',enable_events=True)],
    [],
    [],
    [sg.Text('Gaussian'),sg.Radio('ON','GAUSS',key='-GO-',visible=False)]
])
thres_col = sg.Frame('Tresholding',[
    [sg.Text('Binary'),sg.Slider(range=(0,255),orientation='h',key='-BTRESH-',enable_events=True,)],
    [sg.Text('Binary INV'),sg.Slider(range=(0,255),orientation='h',key='-BITRESH-',enable_events=True)],
    [sg.Text('Trunc'),sg.Slider(range=(0,255),orientation='h',key='-TTRESH-',enable_events=True)],
    [sg.Text('TOZERO'),sg.Slider(range=(0,255),orientation='h',key='-ZTRESH-',enable_events=True)],
    [sg.Text('TOZERO INV'),sg.Slider(range=(0,255),orientation='h',key='-ZITRESH-',enable_events=True)],
])

morph_col = sg.Frame('Morphological Transformation',[
    [sg.Text('Erosion'),sg.Slider(range=(0,10),orientation='h',key='-ERO-',enable_events=True)],
    [sg.Text('Dilation'),sg.Slider(range=(0,10),orientation='h',key='-DIL-',enable_events=True)],
    [sg.Text('Opening'),sg.Slider(range=(0,10),orientation='h',key='-OPE-',enable_events=True)],
    [sg.Text('Closing'),sg.Slider(range=(0,10),orientation='h',key='-CLO-',enable_events=True)],
    [sg.Text('Morphological Gradient'),sg.Slider(range=(0,10),orientation='h',key='-MG-',enable_events=True)],
    [sg.Text('Top Hat'),sg.Slider(range=(0,10),orientation='h',key='-TH-',enable_events=True)],
    [sg.Text('Black Hat'),sg.Slider(range=(0,10),orientation='h',key='-BH-',enable_events=True)],
])

smoot_f_col = sg.Frame('Smoothing Final',[
    [sg.Text('2D Convolution'),sg.Slider(range=(0,10),orientation='h',key='-SMT-F',enable_events=True)],
    [sg.Text('Image Blurring'),sg.Slider(range=(0,10),orientation='h',key='-BLUR-F',enable_events=True)],
    [sg.Text('Gaussian Blurring'),sg.Slider(range=(0,10),orientation='h',key='-GBLUR-F',enable_events=True)],
    [sg.Text('Median Blurring'),sg.Slider(range=(0,10),orientation='h',key='-MBLUR-F',enable_events=True)],
    [sg.Text('Bilateral Filtering'),sg.Slider(range=(0,10),orientation='h',key='-BBLUR-F',enable_events=True)],
])

thres_f_col = sg.Frame('Tresholding Final',[
    [sg.Text('Binary'),sg.Slider(range=(0,255),orientation='h',key='-BTRESH-F',enable_events=True)],
    [sg.Text('Binary INV'),sg.Slider(range=(0,255),orientation='h',key='-BITRESH-F',enable_events=True)],
    [sg.Text('Trunc'),sg.Slider(range=(0,255),orientation='h',key='-TTRESH-F',enable_events=True)],
    [sg.Text('TOZERO'),sg.Slider(range=(0,255),orientation='h',key='-ZTRESH-F',enable_events=True)],
    [sg.Text('TOZERO INV'),sg.Slider(range=(0,255),orientation='h',key='-ZITRESH-F',enable_events=True)],
])

layout=[
    [image_elem_org,image_elem,image_sub],
    [],
    [[sg.Button('Browse'),sg.Button('Reset',key='-RST-'),sg.Button('Save',key='-SAVE-')]],
    [smoot_col,thres_col,morph_col,iter_test_col,smoot_f_col,thres_f_col],
    [adap_thres_col]
]

window = sg.Window('FolyoGör Test',layout,grab_anywhere = False,finalize=True)
window.maximize()
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840) # 4k/high_res
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160) # 4k/high_res


while True:
    event, values = window.read(timeout=20)


    ret, frame = cap.read()
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto

    # Orjinal görüntü
    nparr = np.fromstring(imgbytes, np.uint8)
    img_o = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    #img_o = cv2.resize(img_o,[int(3840/6),int(2160/6)])
    o_imgbytes = cv2.imencode('.png', img_o)[1].tobytes()
    window['-IMGO-'].update(data=o_imgbytes)
    img = copy.deepcopy(img_o)

    if values['-SMT-'] != 0:
        kernel = np.ones((int(values['-SMT-']),int(values['-SMT-'])),np.float32)/(int(values['-SMT-'])*int(values['-SMT-']))
        img = cv2.filter2D(img,-1,kernel)
    if values['-BLUR-'] != 0: img = cv2.blur(img,(int(values['-BLUR-']),int(values['-BLUR-'])))
    if values['-GBLUR-'] != 0:
        if values['-GBLUR-'] % 2 != 0:
            img = cv2.GaussianBlur(img,(int(values['-GBLUR-']),int(values['-GBLUR-'])),0)
    if values['-MBLUR-'] != 0: 
        if values['-MBLUR-'] % 2 != 0:
            img = cv2.medianBlur(img,int(values['-MBLUR-']))
    if values['-BBLUR-'] != 0: img = cv2.bilateralFilter(img,int(values['-BBLUR-']),75,75)
    if values['-BTRESH-'] != 0: ret,img = cv2.threshold(img,int(values['-BTRESH-']),255,cv2.THRESH_BINARY)
    if values['-BITRESH-'] != 0: ret,img = cv2.threshold(img,int(values['-BITRESH-']),255,cv2.THRESH_BINARY_INV)
    if values['-TTRESH-'] != 0: ret,img = cv2.threshold(img,int(values['-TTRESH-']),255,cv2.THRESH_TRUNC)
    if values['-ZTRESH-'] != 0: ret,img = cv2.threshold(img,int(values['-ZTRESH-']),255,cv2.THRESH_TOZERO)
    if values['-ZITRESH-'] != 0: ret,img = cv2.threshold(img,int(values['-ZITRESH-']),255,cv2.THRESH_TOZERO_INV)

    if values['-ERO-'] != 0:
        kernel = np.ones((int(values['-ERO-']),int(values['-ERO-'])),np.uint8)
        img = cv2.erode(img,kernel,iterations=int(values['-EI-']))
    if values['-DIL-'] != 0:
        kernel = np.ones((int(values['-DIL-']),int(values['-DIL-'])),np.uint8)
        img = cv2.dilate(img,kernel,iterations=int(values['-DI-']))
    if values['-OPE-'] != 0:
        kernel = np.ones((int(values['-OPE-']),int(values['-OPE-'])),np.uint8)
        img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    if values['-CLO-'] != 0:
        kernel = np.ones((int(values['-CLO-']),int(values['-CLO-'])),np.uint8)
        img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    if values['-MG-'] != 0:
        kernel = np.ones((int(values['-MG-']),int(values['-MG-'])),np.uint8)
        img = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
    if values['-TH-'] != 0:
        kernel = np.ones((int(values['-TH-']),int(values['-TH-'])),np.uint8)
        img = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
    if values['-BH-'] != 0:
        kernel = np.ones((int(values['-BH-']),int(values['-BH-'])),np.uint8)
        img = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
    if values['-MO-'] is True:
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,int(values['-MOB-']),int(values['-MOC-']))
    m_imgbytes = cv2.imencode('.png', img)[1].tobytes()  
    window['-IMG-'].update(data=m_imgbytes)
    img_s = cv2.subtract(img_o,img)

    if values['-SMT-F'] != 0:
        kernel = np.ones((int(values['-SMT-F']),int(values['-SMT-F'])),np.float32)/(int(values['-SMT-F'])*int(values['-SMT-F']))
        img_s = cv2.filter2D(img_s,-1,kernel)
    if values['-BLUR-F'] != 0: img_s = cv2.blur(img_s,(int(values['-BLUR-F']),int(values['-BLUR-F'])))
    if values['-GBLUR-F'] != 0:
        if values['-GBLUR-F'] % 2 != 0:
            img_s = cv2.GaussianBlur(img_s,(int(values['-GBLUR-F']),int(values['-GBLUR-F'])),0)
    if values['-MBLUR-F'] != 0: 
        if values['-MBLUR-F'] % 2 != 0:
            img_s = cv2.medianBlur(img_s,int(values['-MBLUR-F']))
    if values['-BBLUR-F'] != 0: img_s = cv2.bilateralFilter(img_s,int(values['-BBLUR-F']),75,75)
    if values['-BTRESH-F'] != 0: ret,img_s = cv2.threshold(img_s,int(values['-BTRESH-F']),255,cv2.THRESH_BINARY)
    if values['-BITRESH-F'] != 0: ret,img_s = cv2.threshold(img_s,int(values['-BITRESH-F']),255,cv2.THRESH_BINARY_INV)
    if values['-TTRESH-F'] != 0: ret,img_s = cv2.threshold(img_s,int(values['-TTRESH-F']),255,cv2.THRESH_TRUNC)
    if values['-ZTRESH-F'] != 0: ret,img_s = cv2.threshold(img_s,int(values['-ZTRESH-F']),255,cv2.THRESH_TOZERO)
    if values['-ZITRESH-F'] != 0: ret,img_s = cv2.threshold(img_s,int(values['-ZITRESH-F']),255,cv2.THRESH_TOZERO_INV)

    imgbytes_s = cv2.imencode('.png', img_s)[1].tobytes()
    window['-IMGS-'].update(data=imgbytes_s)

    if event == '-SAVE-':

        nparr = np.fromstring(imgbytes, np.uint8)
        img_o = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        o_imgbytes = cv2.imencode('.png', img_o)[1].tobytes()
        img = copy.deepcopy(img_o)

        if values['-SMT-'] != 0:
            kernel = np.ones((int(values['-SMT-']),int(values['-SMT-'])),np.float32)/(int(values['-SMT-'])*int(values['-SMT-']))
            img = cv2.filter2D(img,-1,kernel)
        if values['-BLUR-'] != 0: img = cv2.blur(img,(int(values['-BLUR-']),int(values['-BLUR-'])))
        if values['-GBLUR-'] != 0:
            if values['-GBLUR-'] % 2 != 0:
                img = cv2.GaussianBlur(img,(int(values['-GBLUR-']),int(values['-GBLUR-'])),0)
        if values['-MBLUR-'] != 0: 
            if values['-MBLUR-'] % 2 != 0:
                img = cv2.medianBlur(img,int(values['-MBLUR-']))
        if values['-BBLUR-'] != 0: img = cv2.bilateralFilter(img,int(values['-BBLUR-']),75,75)
        if values['-BTRESH-'] != 0: ret,img = cv2.threshold(img,int(values['-BTRESH-']),255,cv2.THRESH_BINARY)
        if values['-BITRESH-'] != 0: ret,img = cv2.threshold(img,int(values['-BITRESH-']),255,cv2.THRESH_BINARY_INV)
        if values['-TTRESH-'] != 0: ret,img = cv2.threshold(img,int(values['-TTRESH-']),255,cv2.THRESH_TRUNC)
        if values['-ZTRESH-'] != 0: ret,img = cv2.threshold(img,int(values['-ZTRESH-']),255,cv2.THRESH_TOZERO)
        if values['-ZITRESH-'] != 0: ret,img = cv2.threshold(img,int(values['-ZITRESH-']),255,cv2.THRESH_TOZERO_INV)

        if values['-ERO-'] != 0:
            kernel = np.ones((int(values['-ERO-']),int(values['-ERO-'])),np.uint8)
            img = cv2.erode(img,kernel,iterations=int(values['-EI-']))
        if values['-DIL-'] != 0:
            kernel = np.ones((int(values['-DIL-']),int(values['-DIL-'])),np.uint8)
            img = cv2.dilate(img,kernel,iterations=int(values['-DI-']))
        if values['-OPE-'] != 0:
            kernel = np.ones((int(values['-OPE-']),int(values['-OPE-'])),np.uint8)
            img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
        if values['-CLO-'] != 0:
            kernel = np.ones((int(values['-CLO-']),int(values['-CLO-'])),np.uint8)
            img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
        if values['-MG-'] != 0:
            kernel = np.ones((int(values['-MG-']),int(values['-MG-'])),np.uint8)
            img = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
        if values['-TH-'] != 0:
            kernel = np.ones((int(values['-TH-']),int(values['-TH-'])),np.uint8)
            img = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
        if values['-BH-'] != 0:
            kernel = np.ones((int(values['-BH-']),int(values['-BH-'])),np.uint8)
            img = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
        if values['-MO-'] is True:
            img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,int(values['-MOB-']),int(values['-MOC-']))
        m_imgbytes = cv2.imencode('.png', img)[1].tobytes()  
        img_s = cv2.subtract(img_o,img)

        if values['-SMT-F'] != 0:
            kernel = np.ones((int(values['-SMT-F']),int(values['-SMT-F'])),np.float32)/(int(values['-SMT-F'])*int(values['-SMT-F']))
            img_s = cv2.filter2D(img_s,-1,kernel)
        if values['-BLUR-F'] != 0: img_s = cv2.blur(img_s,(int(values['-BLUR-F']),int(values['-BLUR-F'])))
        if values['-GBLUR-F'] != 0:
            if values['-GBLUR-F'] % 2 != 0:
                img_s = cv2.GaussianBlur(img_s,(int(values['-GBLUR-F']),int(values['-GBLUR-F'])),0)
        if values['-MBLUR-F'] != 0: 
            if values['-MBLUR-F'] % 2 != 0:
                img_s = cv2.medianBlur(img_s,int(values['-MBLUR-F']))
        if values['-BBLUR-F'] != 0: img_s = cv2.bilateralFilter(img_s,int(values['-BBLUR-F']),75,75)
        if values['-BTRESH-F'] != 0: ret,img_s = cv2.threshold(img_s,int(values['-BTRESH-F']),255,cv2.THRESH_BINARY)
        if values['-BITRESH-F'] != 0: ret,img_s = cv2.threshold(img_s,int(values['-BITRESH-F']),255,cv2.THRESH_BINARY_INV)
        if values['-TTRESH-F'] != 0: ret,img_s = cv2.threshold(img_s,int(values['-TTRESH-F']),255,cv2.THRESH_TRUNC)
        if values['-ZTRESH-F'] != 0: ret,img_s = cv2.threshold(img_s,int(values['-ZTRESH-F']),255,cv2.THRESH_TOZERO)
        if values['-ZITRESH-F'] != 0: ret,img_s = cv2.threshold(img_s,int(values['-ZITRESH-F']),255,cv2.THRESH_TOZERO_INV)



        file_name = sg.popup_get_text('Resim ismi girin')
        cv2.imwrite(f'{file_name}.jpeg',img_s)