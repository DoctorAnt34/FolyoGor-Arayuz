import PySimpleGUI as sg
import cv2
import numpy as np
import copy

sg.theme('Default1')

IMAGE_CROP_SIZE = 450


org_filtered_img_1 = sg.Frame('Orginal Image',[[sg.Image(filename=None, key="org_img")]])
filtered_img_1 = sg.Frame('Filtered Image 1',[[sg.Image(filename=None, key="filtered_img_1")]])
filtered_img_2 = sg.Frame('Filtered Image 2',[[sg.Image(filename=None, key="filtered_img_2")]])
final_img = sg.Frame('Final Image',[[sg.Image(filename=None, key="final_img")]])

org_vid = sg.Frame('Orginal Video',[[sg.Image(filename=None, key="org_vid")]])
filtered_vid_1 = sg.Frame('Filtered Video 1',[[sg.Image(filename=None, key="filtered_vid_1")]])
filtered_vid_2 = sg.Frame('Filtered Video 2',[[sg.Image(filename=None, key="filtered_vid_2")]])
final_vid = sg.Frame('Final Video',[[sg.Image(filename=None, key="final_vid")]])


img_frame = sg.pin(sg.Frame('Picture',[[org_filtered_img_1,filtered_img_1,filtered_img_2,final_img]],key='pic_frame',visible=True))

vid_frame = sg.pin(sg.Frame('Video',[[org_vid,filtered_vid_1,filtered_vid_2,final_vid]],key='vid_frame',visible=False))



def crop_square(img, size, interpolation=cv2.INTER_AREA):
    '''resmi bozmadan parça kesme(küçültme)'''
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized






'''Image 1 filters'''

img1_iter_col = sg.Frame('Iter Test',[
    [sg.Text('Erosion Iteration'),sg.Slider(range=(1,10),orientation='h',key='img1_erosion_iteration',enable_events=True)],
    [sg.Text('Dilation Iteration'),sg.Slider(range=(1,10),orientation='h',key='img1_dilation_iteration',enable_events=True)],
])
img1_smoot_col = sg.Frame('Smoothing',[
    [sg.Text('2D Convolution'),sg.Slider(range=(0,255),orientation='h',key='img1_2d',enable_events=True)],
    [sg.Text('Image Blurring'),sg.Slider(range=(0,255),orientation='h',key='img1_blur',enable_events=True)],
    [sg.Text('Gaussian Blurring'),sg.Slider(range=(0,255),orientation='h',key='img1_gaussian',enable_events=True)],
    [sg.Text('Median Blurring'),sg.Slider(range=(0,255),orientation='h',key='img1_median',enable_events=True)],
    [sg.Text('Bilateral Filtering'),sg.Slider(range=(0,255),orientation='h',key='img1_bilateral',enable_events=True)],
])
img1_adap_thres_col=sg.Frame('Adaptive Thresholding',[
    [sg.Text('Mean'),sg.Radio('ON','MEAN',key='img1_mean_on_off',enable_events=True),sg.Radio('OFF','MEAN'),sg.Text('Block Size'),sg.Slider(range=(0,20),orientation='h',key='img1_mean_block_size',enable_events=True),sg.Text('Constant'),sg.Slider(range=(0,20),orientation='h',key='img1_mean_constant',enable_events=True)],
    [],
    [],
    [sg.Text('Canny Edge Detection'),sg.Slider(range=(0,255),orientation='h',key='img1_canny1',enable_events=True),sg.Slider(range=(0,255),orientation='h',key='img1_canny2',enable_events=True)],
    [sg.Text('Sobel Edge Detection'),sg.Slider(range=(0,255),orientation='h',key='img1_sobel',enable_events=True)],
    [sg.Text('dx :'),sg.Radio('ON','img1_dx',key='img1_sobel_dx',enable_events=True),sg.Radio('OFF','img1_dx')],
    [sg.Text('dy :'),sg.Radio('ON','img1_dy',key='img1_sobel_dy',enable_events=True),sg.Radio('OFF','img_1dy')]
])
img1_thres_col = sg.Frame('Tresholding',[
    [sg.Text('Binary'),sg.Slider(range=(0,255),orientation='h',key='img1_binary',enable_events=True,)],
    [sg.Text('Binary INV'),sg.Slider(range=(0,255),orientation='h',key='img1_binary_inv',enable_events=True)],
    [sg.Text('Trunc'),sg.Slider(range=(0,255),orientation='h',key='img1_trunc',enable_events=True)],
    [sg.Text('TOZERO'),sg.Slider(range=(0,255),orientation='h',key='img1_tozero',enable_events=True)],
    [sg.Text('TOZERO INV'),sg.Slider(range=(0,255),orientation='h',key='img1_tozero_inv',enable_events=True)],
])

img1_morph_col = sg.Frame('Morphological Transformation',[
    [sg.Text('Erosion'),sg.Slider(range=(0,255),orientation='h',key='img1_erosion',enable_events=True)],
    [sg.Text('Dilation'),sg.Slider(range=(0,255),orientation='h',key='img1_dilation',enable_events=True)],
    [sg.Text('Opening'),sg.Slider(range=(0,255),orientation='h',key='img1_opening',enable_events=True)],
    [sg.Text('Closing'),sg.Slider(range=(0,255),orientation='h',key='img1_closing',enable_events=True)],
    [sg.Text('Morphological Gradient'),sg.Slider(range=(0,255),orientation='h',key='img1_gradiant',enable_events=True)],
    [sg.Text('Top Hat'),sg.Slider(range=(0,255),orientation='h',key='img1_tophat',enable_events=True)],
    [sg.Text('Black Hat'),sg.Slider(range=(0,255),orientation='h',key='img1_blackhat',enable_events=True)],
])


img1_layout =[
    [img1_smoot_col,img1_thres_col,img1_morph_col,img1_iter_col,img1_adap_thres_col]
]

'''Image 2 filters'''

img2_iter_col = sg.Frame('Iter Test',[
    [sg.Text('Erosion Iteration'),sg.Slider(range=(1,10),orientation='h',key='img2_erosion_iteration',enable_events=True)],
    [sg.Text('Dilation Iteration'),sg.Slider(range=(1,10),orientation='h',key='img2_dilation_iteration',enable_events=True)],
])
img2_smoot_col = sg.Frame('Smoothing',[
    [sg.Text('2D Convolution'),sg.Slider(range=(0,255),orientation='h',key='img2_2d',enable_events=True)],
    [sg.Text('Image Blurring'),sg.Slider(range=(0,255),orientation='h',key='img2_blur',enable_events=True)],
    [sg.Text('Gaussian Blurring'),sg.Slider(range=(0,255),orientation='h',key='img2_gaussian',enable_events=True)],
    [sg.Text('Median Blurring'),sg.Slider(range=(0,255),orientation='h',key='img2_median',enable_events=True)],
    [sg.Text('Bilateral Filtering'),sg.Slider(range=(0,255),orientation='h',key='img2_bilateral',enable_events=True)],
])
img2_adap_thres_col=sg.Frame('Adaptive Thresholding',[
    [sg.Text('Mean'),sg.Radio('ON','MEAN',key='img2_mean_on_off',enable_events=True),sg.Radio('OFF','MEAN'),sg.Text('Block Size'),sg.Slider(range=(0,20),orientation='h',key='img2_mean_block_size',enable_events=True),sg.Text('Constant'),sg.Slider(range=(0,20),orientation='h',key='img2_mean_constant',enable_events=True)],
    [],
    [],
    [sg.Text('Canny Edge Detection'),sg.Slider(range=(0,255),orientation='h',key='img2_canny1',enable_events=True),sg.Slider(range=(0,255),orientation='h',key='img2_canny2',enable_events=True)],
    [sg.Text('Sobel Edge Detection'),sg.Slider(range=(0,255),orientation='h',key='img2_sobel',enable_events=True)],
    [sg.Text('dx :'),sg.Radio('ON','img2_dx',key='img2_sobel_dx',enable_events=True),sg.Radio('OFF','img2_dx')],
    [sg.Text('dy :'),sg.Radio('ON','img2_dy',key='img2_sobel_dy',enable_events=True),sg.Radio('OFF','img2_dy')]
])
img2_thres_col = sg.Frame('Tresholding',[
    [sg.Text('Binary'),sg.Slider(range=(0,255),orientation='h',key='img2_binary',enable_events=True,)],
    [sg.Text('Binary INV'),sg.Slider(range=(0,255),orientation='h',key='img2_binary_inv',enable_events=True)],
    [sg.Text('Trunc'),sg.Slider(range=(0,255),orientation='h',key='img2_trunc',enable_events=True)],
    [sg.Text('TOZERO'),sg.Slider(range=(0,255),orientation='h',key='img2_tozero',enable_events=True)],
    [sg.Text('TOZERO INV'),sg.Slider(range=(0,255),orientation='h',key='img2_tozero_inv',enable_events=True)],
])

img2_morph_col = sg.Frame('Morphological Transformation',[
    [sg.Text('Erosion'),sg.Slider(range=(0,255),orientation='h',key='img2_erosion',enable_events=True)],
    [sg.Text('Dilation'),sg.Slider(range=(0,255),orientation='h',key='img2_dilation',enable_events=True)],
    [sg.Text('Opening'),sg.Slider(range=(0,255),orientation='h',key='img2_opening',enable_events=True)],
    [sg.Text('Closing'),sg.Slider(range=(0,255),orientation='h',key='img2_closing',enable_events=True)],
    [sg.Text('Morphological Gradient'),sg.Slider(range=(0,255),orientation='h',key='img2_gradiant',enable_events=True)],
    [sg.Text('Top Hat'),sg.Slider(range=(0,255),orientation='h',key='img2_tophat',enable_events=True)],
    [sg.Text('Black Hat'),sg.Slider(range=(0,255),orientation='h',key='img2_blackhat',enable_events=True)],
])


img2_layout =[
    [img2_smoot_col,img2_thres_col,img2_morph_col,img2_iter_col,img2_adap_thres_col]
]


'''Image Final filters'''

img_final_iter_col = sg.Frame('Iter Test',[
    [sg.Text('Erosion Iteration'),sg.Slider(range=(1,10),orientation='h',key='img_final_erosion_iteration',enable_events=True)],
    [sg.Text('Dilation Iteration'),sg.Slider(range=(1,10),orientation='h',key='img_final_dilation_iteration',enable_events=True)],
])
img_final_smoot_col = sg.Frame('Smoothing',[
    [sg.Text('2D Convolution'),sg.Slider(range=(0,255),orientation='h',key='img_final_2d',enable_events=True)],
    [sg.Text('Image Blurring'),sg.Slider(range=(0,255),orientation='h',key='img_final_blur',enable_events=True)],
    [sg.Text('Gaussian Blurring'),sg.Slider(range=(0,255),orientation='h',key='img_final_gaussian',enable_events=True)],
    [sg.Text('Median Blurring'),sg.Slider(range=(0,255),orientation='h',key='img_final_median',enable_events=True)],
    [sg.Text('Bilateral Filtering'),sg.Slider(range=(0,255),orientation='h',key='img_final_bilateral',enable_events=True)],
])
img_final_adap_thres_col=sg.Frame('Adaptive Thresholding',[
    [sg.Text('Mean'),sg.Radio('ON','MEAN',key='img_final_mean_on_off',enable_events=True),sg.Radio('OFF','MEAN'),sg.Text('Block Size'),sg.Slider(range=(0,20),orientation='h',key='img_final_mean_block_size',enable_events=True),sg.Text('Constant'),sg.Slider(range=(0,20),orientation='h',key='img_final_mean_constant',enable_events=True)],
    [],
    [],
    [sg.Text('Canny Edge Detection'),sg.Slider(range=(0,255),orientation='h',key='img_final_canny1',enable_events=True),sg.Slider(range=(0,255),orientation='h',key='img_final_canny2',enable_events=True)],
    [sg.Text('Sobel Edge Detection'),sg.Slider(range=(0,255),orientation='h',key='img_final_sobel',enable_events=True)],
    [sg.Text('dx :'),sg.Radio('ON','dx',key='img_final_sobel_dx',enable_events=True),sg.Radio('OFF','dx')],
    [sg.Text('dy :'),sg.Radio('ON','dy',key='img_final_sobel_dy',enable_events=True),sg.Radio('OFF','dy')]
])
img_final_thres_col = sg.Frame('Tresholding',[
    [sg.Text('Binary'),sg.Slider(range=(0,255),orientation='h',key='img_final_binary',enable_events=True,)],
    [sg.Text('Binary INV'),sg.Slider(range=(0,255),orientation='h',key='img_final_binary_inv',enable_events=True)],
    [sg.Text('Trunc'),sg.Slider(range=(0,255),orientation='h',key='img_final_trunc',enable_events=True)],
    [sg.Text('TOZERO'),sg.Slider(range=(0,255),orientation='h',key='img_final_tozero',enable_events=True)],
    [sg.Text('TOZERO INV'),sg.Slider(range=(0,255),orientation='h',key='img_final_tozero_inv',enable_events=True)],
])

img_final_morph_col = sg.Frame('Morphological Transformation',[
    [sg.Text('Erosion'),sg.Slider(range=(0,255),orientation='h',key='img_final_erosion',enable_events=True)],
    [sg.Text('Dilation'),sg.Slider(range=(0,255),orientation='h',key='img_final_dilation',enable_events=True)],
    [sg.Text('Opening'),sg.Slider(range=(0,255),orientation='h',key='img_final_opening',enable_events=True)],
    [sg.Text('Closing'),sg.Slider(range=(0,255),orientation='h',key='img_final_closing',enable_events=True)],
    [sg.Text('Morphological Gradient'),sg.Slider(range=(0,255),orientation='h',key='img_final_gradiant',enable_events=True)],
    [sg.Text('Top Hat'),sg.Slider(range=(0,255),orientation='h',key='img_final_tophat',enable_events=True)],
    [sg.Text('Black Hat'),sg.Slider(range=(0,255),orientation='h',key='img_final_blackhat',enable_events=True)],
])


img_final_layout =[
    [img_final_smoot_col,img_final_thres_col,img_final_morph_col,img_final_iter_col,img_final_adap_thres_col]
]





'''Tab Group'''

tabs = sg.TabGroup([[sg.Tab('Image 1',img1_layout), sg.Tab('Image 2', img2_layout), sg.Tab('Final Image', img_final_layout)]])

'''Navigate Buttons'''


selection_row_1 = [
    sg.Push(),
    sg.Radio('Original - Filtered Image 1','sub_change',default=True,enable_events=True,key='o-f1'),
    sg.Radio('Original - Filtered Image 2','sub_change',enable_events=True,key='o-f2'),
    sg.Radio('Filtered Image 1 - Filtered Image 2','sub_change',enable_events=True,key='f1-f2'),
]

selection_row_2 = [
    sg.Push(),
    sg.Radio('Filtered Image 1 - Original','sub_change',enable_events=True,key='f1-o'),
    sg.Radio('Filtered Image 2 - Original','sub_change',enable_events=True,key='f2-o'),
    sg.Radio('Filtered Image 2 - Filtered Image 1','sub_change',enable_events=True,key='f2-f1'),
]

selection_frame = sg.Frame('Çıktı Seçimi',[selection_row_1,selection_row_2])

layout=[
    [img_frame,vid_frame],
    [sg.Radio('Picture','feed_change',default=True,enable_events=True,key='feed_pic'),sg.Radio('Video','feed_change',enable_events=True,key='feed_vid'),sg.Push(),selection_frame],
    [sg.pin(sg.Button('Take Picture',key='take_pic',visible=False)),sg.pin(sg.Button('Browse',key='browse',visible=True)),sg.Button('Reset',key='reset'),sg.pin(sg.Button('Save',key='save',visible=True))],
    [tabs]
]


window = sg.Window('FolyoGör',layout,grab_anywhere=False,finalize=True)
window.maximize()
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
feed = 'pic'
final_state = 'o-f1'
img_set = False

def change_final():
    if final_state == 'o-f1':
        if feed == 'pic':return  cv2.subtract(img_o,filtered_img_1)
        elif feed == 'vid':return cv2.subtract(vid_o,filtered_vid_1)

    if final_state == 'o-f2':
        if feed == 'pic':return  cv2.subtract(img_o,filtered_img_2)
        elif feed == 'vid':return cv2.subtract(vid_o,filtered_vid_2)
    
    if final_state == 'f1-f2':
        if feed == 'pic':return  cv2.subtract(filtered_img_1,filtered_img_2)
        elif feed == 'vid':return cv2.subtract(filtered_vid_1,filtered_vid_2)

    if final_state == 'f1-o':
        if feed == 'pic':return  cv2.subtract(filtered_img_1,img_o)
        elif feed == 'vid':return cv2.subtract(filtered_vid_1,vid_o)

    if final_state == 'f2-o':
        if feed == 'pic':return  cv2.subtract(filtered_img_2,img_o)
        elif feed == 'vid':return cv2.subtract(filtered_vid_2,vid_o)

    if final_state == 'f2-f1':
        if feed == 'pic':return  cv2.subtract(filtered_img_2,filtered_img_1)
        elif feed == 'vid':return cv2.subtract(filtered_vid_2,filtered_vid_1)



while True:
    event, values = window.read(timeout=20)

    if event == sg.WIN_CLOSED or event == 'Exit':
        break

    if event == 'feed_pic':
        feed = 'pic'
        window['browse'].update(visible = True)
        window['take_pic'].update(visible = False)
        window['save'].update(visible = True)
        window['pic_frame'].update(visible = True)
        window['vid_frame'].update(visible = False)

    if event == 'feed_vid':
        feed = 'vid'
        window['browse'].update(visible = False)
        window['take_pic'].update(visible = True)
        window['save'].update(visible = False)
        window['pic_frame'].update(visible = False)
        window['vid_frame'].update(visible = True)

    if event == 'browse':
        file_path = sg.popup_get_file('Açılacak Resmi Seçin',file_types=[('Images','*.png'),('Images','*.jpg')])
        img_o = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
        img_o = crop_square(img_o,IMAGE_CROP_SIZE)
        imgbytes = cv2.imencode('.png', img_o)[1].tobytes()  
        window['filtered_img_1'].update(data=imgbytes)
        window['filtered_img_2'].update(data=imgbytes)
        window['org_img'].update(data=imgbytes)
        filtered_img_1 = copy.deepcopy(img_o)
        filtered_img_2 = copy.deepcopy(img_o)
        img_set = True
        #img_s = cv2.subtract(img_o,img)
        #imgbytes_s = cv2.imencode('.png', img_s)[1].tobytes()
        #window['final_img'].update(data=imgbytes_s)

    if event == 'take_pic':
        ret,frame = cap.read()
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
        # Orjinal görüntü
        nparr = np.fromstring(vidbytes, np.uint8)
        img_o = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        img_o = crop_square(img_o,IMAGE_CROP_SIZE)
        img_o = crop_square(img_o,IMAGE_CROP_SIZE)
        imgbytes = cv2.imencode('.png', img_o)[1].tobytes()  
        window['filtered_img_1'].update(data=imgbytes)
        window['filtered_img_2'].update(data=imgbytes)
        window['org_img'].update(data=imgbytes)
        filtered_img_1 = copy.deepcopy(img_o)
        filtered_img_2 = copy.deepcopy(img_o)
        img_set = True
        feed = 'pic'
        window['browse'].update(visible = True)
        window['take_pic'].update(visible = False)
        window['save'].update(visible = True)
        window['pic_frame'].update(visible = True)
        window['vid_frame'].update(visible = False)
        window['feed_pic'].update(True)

    if event == 'save':
        file_name = sg.popup_get_text('Resim ismi girin')
        if file_name:
            cv2.imwrite(f'export/{file_name}_orginal.png',img_o)
            cv2.imwrite(f'export/{file_name}_filtered_1.png',filtered_img_1)
            cv2.imwrite(f'export/{file_name}_filtered_2.png',filtered_img_2)
            cv2.imwrite(f'export/{file_name}_final.png',final_image)
            info = f'''
    Output State = {final_state}        
    [Image Filter 1]
    [Smoothing]
    2D Convolution = {values["img1_2d"]}
    Image Blurring = {values["img1_blur"]}
    Gaussian Blurring = {values["img1_gaussian"]}
    Median Blurring = {values["img1_median"]}
    Bilateral Filtering = {values["img1_bilateral"]}
    [Tresholding]
    Binary = {values["img1_binary"]}
    Binary INV = {values["img1_binary_inv"]}
    Trunc = {values["img1_trunc"]}
    TOZERO = {values["img1_tozero"]}
    TOZERO INV = {values["img1_tozero_inv"]}
    [Morphological Transformation]
    Erosion = {values["img1_erosion"]}
    Erosion Iteration = {values["img1_erosion_iteration"]}
    Dilation = {values["img1_dilation"]}
    Dlation Iteration = {values["img1_dilation_iteration"]}
    Opening = {values["img1_opening"]}
    Closing = {values["img1_closing"]}
    Morphological Gradient = {values["img1_gradiant"]}
    Top Hat = {values["img1_tophat"]}
    Black Hat = {values["img1_blackhat"]}

    [Image Filter 2]
    [Smoothing]
    2D Convolution = {values["img2_2d"]}
    Image Blurring = {values["img2_blur"]}
    Gaussian Blurring = {values["img2_gaussian"]}
    Median Blurring = {values["img2_median"]}
    Bilateral Filtering = {values["img2_bilateral"]}
    [Tresholding]
    Binary = {values["img2_binary"]}
    Binary INV = {values["img2_binary_inv"]}
    Trunc = {values["img2_trunc"]}
    TOZERO = {values["img2_tozero"]}
    TOZERO INV = {values["img2_tozero_inv"]}
    [Morphological Transformation]
    Erosion = {values["img2_erosion"]}
    Erosion Iteration = {values["img2_erosion_iteration"]}
    Dilation = {values["img2_dilation"]}
    Dlation Iteration = {values["img2_dilation_iteration"]}
    Opening = {values["img2_opening"]}
    Closing = {values["img2_closing"]}
    Morphological Gradient = {values["img2_gradiant"]}
    Top Hat = {values["img2_tophat"]}
    Black Hat = {values["img2_blackhat"]}

    [Image Filter Final]
    [Smoothing]
    2D Convolution = {values["img_final_2d"]}
    Image Blurring = {values["img_final_blur"]}
    Gaussian Blurring = {values["img_final_gaussian"]}
    Median Blurring = {values["img_final_median"]}
    Bilateral Filtering = {values["img_final_bilateral"]}
    [Tresholding]
    Binary = {values["img_final_binary"]}
    Binary INV = {values["img_final_binary_inv"]}
    Trunc = {values["img_final_trunc"]}
    TOZERO = {values["img_final_tozero"]}
    TOZERO INV = {values["img_final_tozero_inv"]}
    [Morphological Transformation]
    Erosion = {values["img_final_erosion"]}
    Erosion Iteration = {values["img_final_erosion_iteration"]}
    Dilation = {values["img_final_dilation"]}
    Dlation Iteration = {values["img_final_dilation_iteration"]}
    Opening = {values["img_final_opening"]}
    Closing = {values["img_final_closing"]}
    Morphological Gradient = {values["img_final_gradiant"]}
    Top Hat = {values["img_final_tophat"]}
    Black Hat = {values["img_final_blackhat"]}
    '''
            txt = open(f'export/{file_name}.txt','w+')
            txt.write(info)
            txt.close()

    '''Picture Filtering'''
    if feed == 'pic':
        if img_set:
            o_imgbytes = cv2.imencode('.png', img_o)[1].tobytes()
            window['org_img'].update(data=o_imgbytes)
            filtered_img_1 = copy.deepcopy(img_o)
            filtered_img_2 = copy.deepcopy(img_o)
            '''Filtered Image 1'''
            if values['img1_2d'] != 0:
                kernel = np.ones((int(values['img1_2d']),int(values['img1_2d'])),np.float32)/(int(values['img1_2d'])*int(values['img1_2d']))
                filtered_img_1 = cv2.filter2D(filtered_img_1,-1,kernel)
            if values['img1_blur'] != 0: filtered_img_1 = cv2.blur(filtered_img_1,(int(values['img1_blur']),int(values['img1_blur'])))
            if values['img1_gaussian'] != 0:
                if values['img1_gaussian'] % 2 != 0:
                    filtered_img_1 = cv2.GaussianBlur(filtered_img_1,(int(values['img1_gaussian']),int(values['img1_gaussian'])),0)
            if values['img1_median'] != 0: 
                if values['img1_median'] % 2 != 0:
                    filtered_img_1 = cv2.medianBlur(filtered_img_1,int(values['img1_median']))
            if values['img1_bilateral'] != 0: filtered_img_1 = cv2.bilateralFilter(filtered_img_1,int(values['img1_bilateral']),75,75)
            if values['img1_binary'] != 0: ret,filtered_img_1 = cv2.threshold(filtered_img_1,int(values['img1_binary']),255,cv2.THRESH_BINARY)
            if values['img1_binary_inv'] != 0: ret,filtered_img_1 = cv2.threshold(filtered_img_1,int(values['img1_binary_inv']),255,cv2.THRESH_BINARY_INV)
            if values['img1_trunc'] != 0: ret,filtered_img_1 = cv2.threshold(filtered_img_1,int(values['img1_trunc']),255,cv2.THRESH_TRUNC)
            if values['img1_tozero'] != 0: ret,filtered_img_1 = cv2.threshold(filtered_img_1,int(values['img1_tozero']),255,cv2.THRESH_TOZERO)
            if values['img1_tozero_inv'] != 0: ret,filtered_img_1 = cv2.threshold(filtered_img_1,int(values['img1_tozero_inv']),255,cv2.THRESH_TOZERO_INV)

            if values['img1_erosion'] != 0:
                kernel = np.ones((int(values['img1_erosion']),int(values['img1_erosion'])),np.uint8)
                filtered_img_1 = cv2.erode(filtered_img_1,kernel,iterations=int(values['img1_erosion_iteration']))
            if values['img1_dilation'] != 0:
                kernel = np.ones((int(values['img1_dilation']),int(values['img1_dilation'])),np.uint8)
                filtered_img_1 = cv2.dilate(filtered_img_1,kernel,iterations=int(values['img1_dilation_iteration']))
            if values['img1_opening'] != 0:
                kernel = np.ones((int(values['img1_opening']),int(values['img1_opening'])),np.uint8)
                filtered_img_1 = cv2.morphologyEx(filtered_img_1,cv2.MORPH_OPEN,kernel)
            if values['img1_closing'] != 0:
                kernel = np.ones((int(values['img1_closing']),int(values['img1_closing'])),np.uint8)
                filtered_img_1 = cv2.morphologyEx(filtered_img_1,cv2.MORPH_CLOSE,kernel)
            if values['img1_gradiant'] != 0:
                kernel = np.ones((int(values['img1_gradiant']),int(values['img1_gradiant'])),np.uint8)
                filtered_img_1 = cv2.morphologyEx(filtered_img_1,cv2.MORPH_GRADIENT,kernel)
            if values['img1_tophat'] != 0:
                kernel = np.ones((int(values['img1_tophat']),int(values['img1_tophat'])),np.uint8)
                filtered_img_1 = cv2.morphologyEx(filtered_img_1,cv2.MORPH_TOPHAT,kernel)
            if values['img1_blackhat'] != 0:
                kernel = np.ones((int(values['img1_blackhat']),int(values['img1_blackhat'])),np.uint8)
                filtered_img_1 = cv2.morphologyEx(filtered_img_1,cv2.MORPH_BLACKHAT,kernel)
            if values['img1_mean_on_off'] is True:
                filtered_img_1 = cv2.adaptiveThreshold(filtered_img_1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,int(values['img1_mean_block_size']),int(values['img1_mean_constant']))
            if values['img1_canny1'] != 0 and values['img1_canny2'] != 0:
                filtered_img_1 = cv2.Canny(filtered_img_1,values['img1_canny1'],values['img1_canny2'])
            if values['img1_sobel'] != 0 and (values['img1_sobel_dx'] or values['img1_sobel_dy']) and values['img1_sobel'] % 2 != 0 and values['img1_sobel']<31 :
                filtered_img_1 = cv2.Sobel(filtered_img_1,ddepth=cv2.CV_64F ,dx = int(values['img1_sobel_dx']), dy = int(values['img1_sobel_dy']), ksize = int(values['img1_sobel']),scale=1,delta=0)
                filtered_img_1 = cv2.convertScaleAbs(filtered_img_1)

            filtered_img1_imgbytes = cv2.imencode('.png', filtered_img_1)[1].tobytes() 
            window['filtered_img_1'].update(data = filtered_img1_imgbytes)

            '''Filtered Image 2'''
            if values['img2_2d'] != 0:
                kernel = np.ones((int(values['img2_2d']),int(values['img2_2d'])),np.float32)/(int(values['img2_2d'])*int(values['img2_2d']))
                filtered_img_2 = cv2.filter2D(filtered_img_2,-1,kernel)
            if values['img2_blur'] != 0: filtered_img_2 = cv2.blur(filtered_img_2,(int(values['img2_blur']),int(values['img2_blur'])))
            if values['img2_gaussian'] != 0:
                if values['img2_gaussian'] % 2 != 0:
                    filtered_img_2 = cv2.GaussianBlur(filtered_img_2,(int(values['img2_gaussian']),int(values['img2_gaussian'])),0)
            if values['img2_median'] != 0: 
                if values['img2_median'] % 2 != 0:
                    filtered_img_2 = cv2.medianBlur(filtered_img_2,int(values['img2_median']))
            if values['img2_bilateral'] != 0: filtered_img_2 = cv2.bilateralFilter(filtered_img_2,int(values['img2_bilateral']),75,75)
            if values['img2_binary'] != 0: ret,filtered_img_2 = cv2.threshold(filtered_img_2,int(values['img2_binary']),255,cv2.THRESH_BINARY)
            if values['img2_binary_inv'] != 0: ret,filtered_img_2 = cv2.threshold(filtered_img_2,int(values['img2_binary_inv']),255,cv2.THRESH_BINARY_INV)
            if values['img2_trunc'] != 0: ret,filtered_img_2 = cv2.threshold(filtered_img_2,int(values['img2_trunc']),255,cv2.THRESH_TRUNC)
            if values['img2_tozero'] != 0: ret,filtered_img_2 = cv2.threshold(filtered_img_2,int(values['img2_tozero']),255,cv2.THRESH_TOZERO)
            if values['img2_tozero_inv'] != 0: ret,filtered_img_2 = cv2.threshold(filtered_img_2,int(values['img2_tozero_inv']),255,cv2.THRESH_TOZERO_INV)

            if values['img2_erosion'] != 0:
                kernel = np.ones((int(values['img2_erosion']),int(values['img2_erosion'])),np.uint8)
                filtered_img_2 = cv2.erode(filtered_img_2,kernel,iterations=int(values['img2_erosion_iteration']))
            if values['img2_dilation'] != 0:
                kernel = np.ones((int(values['img2_dilation']),int(values['img2_dilation'])),np.uint8)
                filtered_img_2 = cv2.dilate(filtered_img_2,kernel,iterations=int(values['img2_dilation_iteration']))
            if values['img2_opening'] != 0:
                kernel = np.ones((int(values['img2_opening']),int(values['img2_opening'])),np.uint8)
                filtered_img_2 = cv2.morphologyEx(filtered_img_2,cv2.MORPH_OPEN,kernel)
            if values['img2_closing'] != 0:
                kernel = np.ones((int(values['img2_closing']),int(values['img2_closing'])),np.uint8)
                filtered_img_2 = cv2.morphologyEx(filtered_img_2,cv2.MORPH_CLOSE,kernel)
            if values['img2_gradiant'] != 0:
                kernel = np.ones((int(values['img2_gradiant']),int(values['img2_gradiant'])),np.uint8)
                filtered_img_2 = cv2.morphologyEx(filtered_img_2,cv2.MORPH_GRADIENT,kernel)
            if values['img2_tophat'] != 0:
                kernel = np.ones((int(values['img2_tophat']),int(values['img2_tophat'])),np.uint8)
                filtered_img_2 = cv2.morphologyEx(filtered_img_2,cv2.MORPH_TOPHAT,kernel)
            if values['img2_blackhat'] != 0:
                kernel = np.ones((int(values['img2_blackhat']),int(values['img2_blackhat'])),np.uint8)
                filtered_img_2 = cv2.morphologyEx(filtered_img_2,cv2.MORPH_BLACKHAT,kernel)
            if values['img2_mean_on_off'] is True:
                filtered_img_2 = cv2.adaptiveThreshold(filtered_img_2,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,int(values['img2_mean_block_size']),int(values['img2_mean_constant']))
            if values['img2_canny1'] != 0 and values['img2_canny2'] != 0:
                filtered_img_2 = cv2.Canny(filtered_img_2,values['img2_canny1'],values['img2_canny2'])
            if values['img2_sobel'] != 0 and (values['img2_sobel_dx'] or values['img2_sobel_dy']) and values['img2_sobel'] % 2 != 0 and values['img2_sobel']<31 :
                filtered_img_2 = cv2.Sobel(filtered_img_2,ddepth=cv2.CV_8U ,dx = int(values['img2_sobel_dx']), dy = int(values['img2_sobel_dy']), ksize = int(values['img2_sobel']),scale=1,delta=0)

            filtered_img2_imgbytes = cv2.imencode('.png', filtered_img_2)[1].tobytes()  
            window['filtered_img_2'].update(data = filtered_img2_imgbytes)


            
            '''Final Image'''
            if event == 'o-f1':
                final_state = 'o-f1'

            if event == 'o-f2':
                final_state = 'o-f2'
            
            if event == 'f1-f2':
                final_state = 'f1-f2'

            if event == 'f1-o':
                final_state = 'f1-o'

            if event == 'f2-o':
                final_state = 'f2-o'

            if event == 'f2-f1':
                final_state = 'f2-f1'

            final_image = change_final()
            if values['img_final_2d'] != 0:
                kernel = np.ones((int(values['img_final_2d']),int(values['img_final_2d'])),np.float32)/(int(values['img_final_2d'])*int(values['img_final_2d']))
                final_image = cv2.filter2D(final_image,-1,kernel)
            if values['img_final_blur'] != 0: final_image = cv2.blur(final_image,(int(values['img_final_blur']),int(values['img_final_blur'])))
            if values['img_final_gaussian'] != 0:
                if values['img_final_gaussian'] % 2 != 0:
                    final_image = cv2.GaussianBlur(final_image,(int(values['img_final_gaussian']),int(values['img_final_gaussian'])),0)
            if values['img_final_median'] != 0: 
                if values['img_final_median'] % 2 != 0:
                    final_image = cv2.medianBlur(final_image,int(values['img_final_median']))
            if values['img_final_bilateral'] != 0: final_image = cv2.bilateralFilter(final_image,int(values['img_final_bilateral']),75,75)
            if values['img_final_binary'] != 0: ret,final_image = cv2.threshold(final_image,int(values['img_final_binary']),255,cv2.THRESH_BINARY)
            if values['img_final_binary_inv'] != 0: ret,final_image = cv2.threshold(final_image,int(values['img_final_binary_inv']),255,cv2.THRESH_BINARY_INV)
            if values['img_final_trunc'] != 0: ret,final_image = cv2.threshold(final_image,int(values['img_final_trunc']),255,cv2.THRESH_TRUNC)
            if values['img_final_tozero'] != 0: ret,final_image = cv2.threshold(final_image,int(values['img_final_tozero']),255,cv2.THRESH_TOZERO)
            if values['img_final_tozero_inv'] != 0: ret,final_image = cv2.threshold(final_image,int(values['img_final_tozero_inv']),255,cv2.THRESH_TOZERO_INV)

            if values['img_final_erosion'] != 0:
                kernel = np.ones((int(values['img_final_erosion']),int(values['img_final_erosion'])),np.uint8)
                final_image = cv2.erode(final_image,kernel,iterations=int(values['img_final_erosion_iteration']))
            if values['img_final_dilation'] != 0:
                kernel = np.ones((int(values['img_final_dilation']),int(values['img_final_dilation'])),np.uint8)
                final_image = cv2.dilate(final_image,kernel,iterations=int(values['img_final_dilation_iteration']))
            if values['img_final_opening'] != 0:
                kernel = np.ones((int(values['img_final_opening']),int(values['img_final_opening'])),np.uint8)
                final_image = cv2.morphologyEx(final_image,cv2.MORPH_OPEN,kernel)
            if values['img_final_closing'] != 0:
                kernel = np.ones((int(values['img_final_closing']),int(values['img_final_closing'])),np.uint8)
                final_image = cv2.morphologyEx(final_image,cv2.MORPH_CLOSE,kernel)
            if values['img_final_gradiant'] != 0:
                kernel = np.ones((int(values['img_final_gradiant']),int(values['img_final_gradiant'])),np.uint8)
                final_image = cv2.morphologyEx(final_image,cv2.MORPH_GRADIENT,kernel)
            if values['img_final_tophat'] != 0:
                kernel = np.ones((int(values['img_final_tophat']),int(values['img_final_tophat'])),np.uint8)
                final_image = cv2.morphologyEx(final_image,cv2.MORPH_TOPHAT,kernel)
            if values['img_final_blackhat'] != 0:
                kernel = np.ones((int(values['img_final_blackhat']),int(values['img_final_blackhat'])),np.uint8)
                final_image = cv2.morphologyEx(final_image,cv2.MORPH_BLACKHAT,kernel)
            if values['img_final_mean_on_off'] is True:
                final_image = cv2.adaptiveThreshold(final_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,int(values['img_final_mean_block_size']),int(values['img_final_mean_constant']))
            if values['img_final_canny1'] != 0 and values['img_final_canny2'] != 0:
                final_image = cv2.Canny(final_image,values['img_final_canny1'],values['img_final_canny2'])
            if values['img_final_sobel'] != 0 and (values['img_final_sobel_dx'] or values['img_final_sobel_dy']) and values['img_final_sobel'] % 2 != 0 and values['img_final_sobel']<31 :
                final_image = cv2.Sobel(final_image,ddepth=cv2.CV_8U ,dx = int(values['img_final_sobel_dx']), dy = int(values['img_final_sobel_dy']), ksize = int(values['img_final_sobel']),scale=1,delta=0)

            final_img_imgbytes = cv2.imencode('.png', final_image)[1].tobytes()
            window['final_img'].update(data = final_img_imgbytes)



    '''Video Filtering'''

    if feed == 'vid':
        ret, frame = cap.read()
        vidbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto

        # Orjinal görüntü
        nparr = np.fromstring(vidbytes, np.uint8)
        vid_o = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        vid_o = crop_square(vid_o,IMAGE_CROP_SIZE)

        #vid_o = cv2.resize(vid_o,[int(3840/6),int(2160/6)])
        o_vidbytes = cv2.imencode('.png', vid_o)[1].tobytes()
        window['org_vid'].update(data=o_vidbytes)
        filtered_vid_1 = copy.deepcopy(vid_o)
        filtered_vid_2 = copy.deepcopy(vid_o)

        '''Filtered Video 1'''
        if values['img1_2d'] != 0:
            kernel = np.ones((int(values['img1_2d']),int(values['img1_2d'])),np.float32)/(int(values['img1_2d'])*int(values['img1_2d']))
            filtered_vid_1 = cv2.filter2D(filtered_vid_1,-1,kernel)
        if values['img1_blur'] != 0: filtered_vid_1 = cv2.blur(filtered_vid_1,(int(values['img1_blur']),int(values['img1_blur'])))
        if values['img1_gaussian'] != 0:
            if values['img1_gaussian'] % 2 != 0:
                filtered_vid_1 = cv2.GaussianBlur(filtered_vid_1,(int(values['img1_gaussian']),int(values['img1_gaussian'])),0)
        if values['img1_median'] != 0: 
            if values['img1_median'] % 2 != 0:
                filtered_vid_1 = cv2.medianBlur(filtered_vid_1,int(values['img1_median']))
        if values['img1_bilateral'] != 0: filtered_vid_1 = cv2.bilateralFilter(filtered_vid_1,int(values['img1_bilateral']),75,75)
        if values['img1_binary'] != 0: ret,filtered_vid_1 = cv2.threshold(filtered_vid_1,int(values['img1_binary']),255,cv2.THRESH_BINARY)
        if values['img1_binary_inv'] != 0: ret,filtered_vid_1 = cv2.threshold(filtered_vid_1,int(values['img1_binary_inv']),255,cv2.THRESH_BINARY_INV)
        if values['img1_trunc'] != 0: ret,filtered_vid_1 = cv2.threshold(filtered_vid_1,int(values['img1_trunc']),255,cv2.THRESH_TRUNC)
        if values['img1_tozero'] != 0: ret,filtered_vid_1 = cv2.threshold(filtered_vid_1,int(values['img1_tozero']),255,cv2.THRESH_TOZERO)
        if values['img1_tozero_inv'] != 0: ret,filtered_vid_1 = cv2.threshold(filtered_vid_1,int(values['img1_tozero_inv']),255,cv2.THRESH_TOZERO_INV)

        if values['img1_erosion'] != 0:
            kernel = np.ones((int(values['img1_erosion']),int(values['img1_erosion'])),np.uint8)
            filtered_vid_1 = cv2.erode(filtered_vid_1,kernel,iterations=int(values['img1_erosion_iteration']))
        if values['img1_dilation'] != 0:
            kernel = np.ones((int(values['img1_dilation']),int(values['img1_dilation'])),np.uint8)
            filtered_vid_1 = cv2.dilate(filtered_vid_1,kernel,iterations=int(values['img1_dilation_iteration']))
        if values['img1_opening'] != 0:
            kernel = np.ones((int(values['img1_opening']),int(values['img1_opening'])),np.uint8)
            filtered_vid_1 = cv2.morphologyEx(filtered_vid_1,cv2.MORPH_OPEN,kernel)
        if values['img1_closing'] != 0:
            kernel = np.ones((int(values['img1_closing']),int(values['img1_closing'])),np.uint8)
            filtered_vid_1 = cv2.morphologyEx(filtered_vid_1,cv2.MORPH_CLOSE,kernel)
        if values['img1_gradiant'] != 0:
            kernel = np.ones((int(values['img1_gradiant']),int(values['img1_gradiant'])),np.uint8)
            filtered_vid_1 = cv2.morphologyEx(filtered_vid_1,cv2.MORPH_GRADIENT,kernel)
        if values['img1_tophat'] != 0:
            kernel = np.ones((int(values['img1_tophat']),int(values['img1_tophat'])),np.uint8)
            filtered_vid_1 = cv2.morphologyEx(filtered_vid_1,cv2.MORPH_TOPHAT,kernel)
        if values['img1_blackhat'] != 0:
            kernel = np.ones((int(values['img1_blackhat']),int(values['img1_blackhat'])),np.uint8)
            filtered_vid_1 = cv2.morphologyEx(filtered_vid_1,cv2.MORPH_BLACKHAT,kernel)
        if values['img1_mean_on_off'] is True:
            filtered_vid_1 = cv2.adaptiveThreshold(filtered_vid_1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,int(values['img1_mean_block_size']),int(values['img1_mean_constant']))
        if values['img1_canny1'] != 0 and values['img1_canny2'] != 0:
            filtered_vid_1 = cv2.Canny(filtered_vid_1,values['img1_canny1'],values['img1_canny2'])
        if values['img1_sobel'] != 0 and (values['img1_sobel_dx'] or values['img1_sobel_dy']) and values['img1_sobel'] % 2 != 0 and values['img1_sobel']<31 :
            filtered_vid_1 = cv2.Sobel(filtered_vid_1,ddepth=cv2.CV_8U ,dx = int(values['img1_sobel_dx']), dy = int(values['img1_sobel_dy']), ksize = int(values['img1_sobel']),scale=1,delta=0)
        filtered_vid_1_vidbytes = cv2.imencode('.png', filtered_vid_1)[1].tobytes() 
        window['filtered_vid_1'].update(data = filtered_vid_1_vidbytes)

        '''Filtered Video 2'''
        if values['img2_2d'] != 0:
            kernel = np.ones((int(values['img2_2d']),int(values['img2_2d'])),np.float32)/(int(values['img2_2d'])*int(values['img2_2d']))
            filtered_vid_2 = cv2.filter2D(filtered_vid_2,-1,kernel)
        if values['img2_blur'] != 0: filtered_vid_2 = cv2.blur(filtered_vid_2,(int(values['img2_blur']),int(values['img2_blur'])))
        if values['img2_gaussian'] != 0:
            if values['img2_gaussian'] % 2 != 0:
                filtered_vid_2 = cv2.GaussianBlur(filtered_vid_2,(int(values['img2_gaussian']),int(values['img2_gaussian'])),0)
        if values['img2_median'] != 0: 
            if values['img2_median'] % 2 != 0:
                filtered_vid_2 = cv2.medianBlur(filtered_vid_2,int(values['img2_median']))
        if values['img2_bilateral'] != 0: filtered_vid_2 = cv2.bilateralFilter(filtered_vid_2,int(values['img2_bilateral']),75,75)
        if values['img2_binary'] != 0: ret,filtered_vid_2 = cv2.threshold(filtered_vid_2,int(values['img2_binary']),255,cv2.THRESH_BINARY)
        if values['img2_binary_inv'] != 0: ret,filtered_vid_2 = cv2.threshold(filtered_vid_2,int(values['img2_binary_inv']),255,cv2.THRESH_BINARY_INV)
        if values['img2_trunc'] != 0: ret,filtered_vid_2 = cv2.threshold(filtered_vid_2,int(values['img2_trunc']),255,cv2.THRESH_TRUNC)
        if values['img2_tozero'] != 0: ret,filtered_vid_2 = cv2.threshold(filtered_vid_2,int(values['img2_tozero']),255,cv2.THRESH_TOZERO)
        if values['img2_tozero_inv'] != 0: ret,filtered_vid_2 = cv2.threshold(filtered_vid_2,int(values['img2_tozero_inv']),255,cv2.THRESH_TOZERO_INV)

        if values['img2_erosion'] != 0:
            kernel = np.ones((int(values['img2_erosion']),int(values['img2_erosion'])),np.uint8)
            filtered_vid_2 = cv2.erode(filtered_vid_2,kernel,iterations=int(values['img2_erosion_iteration']))
        if values['img2_dilation'] != 0:
            kernel = np.ones((int(values['img2_dilation']),int(values['img2_dilation'])),np.uint8)
            filtered_vid_2 = cv2.dilate(filtered_vid_2,kernel,iterations=int(values['img2_dilation_iteration']))
        if values['img2_opening'] != 0:
            kernel = np.ones((int(values['img2_opening']),int(values['img2_opening'])),np.uint8)
            filtered_vid_2 = cv2.morphologyEx(filtered_vid_2,cv2.MORPH_OPEN,kernel)
        if values['img2_closing'] != 0:
            kernel = np.ones((int(values['img2_closing']),int(values['img2_closing'])),np.uint8)
            filtered_vid_2 = cv2.morphologyEx(filtered_vid_2,cv2.MORPH_CLOSE,kernel)
        if values['img2_gradiant'] != 0:
            kernel = np.ones((int(values['img2_gradiant']),int(values['img2_gradiant'])),np.uint8)
            filtered_vid_2 = cv2.morphologyEx(filtered_vid_2,cv2.MORPH_GRADIENT,kernel)
        if values['img2_tophat'] != 0:
            kernel = np.ones((int(values['img2_tophat']),int(values['img2_tophat'])),np.uint8)
            filtered_vid_2 = cv2.morphologyEx(filtered_vid_2,cv2.MORPH_TOPHAT,kernel)
        if values['img2_blackhat'] != 0:
            kernel = np.ones((int(values['img2_blackhat']),int(values['img2_blackhat'])),np.uint8)
            filtered_vid_2 = cv2.morphologyEx(filtered_vid_2,cv2.MORPH_BLACKHAT,kernel)
        if values['img2_mean_on_off'] is True:
            filtered_vid_2 = cv2.adaptiveThreshold(filtered_vid_2,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,int(values['img2_mean_block_size']),int(values['img2_mean_constant']))
        if values['img2_canny1'] != 0 and values['img2_canny2'] != 0:
            filtered_vid_2 = cv2.Canny(filtered_vid_2,values['img2_canny1'],values['img2_canny2'])
        if values['img2_sobel'] != 0 and (values['img2_sobel_dx'] or values['img2_sobel_dy']) and values['img2_sobel'] % 2 != 0 and values['img2_sobel']<31 :
            filtered_vid_2 = cv2.Sobel(filtered_vid_2,ddepth=cv2.CV_8U ,dx = int(values['img2_sobel_dx']), dy = int(values['img2_sobel_dy']), ksize = int(values['img2_sobel']),scale=1,delta=0)
        filtered_vid_2_vidbytes = cv2.imencode('.png', filtered_vid_2)[1].tobytes()  
        window['filtered_vid_2'].update(data = filtered_vid_2_vidbytes)


        
        '''Final Video'''
        if event == 'o-f1':
            final_state = 'o-f1'

        if event == 'o-f2':
            final_state = 'o-f2'
        
        if event == 'f1-f2':
            final_state = 'f1-f2'

        if event == 'f1-o':
            final_state = 'f1-o'

        if event == 'f2-o':
            final_state = 'f2-o'

        if event == 'f2-f1':
            final_state = 'f2-f1'

        final_vid = change_final()
        if values['img_final_2d'] != 0:
            kernel = np.ones((int(values['img_final_2d']),int(values['img_final_2d'])),np.float32)/(int(values['img_final_2d'])*int(values['img_final_2d']))
            final_vid = cv2.filter2D(final_vid,-1,kernel)
        if values['img_final_blur'] != 0: final_vid = cv2.blur(final_vid,(int(values['img_final_blur']),int(values['img_final_blur'])))
        if values['img_final_gaussian'] != 0:
            if values['img_final_gaussian'] % 2 != 0:
                final_vid = cv2.GaussianBlur(final_vid,(int(values['img_final_gaussian']),int(values['img_final_gaussian'])),0)
        if values['img_final_median'] != 0: 
            if values['img_final_median'] % 2 != 0:
                final_vid = cv2.medianBlur(final_vid,int(values['img_final_median']))
        if values['img_final_bilateral'] != 0: final_vid = cv2.bilateralFilter(final_vid,int(values['img_final_bilateral']),75,75)
        if values['img_final_binary'] != 0: ret,final_vid = cv2.threshold(final_vid,int(values['img_final_binary']),255,cv2.THRESH_BINARY)
        if values['img_final_binary_inv'] != 0: ret,final_vid = cv2.threshold(final_vid,int(values['img_final_binary_inv']),255,cv2.THRESH_BINARY_INV)
        if values['img_final_trunc'] != 0: ret,final_vid = cv2.threshold(final_vid,int(values['img_final_trunc']),255,cv2.THRESH_TRUNC)
        if values['img_final_tozero'] != 0: ret,final_vid = cv2.threshold(final_vid,int(values['img_final_tozero']),255,cv2.THRESH_TOZERO)
        if values['img_final_tozero_inv'] != 0: ret,final_vid = cv2.threshold(final_vid,int(values['img_final_tozero_inv']),255,cv2.THRESH_TOZERO_INV)

        if values['img_final_erosion'] != 0:
            kernel = np.ones((int(values['img_final_erosion']),int(values['img_final_erosion'])),np.uint8)
            final_vid = cv2.erode(final_vid,kernel,iterations=int(values['img_final_erosion_iteration']))
        if values['img_final_dilation'] != 0:
            kernel = np.ones((int(values['img_final_dilation']),int(values['img_final_dilation'])),np.uint8)
            final_vid = cv2.dilate(final_vid,kernel,iterations=int(values['img_final_dilation_iteration']))
        if values['img_final_opening'] != 0:
            kernel = np.ones((int(values['img_final_opening']),int(values['img_final_opening'])),np.uint8)
            final_vid = cv2.morphologyEx(final_vid,cv2.MORPH_OPEN,kernel)
        if values['img_final_closing'] != 0:
            kernel = np.ones((int(values['img_final_closing']),int(values['img_final_closing'])),np.uint8)
            final_vid = cv2.morphologyEx(final_vid,cv2.MORPH_CLOSE,kernel)
        if values['img_final_gradiant'] != 0:
            kernel = np.ones((int(values['img_final_gradiant']),int(values['img_final_gradiant'])),np.uint8)
            final_vid = cv2.morphologyEx(final_vid,cv2.MORPH_GRADIENT,kernel)
        if values['img_final_tophat'] != 0:
            kernel = np.ones((int(values['img_final_tophat']),int(values['img_final_tophat'])),np.uint8)
            final_vid = cv2.morphologyEx(final_vid,cv2.MORPH_TOPHAT,kernel)
        if values['img_final_blackhat'] != 0:
            kernel = np.ones((int(values['img_final_blackhat']),int(values['img_final_blackhat'])),np.uint8)
            final_vid = cv2.morphologyEx(final_vid,cv2.MORPH_BLACKHAT,kernel)
        if values['img_final_mean_on_off'] is True:
            final_vid = cv2.adaptiveThreshold(final_vid,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,int(values['img_final_mean_block_size']),int(values['img_final_mean_constant']))
        if values['img_final_canny1'] != 0 and values['img_final_canny2'] != 0:
            final_vid = cv2.Canny(final_vid,values['img_final_canny1'],values['img_final_canny2'])
        if values['img_final_sobel'] != 0 and (values['img_final_sobel_dx'] or values['img_final_sobel_dy']) and values['img_final_sobel'] % 2 != 0 and values['img_final_sobel']<31 :
            final_vid = cv2.Sobel(final_vid,ddepth=cv2.CV_8U ,dx = int(values['img_final_sobel_dx']), dy = int(values['img_final_sobel_dy']), ksize = int(values['img_final_sobel']),scale=1,delta=0)
        final_vid_vidbytes = cv2.imencode('.png', final_vid)[1].tobytes()
        window['final_vid'].update(data = final_vid_vidbytes)

    if event == 'reset':
        window['img1_erosion_iteration'].update(1)
        window['img1_dilation_iteration'].update(1)
        window['img1_2d'].update(0)
        window['img1_blur'].update(0)
        window['img1_gaussian'].update(0)
        window['img1_median'].update(0)
        window['img1_bilateral'].update(0)
        window['img1_mean_block_size'].update(0)
        window['img1_binary'].update(0)
        window['img1_binary_inv'].update(0)
        window['img1_trunc'].update(0)
        window['img1_tozero'].update(0)
        window['img1_tozero_inv'].update(0)
        window['img1_erosion'].update(0)
        window['img1_dilation'].update(0)
        window['img1_opening'].update(0)
        window['img1_closing'].update(0)
        window['img1_gradiant'].update(0)
        window['img1_tophat'].update(0)
        window['img1_blackhat'].update(0)
        window['img2_erosion_iteration'].update(1)
        window['img2_dilation_iteration'].update(1)
        window['img2_2d'].update(0)
        window['img2_blur'].update(0)
        window['img2_gaussian'].update(0)
        window['img2_median'].update(0)
        window['img2_bilateral'].update(0)
        window['img2_mean_block_size'].update(0)
        window['img2_binary'].update(0)
        window['img2_binary_inv'].update(0)
        window['img2_trunc'].update(0)
        window['img2_tozero'].update(0)
        window['img2_tozero_inv'].update(0)
        window['img2_erosion'].update(0)
        window['img2_dilation'].update(0)
        window['img2_opening'].update(0)
        window['img2_closing'].update(0)
        window['img2_gradiant'].update(0)
        window['img2_tophat'].update(0)
        window['img2_blackhat'].update(0)
        window['img_final_erosion_iteration'].update(1)
        window['img_final_dilation_iteration'].update(1)
        window['img_final_2d'].update(0)
        window['img_final_blur'].update(0)
        window['img_final_gaussian'].update(0)
        window['img_final_median'].update(0)
        window['img_final_bilateral'].update(0)
        window['img_final_mean_block_size'].update(0)
        window['img_final_binary'].update(0)
        window['img_final_binary_inv'].update(0)
        window['img_final_trunc'].update(0)
        window['img_final_tozero'].update(0)
        window['img_final_tozero_inv'].update(0)
        window['img_final_erosion'].update(0)
        window['img_final_dilation'].update(0)
        window['img_final_opening'].update(0)
        window['img_final_closing'].update(0)
        window['img_final_gradiant'].update(0)
        window['img_final_tophat'].update(0)
        window['img_final_blackhat'].update(0)

