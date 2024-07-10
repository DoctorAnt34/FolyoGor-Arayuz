import PySimpleGUI as sg
import cv2
import numpy as np
import copy

sg.theme('Default1')

org_img = sg.Image(filename=None, key="org_img")
filtered_img_1 = sg.Image(filename=None, key="filtered_img_1")
filtered_img_2 = sg.Image(filename=None, key="filtered_img_2")
final_img = sg.Image(filename=None, key="final_img")

org_vid = sg.Image(filename=None, key="org_vid")
filtered_vid_1 = sg.Image(filename=None, key="filtered_vid_1")
filtered_vid_2 = sg.Image(filename=None, key="filtered_vid_2")
final_vid = sg.Image(filename=None, key="final_vid")


img_frame = sg.pin(sg.Frame('Picture',[[org_img,filtered_img_1,filtered_img_2,final_img]],key='pic_frame',visible=True))

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
    [sg.Text('2D Convolution'),sg.Slider(range=(0,10),orientation='h',key='img1_2d',enable_events=True)],
    [sg.Text('Image Blurring'),sg.Slider(range=(0,10),orientation='h',key='img1_blur',enable_events=True)],
    [sg.Text('Gaussian Blurring'),sg.Slider(range=(0,10),orientation='h',key='img1_gaussian',enable_events=True)],
    [sg.Text('Median Blurring'),sg.Slider(range=(0,10),orientation='h',key='img1_median',enable_events=True)],
    [sg.Text('Bilateral Filtering'),sg.Slider(range=(0,10),orientation='h',key='img1_bilateral',enable_events=True)],
])
img1_adap_thres_col=sg.Frame('Adaptive Thresholding',[
    [sg.Text('Mean'),sg.Radio('ON','MEAN',key='img1_mean_on_off',enable_events=True),sg.Radio('OFF','MEAN'),sg.Text('Block Size'),sg.Slider(range=(0,20),orientation='h',key='img1_mean_block_size',enable_events=True),sg.Text('Constant'),sg.Slider(range=(0,20),orientation='h',key='img1_mean_constant',enable_events=True)],
    [],
    [],
    [sg.Text('Gaussian'),sg.Radio('ON','GAUSS',key='img1_gaussian',visible=False)]
])
img1_thres_col = sg.Frame('Tresholding',[
    [sg.Text('Binary'),sg.Slider(range=(0,255),orientation='h',key='img1_binary',enable_events=True,)],
    [sg.Text('Binary INV'),sg.Slider(range=(0,255),orientation='h',key='img1_binary_inv',enable_events=True)],
    [sg.Text('Trunc'),sg.Slider(range=(0,255),orientation='h',key='img1_trunc',enable_events=True)],
    [sg.Text('TOZERO'),sg.Slider(range=(0,255),orientation='h',key='img1_tozero',enable_events=True)],
    [sg.Text('TOZERO INV'),sg.Slider(range=(0,255),orientation='h',key='img1_tozero_inv',enable_events=True)],
])

img1_morph_col = sg.Frame('Morphological Transformation',[
    [sg.Text('Erosion'),sg.Slider(range=(0,10),orientation='h',key='img1_erosion',enable_events=True)],
    [sg.Text('Dilation'),sg.Slider(range=(0,10),orientation='h',key='img1_dilation',enable_events=True)],
    [sg.Text('Opening'),sg.Slider(range=(0,10),orientation='h',key='img1_opening',enable_events=True)],
    [sg.Text('Closing'),sg.Slider(range=(0,10),orientation='h',key='img1_closing',enable_events=True)],
    [sg.Text('Morphological Gradient'),sg.Slider(range=(0,10),orientation='h',key='img1_gradiant',enable_events=True)],
    [sg.Text('Top Hat'),sg.Slider(range=(0,10),orientation='h',key='img1_tophat',enable_events=True)],
    [sg.Text('Black Hat'),sg.Slider(range=(0,10),orientation='h',key='img1_blackhat',enable_events=True)],
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
    [sg.Text('2D Convolution'),sg.Slider(range=(0,10),orientation='h',key='img2_2d',enable_events=True)],
    [sg.Text('Image Blurring'),sg.Slider(range=(0,10),orientation='h',key='img2_blur',enable_events=True)],
    [sg.Text('Gaussian Blurring'),sg.Slider(range=(0,10),orientation='h',key='img2_gaussian',enable_events=True)],
    [sg.Text('Median Blurring'),sg.Slider(range=(0,10),orientation='h',key='img2_median',enable_events=True)],
    [sg.Text('Bilateral Filtering'),sg.Slider(range=(0,10),orientation='h',key='img2_bilateral',enable_events=True)],
])
img2_adap_thres_col=sg.Frame('Adaptive Thresholding',[
    [sg.Text('Mean'),sg.Radio('ON','MEAN',key='img2_mean_on_off',enable_events=True),sg.Radio('OFF','MEAN'),sg.Text('Block Size'),sg.Slider(range=(0,20),orientation='h',key='img2_mean_block_size',enable_events=True),sg.Text('Constant'),sg.Slider(range=(0,20),orientation='h',key='img2_mean_constant',enable_events=True)],
    [],
    [],
    [sg.Text('Gaussian'),sg.Radio('ON','GAUSS',key='img2_gaussian',visible=False)]
])
img2_thres_col = sg.Frame('Tresholding',[
    [sg.Text('Binary'),sg.Slider(range=(0,255),orientation='h',key='img2_binary',enable_events=True,)],
    [sg.Text('Binary INV'),sg.Slider(range=(0,255),orientation='h',key='img2_binary_inv',enable_events=True)],
    [sg.Text('Trunc'),sg.Slider(range=(0,255),orientation='h',key='img2_trunc',enable_events=True)],
    [sg.Text('TOZERO'),sg.Slider(range=(0,255),orientation='h',key='img2_tozero',enable_events=True)],
    [sg.Text('TOZERO INV'),sg.Slider(range=(0,255),orientation='h',key='img2_tozero_inv',enable_events=True)],
])

img2_morph_col = sg.Frame('Morphological Transformation',[
    [sg.Text('Erosion'),sg.Slider(range=(0,10),orientation='h',key='img2_erosion',enable_events=True)],
    [sg.Text('Dilation'),sg.Slider(range=(0,10),orientation='h',key='img2_dilation',enable_events=True)],
    [sg.Text('Opening'),sg.Slider(range=(0,10),orientation='h',key='img2_opening',enable_events=True)],
    [sg.Text('Closing'),sg.Slider(range=(0,10),orientation='h',key='img2_closing',enable_events=True)],
    [sg.Text('Morphological Gradient'),sg.Slider(range=(0,10),orientation='h',key='img2_gradiant',enable_events=True)],
    [sg.Text('Top Hat'),sg.Slider(range=(0,10),orientation='h',key='img2_tophat',enable_events=True)],
    [sg.Text('Black Hat'),sg.Slider(range=(0,10),orientation='h',key='img2_blackhat',enable_events=True)],
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
    [sg.Text('2D Convolution'),sg.Slider(range=(0,10),orientation='h',key='img_final_2d',enable_events=True)],
    [sg.Text('Image Blurring'),sg.Slider(range=(0,10),orientation='h',key='img_final_blur',enable_events=True)],
    [sg.Text('Gaussian Blurring'),sg.Slider(range=(0,10),orientation='h',key='img_final_gaussian',enable_events=True)],
    [sg.Text('Median Blurring'),sg.Slider(range=(0,10),orientation='h',key='img_final_median',enable_events=True)],
    [sg.Text('Bilateral Filtering'),sg.Slider(range=(0,10),orientation='h',key='img_final_bilateral',enable_events=True)],
])
img_final_adap_thres_col=sg.Frame('Adaptive Thresholding',[
    [sg.Text('Mean'),sg.Radio('ON','MEAN',key='img_final_mean_on_off',enable_events=True),sg.Radio('OFF','MEAN'),sg.Text('Block Size'),sg.Slider(range=(0,20),orientation='h',key='img_final_mean_block_size',enable_events=True),sg.Text('Constant'),sg.Slider(range=(0,20),orientation='h',key='img_final_mean_constant',enable_events=True)],
    [],
    [],
    [sg.Text('Gaussian'),sg.Radio('ON','GAUSS',key='img_final_gaussian',visible=False)]
])
img_final_thres_col = sg.Frame('Tresholding',[
    [sg.Text('Binary'),sg.Slider(range=(0,255),orientation='h',key='img_final_binary',enable_events=True,)],
    [sg.Text('Binary INV'),sg.Slider(range=(0,255),orientation='h',key='img_final_binary_inv',enable_events=True)],
    [sg.Text('Trunc'),sg.Slider(range=(0,255),orientation='h',key='img_final_trunc',enable_events=True)],
    [sg.Text('TOZERO'),sg.Slider(range=(0,255),orientation='h',key='img_final_tozero',enable_events=True)],
    [sg.Text('TOZERO INV'),sg.Slider(range=(0,255),orientation='h',key='img_final_tozero_inv',enable_events=True)],
])

img_final_morph_col = sg.Frame('Morphological Transformation',[
    [sg.Text('Erosion'),sg.Slider(range=(0,10),orientation='h',key='img_final_erosion',enable_events=True)],
    [sg.Text('Dilation'),sg.Slider(range=(0,10),orientation='h',key='img_final_dilation',enable_events=True)],
    [sg.Text('Opening'),sg.Slider(range=(0,10),orientation='h',key='img_final_opening',enable_events=True)],
    [sg.Text('Closing'),sg.Slider(range=(0,10),orientation='h',key='img_final_closing',enable_events=True)],
    [sg.Text('Morphological Gradient'),sg.Slider(range=(0,10),orientation='h',key='img_final_gradiant',enable_events=True)],
    [sg.Text('Top Hat'),sg.Slider(range=(0,10),orientation='h',key='img_final_tophat',enable_events=True)],
    [sg.Text('Black Hat'),sg.Slider(range=(0,10),orientation='h',key='img_final_blackhat',enable_events=True)],
])


img_final_layout =[
    [img_final_smoot_col,img_final_thres_col,img_final_morph_col,img_final_iter_col,img_final_adap_thres_col]
]





'''Tab Group'''

tabs = sg.TabGroup([[sg.Tab('Image 1',img1_layout), sg.Tab('Image 2', img2_layout), sg.Tab('Final Image', img_final_layout)]])

'''Navigate Buttons'''

feed_chage_radio = sg.Radio('Picture','feed_change',default=True,enable_events=True,key='feed_pic'),sg.Radio('Video','feed_change',enable_events=True,key='feed_vid')




layout=[
    [img_frame,vid_frame],
    [feed_chage_radio,sg.pin(sg.Button('Take Picture',key='take_pic',visible=False)),sg.pin(sg.Button('Browse',key='browse',visible=True)),sg.Button('Reset',key='reset'),sg.Button('Save',key='save')],
    [tabs]
]


window = sg.Window('FolyoGör',layout,grab_anywhere=False,finalize=True)
window.maximize()
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
feed = 'pic'


while True:
    event, values = window.read(timeout=20)

    if event == sg.WIN_CLOSED or event == 'Exit':
        break

    if event == 'feed_pic':
        feed = 'pic'
        window['browse'].update(visible = True)
        window['take_pic'].update(visible = False)
        window['pic_frame'].update(visible = True)
        window['vid_frame'].update(visible = False)

    if event == 'feed_vid':
        feed = 'vid'
        window['browse'].update(visible = False)
        window['take_pic'].update(visible = True)
        window['pic_frame'].update(visible = False)
        window['vid_frame'].update(visible = True)

    if event == 'browse':
        file_path = sg.popup_get_file('Açılacak Resmi Seçin',file_types=[('Images','*.png'),('Images','*.jpg')])
        img_o = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
        img_o = crop_square(img_o,500)
        imgbytes = cv2.imencode('.png', img_o)[1].tobytes()  
        window['filtered_img_1'].update(data=imgbytes)
        window['filtered_img_2'].update(data=imgbytes)
        window['org_img'].update(data=imgbytes)
        img = copy.deepcopy(img_o)
        img_s = cv2.subtract(img_o,img)
        imgbytes_s = cv2.imencode('.png', img_s)[1].tobytes()
        window['final_img'].update(data=imgbytes_s)

    if feed == 'vid':
        ret, frame = cap.read()
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto

        # Orjinal görüntü
        nparr = np.fromstring(imgbytes, np.uint8)
        img_o = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        #img_o = cv2.resize(img_o,[int(3840/6),int(2160/6)])
        o_imgbytes = cv2.imencode('.png', img_o)[1].tobytes()
        window['org_vid'].update(data=o_imgbytes)
        img = copy.deepcopy(img_o)
