import easyocr
import os

def model(custom_model = False):
    global reader
    if custom_model == True:
        user_network_path = os.path.dirname(easyocr.__file__)
        reader = easyocr.Reader(['ko'], gpu=True, model_storage_directory= user_network_path+'/user_network/',
                                user_network_directory=user_network_path+'/user_network/', recog_network='custom')
    else:
        reader = easyocr.Reader(['ko', 'en'], gpu=True)

    return reader