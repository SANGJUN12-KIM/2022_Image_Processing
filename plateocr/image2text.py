# -*- coding: utf-8 -*-
import json
import os
import numpy
from collections import OrderedDict
from numpyencoder import NumpyEncoder
from PIL import Image
import pandas as pd
import csv

global reader

def output(extracted_data, pages_num, image_name,  width, height, output_path):
    file_name = os.path.splitext(image_name)[0]
    file_data = OrderedDict()
    file_data['page'] = pages_num
    file_data['width'] = width
    file_data['height'] = height
    item_list = []

    if output_path != None:
        for id, (area, text, credibility) in enumerate(extracted_data):
            item = {}
            item['id'] = id + 1
            item['order'] = id + 1
            item['type'] = 'text'

            area_2points = area[0], area[2]
            item['area'] = area_2points

            text = text
            item['content'] = text

            item['credibility'] = credibility
            item['url'] = ''
            item_list.append(item)

        file_data['object'] = item_list

        root_dir = output_path
        if os.path.isdir(root_dir) == False:
            os.mkdir(root_dir)
        os.chdir(root_dir)
        with open(file_name + '.json', 'w', encoding="utf-8") as make_file:
            json.dump(file_data, make_file, ensure_ascii=False, cls=NumpyEncoder, indent="\t")
        os.chdir(os.path.join(os.pardir))

def convert(reader, link_threshold, width_ths, input_dir = './target_pages/', json_output_dir = './OCRresult', paragraph= False):
    IMAGE_PATH = input_dir
    try:
        pages_num = 0
        IMAGE_PATH = input_dir
        img = Image.open(IMAGE_PATH)
        width = img.size[0]
        height = img.size[1]
        print("{} 처리중".format(IMAGE_PATH))

        if paragraph == True:
            result = reader.readtext(IMAGE_PATH, link_threshold, width_ths, paragraph=True)
            for ocr_data in result:
                ocr_data = ocr_data.append('')
        else:
            result = reader.readtext(IMAGE_PATH, link_threshold, width_ths)

        image_file = os.path.basename(IMAGE_PATH)

        output(result, pages_num, image_file, width, height, json_output_dir)
    except ValueError:
        print("'{}'는 지원하는 이미지 파일이 아닙니다.".format(IMAGE_PATH))
        pass
    except Image.UnidentifiedImageError:
        print("'{}'는 지원하는 이미지 파일이 아닙니다.".format(IMAGE_PATH))
        pass

    ocr_result = []

    for id, (area, text, credibility) in enumerate(result):
        area_2points = area[0], area[2]
        text = text
        credibility = credibility
        ocr_data = []
        ocr_data.append(area_2points)
        ocr_data.append(text)
        ocr_data.append(credibility)

        ocr_result.append(ocr_data)

    return ocr_result

def read_text_area(reader, input_file = './target_pages/0001.jpg'):

    """
    if custom_model == True:
        user_network_path = os.path.dirname(easyocr.__file__)
        reader = easyocr.Reader(['ko', 'en'], gpu=True, model_storage_directory= user_network_path+'/user_network/',
                                user_network_directory=user_network_path+'/user_network/', recog_network='custom')
    else:
        reader = easyocr.Reader(['ko', 'en'], gpu=True)
    """
    IMAGE_PATH = input_file
    try:
        pages_num = 0
        IMAGE_PATH = input_file
        print("{} 처리중".format(IMAGE_PATH))
        result = reader.readtext(IMAGE_PATH)

    except ValueError:
        print("'{}'는 지원하는 이미지 파일이 아닙니다.".format(IMAGE_PATH))
        pass
    except Image.UnidentifiedImageError:
        print("'{}'는 지원하는 이미지 파일이 아닙니다.".format(IMAGE_PATH))
        pass

    ocr_result = []
    whole_text = ""
    whole_credibility = []
    for id, (area, text, credibility) in enumerate(result):
        text = text
        credibility = float(credibility)
        for parted_text in text:
            whole_text += parted_text

        whole_credibility.append(credibility)

    ocr_result.append(whole_text)
    avg_credibility = numpy.mean(whole_credibility)
    ocr_result.append(avg_credibility)
    return tuple(ocr_result)

def conv2txt(reader, input_dir = './target_pages/', txt_output_dir = './OCRresultTXT'):

    """
    if custom_model == True:
        user_network_path = os.path.dirname(easyocr.__file__)
        reader = easyocr.Reader(['ko', 'en'], gpu=True, model_storage_directory= user_network_path+'/user_network/',
                                user_network_directory=user_network_path+'/user_network/', recog_network='custom')
    else:
        reader = easyocr.Reader(['ko', 'en'], gpu=True)
    """

    if os.path.isdir(input_dir) == True:
        pages_num = 0
        file_list = os.listdir(input_dir)
        print("input_dir = '{}' , {}".format(input_dir, file_list))

        for image_file in sorted(os.listdir(input_dir)):
            try:
                pages_num += 1
                IMAGE_PATH = input_dir + image_file
                print("{} 처리중".format(image_file))
                result = ''.join(reader.readtext(IMAGE_PATH, detail=0))

                root_dir = txt_output_dir
                if os.path.isdir(root_dir) == False:
                    os.mkdir(root_dir)
                os.chdir(root_dir)
                with open(os.path.splitext(image_file)[0] + '.txt', 'w', encoding="utf-8") as f:
                    f.write(result)
                os.chdir(os.path.join(os.pardir))

            except ValueError:
                print("'{}'는 지원하는 이미지 파일이 아닙니다.".format(IMAGE_PATH))
                pages_num -= 1
                pass
            except Image.UnidentifiedImageError:
                print("'{}'는 지원하는 이미지 파일이 아닙니다.".format(IMAGE_PATH))
                pages_num -= 1
                pass
            except FileNotFoundError:
                print("디렉토리 '{}'를 찾을 수 없습니다.".format(IMAGE_PATH))

def evaluation(reader, test_dataset_dir, output_dir):
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)
    gt_txt_dir = test_dataset_dir+'/gt.txt'
    test_img_dir = test_dataset_dir+'/class/'
    file_name = []
    gt_list = []
    n_correct = 0
    length_of_data = 0
    df = pd.DataFrame()

    try:

        with open(gt_txt_dir, 'r') as f:
            gt_data = f.readlines()
            for line in gt_data:
                file_name.append(line.split("\t")[0])
                gt_list.append(line.split("\t")[1].strip())

        for i, image_file in enumerate(sorted(os.listdir(test_img_dir))):
            IMAGE_PATH = test_img_dir + '/' + image_file
            text_ocr_result = reader.readtext(IMAGE_PATH, detail=0)
            text_ocr_result = ''.join(text_ocr_result)

            if text_ocr_result == gt_list[i]:
                n_correct += 1
                syllable_match_socre = 1

            else:
                mismatch_syllable=[]

                text_ocr_syllable= text_ocr_result
                gt_data = ''.join(gt_list[i])

                for syllable in text_ocr_syllable:
                    if syllable not in gt_data:
                        mismatch_syllable.append(syllable)

                syllable_match_socre = (len(gt_data)- len(mismatch_syllable))/len(gt_data)

            length_of_data += 1

            data = {'파일명': file_name[i], '인식결과': text_ocr_result, '정답값': gt_list[i], '일치여부': text_ocr_result == gt_list[i], '음절단위 부분일치도': syllable_match_socre}
            df= df.append(data, ignore_index=True)
            print(data)

    finally:
        df.to_csv(output_dir + '/testResult.csv')
        mismatch_syllable_list = df['음절단위 부분일치도'] != 1
        mismatch_syllable_score = df[mismatch_syllable_list]

        avg_syllable_match_socre = mismatch_syllable_score['음절단위 부분일치도'].mean() * 100
        accuracy = n_correct / float(length_of_data) * 100


        if os.path.isfile(output_dir+ '/testResult.csv') == True:
            print('검증 데이터 갯수: ', length_of_data)
            print('일치도: ', accuracy)
            print('불일치시 음절단위 부분일치도: ', avg_syllable_match_socre)

            with open(output_dir + '/testResult.csv', 'a', newline='\n') as csvfile:
                wr = csv.writer(csvfile)
                wr.writerow(['검증 데이터 갯수', length_of_data])
                wr.writerow(['일치도', accuracy])
                wr.writerow(['불일치시 음절단위 부분일치도', avg_syllable_match_socre])

