import pandas as pd
import numpy as np
from encodings import unicode_escape
import tensorflow as tf
import time
import string
import random
punc = string.punctuation
print(punc)
punc = punc +"。？，、/\\！；//【】（）~·《》：“”"
print(punc)
def remove_unlaw(st):
    res = ""
    length = len(st) - 2
    i = 0
    while i < length:
        if st[i: i + 3] == "^,^":
            i = i + 2
        else:
            res = res + st[i]
        i = i + 1
    while i < len(st):
        res = res + st[i]
        i = i + 1
    res = res.replace('^', '')
    return res
def get_keyword(nums=1, words=""):
    keyword_tuple_list = []
    keyword_list = words.split(',')
    for item in keyword_list:
        keyword_tuple_list.append((float(item[-8:]),item[:-9]))
    keyword_tuple_list.sort()
    keyword_tuple_list.reverse()
    result_list = []
    for i in range(nums):
        result_list.append(keyword_tuple_list[i][1])
    return result_list
def pre_process_train():
    df = pd.read_csv('../CTR2021/train_info.txt', header=None)
    user_info_dict = {}
    news_info_dict = {}
    user_info_df = pd.read_csv('../CTR2021/user_info.txt', sep='\t', header=None)
    news_info_df = pd.read_csv('../CTR2021/news_info.txt',  sep='\t', header=None)
    user_info_list = user_info_df.values
    news_info_list = news_info_df.values
    cou = 0
    Cou = [0, 0, 0, 0, 0, 0, 0]
    print(len(news_info_list))
    for i in range(len(news_info_list)):
        cou = cou + 1
        for j in range(len(news_info_list[i])):
            flag = False
            if type(news_info_list[i][j] == str):
                if news_info_list[i][j] != news_info_list[i][j]:
                    flag = True
            elif np.isnan(news_info_list[i][j]):
                    flag = True
            else:
                pass
            if flag:
                if j == 1:
                    news_info_list[i][j] = news_info_list[i - 1][j]
                while(news_info_list[i][1][-1] in punc):
                    news_info_list[i][1] = news_info_list[i][1][:-1]
                while(news_info_list[i][1][0] in punc):
                    news_info_list[i][1] = news_info_list[i][1][1:]
                if j == 2 or j == 3:
                    news_info_list[i][j] = (news_info_list[i - 1][j] + news_info_list[i - 2][j] + news_info_list[i - 3][j] + news_info_list[i - 4][j] + news_info_list[i - 5][j]) / 5
                if j == 4:
                    if news_info_list[i][6] == news_info_list[i][6]:
                        keyword_list = get_keyword(1, news_info_list[i][6])
                        news_info_list[i][j] = keyword_list[0]
                    else:
                        news_info_list[i][j] = news_info_list[i][1][-2:]
                if j == 5:
                    if news_info_list[i][6] == news_info_list[i][6]:
                        keyword_list = get_keyword(2, news_info_list[i][6])
                        news_info_list[i][j] = keyword_list[0] + '/' + keyword_list[1]
                    else:
                        news_info_list[i][j] = news_info_list[i][1][-2:] + "/" + news_info_list[i][1][:2]
                if j == 6:
                    str_list = news_info_list[i][5].split('/')
                    news_info_list[i][j] = str_list[-1]
                Cou[j] = Cou[j] + 1
            else:
                if j == 6:
                    news_info_list[i][j] = remove_unlaw(news_info_list[i][j])
                    news_in = get_keyword(1, news_info_list[i][j])
                    news_info_list[i][j] = news_in[0]
        news_id = news_info_list[i][0]
        publish_time = news_info_list[i][2]
        publish_time = int(publish_time) // 1000
        local_publish_time = time.localtime(publish_time)
        dt = time.strftime("%Y-%m-%d %H:%M:%S", local_publish_time)
        timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
        publish_hour = timeArray.tm_hour
        pubulish_time_type = publish_hour // 6
        class_list = news_info_list[i][5].split("/")
        second_class = class_list[-1]
        news_info_dict[news_id] = {"publish_time": pubulish_time_type, "picture_nums": news_info_list[i][3],
                                   "first_class": convert_chinese(news_info_list[i][4]),
                                   "second_class": convert_chinese(second_class),
                                   "keyword": convert_chinese(news_info_list[i][6])}
    print(cou)
    cou = 0
    with open('../CTR2021/user_info.txt', encoding='utf-8') as f:
        age_prob = [0, 0, 0, 0]
        line = f.readline()
        while line:
            flag = False
            s_split_list = line.split('\t')
            q = 0
            if(len(s_split_list) < 7):
                flag = True
            for l in s_split_list:
                if l == "":
                    flag = True
                    break
                q = q + 1
            if flag:
                line = f.readline()
                continue
            cou = cou + 1
            user_id = s_split_list[0]
            age_str = s_split_list[5]
            sex_str = s_split_list[6]
            age_list = age_str.split(',')
            max_pro = "0.000000"
            for i in range(4):
                age_prob[i] = float(age_list[i][:-8])
            sex_list = sex_str.split(',')
            if len(sex_list) == 1:
                print(sex_list)
                if(sex_list[0] == "\n"):
                    sex = 0.5
                elif sex_list[0][0] == 'm':
                    sex = 1 - float(sex_list[0][-9:-1])
                else:
                    sex = float(sex_list[0][-9:-1])
            else:
                sex = float(sex_list[0][-8:])
            user_info_dict[user_id] = {'device': s_split_list[1], 'os': s_split_list[2], 'province': convert_chinese(s_split_list[3]),
                                       'city': convert_chinese(s_split_list[4]), 'age0': age_prob[0], 'age1': age_prob[1], 'age2': age_prob[2], 'age3': age_prob[3], 'sex': sex}
            line = f.readline()

    print(cou)

    content_list = df.values
    user_id_list = []
    news_id_list = []
    internet_env_list = []
    f5_times_list = []
    pos_list = []
    is_click_list = []
    duration_list = []
    age0_list = []
    age1_list = []
    age2_list = []
    age3_list = []
    sex_list = []
    device_list = []
    ope_list = []
    picture_nums_list = []
    publish_time_type_list = []
    view_time_type_list = []
    province_list = []
    city_list = []
    first_class_list = []
    second_class_list = []
    keyword_list = []

    cou = 1
    for content in content_list:
        s = content[0]
        if cou % 5000000 == 0:
            print(cou)
        if cou > 20000000:
            break
        cou = cou + 1
        s_split_list = s.split('\t')
        user_id = s_split_list[0]
        doc_id = int(s_split_list[1])
        if user_id not in user_info_dict.keys() or doc_id not in news_info_dict.keys():
            continue
        user_id_list.append(user_id)
        news_id_list.append(s_split_list[1])
        view_time = s_split_list[2]
        view_time = int(view_time) // 1000
        local_publish_time = time.localtime(view_time)
        dt = time.strftime("%Y-%m-%d %H:%M:%S", local_publish_time)
        timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
        view_hour = timeArray.tm_hour
        view_time_type = view_hour // 6
        view_time_type_list.append(view_time_type)
        internet_env_list.append(s_split_list[3])
        f5_times_list.append(int(s_split_list[4]))
        pos_list.append(s_split_list[5])
        is_click_list.append(s_split_list[6])
        duration_list.append(int(s_split_list[7]))
        age0_list.append(user_info_dict[user_id]['age0'])
        age1_list.append(user_info_dict[user_id]['age1'])
        age2_list.append(user_info_dict[user_id]['age2'])
        age3_list.append(user_info_dict[user_id]['age3'])
        sex_list.append(user_info_dict[user_id]['sex'])
        device_list.append(user_info_dict[user_id]['device'])
        ope_list.append(user_info_dict[user_id]['os'])
        province_list.append(user_info_dict[user_id]['province'])
        city_list.append(user_info_dict[user_id]['city'])
        publish_time_type_list.append(news_info_dict[doc_id]["publish_time"])
        picture_nums_list.append(news_info_dict[doc_id]["picture_nums"])
        first_class_list.append(news_info_dict[doc_id]["first_class"])
        second_class_list.append(news_info_dict[doc_id]["second_class"])
        keyword_list.append(news_info_dict[doc_id]["keyword"])
    data = {'label': is_click_list, 'I1': f5_times_list, 'I2': picture_nums_list,'I3': age0_list,
            'I4': age1_list, 'I5': age2_list, 'I6': age3_list, 'I7': sex_list, 'C1': user_id_list,
            'C2': news_id_list, 'C3': internet_env_list,  'C4': device_list,
            'C5': ope_list, 'C6': publish_time_type_list, 'C7': province_list, 'C8': city_list,
            'C9': view_time_type_list, 'C10': first_class_list, 'C11': second_class_list, 'C12': keyword_list}
    df_data = pd.DataFrame(data)
    df_data.to_csv('../CTR2021/train_data.csv')
    print("write data success!")


def pre_process_test():
    df = pd.read_csv('../CTR2021/test_info.txt', header=None)
    user_info_dict = {}
    news_info_dict = {}
    user_info_df = pd.read_csv('../CTR2021/user_info.txt', sep='\t', header=None)
    news_info_df = pd.read_csv('../CTR2021/news_info.txt',  sep='\t', header=None)
    user_info_list = user_info_df.values
    news_info_list = news_info_df.values
    cou = 0
    Cou = [0, 0, 0, 0, 0, 0, 0]
    print(len(news_info_list))
    for i in range(len(news_info_list)):
        cou = cou + 1
        for j in range(len(news_info_list[i])):
            flag = False
            if type(news_info_list[i][j] == str):
                if news_info_list[i][j] != news_info_list[i][j]:
                    flag = True
            elif np.isnan(news_info_list[i][j]):
                    flag = True
            else:
                pass
            if flag:
                if j == 1:
                    news_info_list[i][j] = news_info_list[i - 1][j]
                while(news_info_list[i][1][-1] in punc):
                    news_info_list[i][1] = news_info_list[i][1][:-1]
                while(news_info_list[i][1][0] in punc):
                    news_info_list[i][1] = news_info_list[i][1][1:]
                if j == 2 or j == 3:
                    news_info_list[i][j] = (news_info_list[i - 1][j] + news_info_list[i - 2][j] + news_info_list[i - 3][j] + news_info_list[i - 4][j] + news_info_list[i - 5][j]) / 5
                if j == 4:
                    if news_info_list[i][6] == news_info_list[i][6]:
                        keyword_list = get_keyword(1, news_info_list[i][6])
                        news_info_list[i][j] = keyword_list[0]
                    else:
                        news_info_list[i][j] = news_info_list[i][1][-2:]
                if j == 5:
                    if news_info_list[i][6] == news_info_list[i][6]:
                        keyword_list = get_keyword(2, news_info_list[i][6])
                        news_info_list[i][j] = keyword_list[0] + '/' + keyword_list[1]
                    else:
                        news_info_list[i][j] = news_info_list[i][1][-2:] + "/" + news_info_list[i][1][:2]
                if j == 6:
                    str_list = news_info_list[i][5].split('/')
                    news_info_list[i][j] = str_list[-1]
                Cou[j] = Cou[j] + 1
            else:
                if j == 6:
                    news_info_list[i][j] = remove_unlaw(news_info_list[i][j])
                    news_in = get_keyword(1, news_info_list[i][j])
                    news_info_list[i][j] = news_in[0]
        news_id = news_info_list[i][0]
        publish_time = news_info_list[i][2]
        publish_time = int(publish_time) // 1000
        local_publish_time = time.localtime(publish_time)
        dt = time.strftime("%Y-%m-%d %H:%M:%S", local_publish_time)
        timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
        publish_hour = timeArray.tm_hour
        pubulish_time_type = publish_hour // 6
        class_list = news_info_list[i][5].split("/")
        second_class = class_list[-1]
        news_info_dict[news_id] = {"publish_time": pubulish_time_type, "picture_nums": news_info_list[i][3],
                                   "first_class": convert_chinese(news_info_list[i][4]),
                                   "second_class": convert_chinese(second_class),
                                   "keyword": convert_chinese(news_info_list[i][6])}
    print(cou)
    cou = 0
    with open('../CTR2021/user_info.txt', encoding='utf-8') as f:
        age_prob = [0, 0, 0, 0]
        line = f.readline()
        while line:
            flag = False
            s_split_list = line.split('\t')
            q = 0
            if(len(s_split_list) < 7):
                flag = True
            for l in s_split_list:
                if l == "":
                    flag = True
                    break
                q = q + 1
            if flag:
                line = f.readline()
                continue
            cou = cou + 1
            user_id = s_split_list[0]
            age_str = s_split_list[5]
            sex_str = s_split_list[6]
            age_list = age_str.split(',')
            max_pro = "0.000000"
            for i in range(4):
                age_prob[i] = float(age_list[i][-8:])
            sex_list = sex_str.split(',')
            if len(sex_list) == 1:
                print(sex_list)
                if(sex_list[0] == "\n"):
                    sex = 0.5
                elif sex_list[0][0] == 'm':
                    sex = 1 - float(sex_list[0][-9:-1])
                else:
                    sex = float(sex_list[0][-9:-1])
            else:
                sex = float(sex_list[0][-8:])
            user_info_dict[user_id] = {'device': s_split_list[1], 'os': s_split_list[2], 'province': convert_chinese(s_split_list[3]),
                                       'city': convert_chinese(s_split_list[4]), 'age0': age_prob[0], 'age1': age_prob[1], 'age2': age_prob[2], 'age3': age_prob[3], 'sex': sex}
            line = f.readline()

    print(cou)

    content_list = df.values
    user_id_list = []
    news_id_list = []
    internet_env_list = []
    f5_times_list = []
    pos_list = []
    is_click_list = []
    duration_list = []
    age0_list = []
    age1_list = []
    age2_list = []
    age3_list = []
    sex_list = []
    device_list = []
    ope_list = []
    picture_nums_list = []
    publish_time_type_list = []
    view_time_type_list = []
    province_list = []
    city_list = []
    first_class_list = []
    second_class_list = []
    keyword_list = []

    cou = 1
    user_queshi = 0
    news_queshi = 0
    for content in content_list:
        s = content[0]
        cou = cou + 1
        s_split_list = s.split('\t')
        user_id = s_split_list[1]
        doc_id = int(s_split_list[2])
        if user_id not in user_info_dict.keys():
            user_id = random.choice(list(user_info_dict))
            print(user_id)
        user_id_list.append(user_id)
        news_id_list.append(s_split_list[2])
        view_time = s_split_list[3]
        view_time = int(view_time) // 1000
        local_publish_time = time.localtime(view_time)
        dt = time.strftime("%Y-%m-%d %H:%M:%S", local_publish_time)
        timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
        view_hour = timeArray.tm_hour
        view_time_type = view_hour // 6
        view_time_type_list.append(view_time_type)
        internet_env_list.append(s_split_list[4])
        f5_times_list.append(int(s_split_list[5]))
        is_click_list.append(0)
        age0_list.append(user_info_dict[user_id]['age0'])
        age1_list.append(user_info_dict[user_id]['age1'])
        age2_list.append(user_info_dict[user_id]['age2'])
        age3_list.append(user_info_dict[user_id]['age3'])
        sex_list.append(user_info_dict[user_id]['sex'])
        device_list.append(user_info_dict[user_id]['device'])
        ope_list.append(user_info_dict[user_id]['os'])
        province_list.append(user_info_dict[user_id]['province'])
        city_list.append(user_info_dict[user_id]['city'])
        publish_time_type_list.append(news_info_dict[doc_id]["publish_time"])
        picture_nums_list.append(news_info_dict[doc_id]["picture_nums"])
        first_class_list.append(news_info_dict[doc_id]["first_class"])
        second_class_list.append(news_info_dict[doc_id]["second_class"])
        keyword_list.append(news_info_dict[doc_id]["keyword"])
    data = {'label': is_click_list, 'I1': f5_times_list, 'I2': picture_nums_list,'I3': age0_list,
            'I4': age1_list, 'I5': age2_list, 'I6': age3_list, 'I7': sex_list, 'C1': user_id_list,
            'C2': news_id_list, 'C3': internet_env_list,  'C4': device_list,
            'C5': ope_list, 'C6': publish_time_type_list, 'C7': province_list, 'C8': city_list,
            'C9': view_time_type_list, 'C10': first_class_list, 'C11': second_class_list, 'C12': keyword_list}
    df_data = pd.DataFrame(data)
    print(user_queshi)
    print(news_queshi)
    df_data.to_csv('../CTR2021/test_data.csv')
    print("write data success!")

def convert_chinese(st):
    return  st.encode('unicode_escape').decode('ascii')
def process_train():
    df = pd.read_csv('../CTR2021/train_info.txt')
    user_info_dict = {}
    doc_info_dict = {}
    with open('../CTR2021/user_info.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            flag = False
            s_split_list = line.split('\t')
            q = 0
            if(len(s_split_list) < 7):
                flag = True
            for l in s_split_list:
                if l == "":
                    flag = True
                    break
                q = q + 1
            if flag:
                line = f.readline()
                continue
            user_id = s_split_list[0]
            age_str = s_split_list[5]
            sex_str = s_split_list[6]
            age_list = age_str.split(',')
            max_pro = "0.000000"
            age_index = -1
            for i in range(4):
                if age_list[i][-8:] > max_pro:
                    max_pro = age_list[i][-8:]
                    age_index = i
            sex_list = sex_str.split(',')
            if len(sex_list) == 1:
                if sex_list[0][0] == 'm':
                    if sex_list[0][-9:-1] < '0.500000':
                        sex = 0
                    else:
                        sex = 1
                else:
                    if sex_list[0][-9:-1] < '0.500000':
                        sex = 1
                    else:
                        sex = 0
            else:
                if sex_list[0][-8:] > sex_list[1][-9:-1]:
                    sex = 0
                elif sex_list[0][-8:] < sex_list[1][-9:-1]:
                    sex = 1
                else:
                    sex = np.random.randint(2)
            user_info_dict[user_id] = {'device': s_split_list[1], 'os': s_split_list[2], 'provice': s_split_list[3],
                                       'city': s_split_list[4], 'age': age_index, 'sex': sex}
            line = f.readline()
    with open('../CTR2021/news_info.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            flag = False
            s_split_list = line.split('\t')
            if len(s_split_list) < 7:
                flag = True
            for l in s_split_list:
                if l == "":
                    flag = True
                    break
            if flag:
                line = f.readline()
                continue
            doc_id = s_split_list[0]
            doc_title = s_split_list[1]
            publish_time = s_split_list[2]
            publish_time = int(publish_time[:-2]) // 1000
            local_publish_time = time.localtime(publish_time)
            dt = time.strftime("%Y-%m-%d %H:%M:%S", local_publish_time)
            timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
            publish_hour = timeArray.tm_hour
            pubulish_time_type = publish_hour // 6
            first_class = s_split_list[4]
            second_class = s_split_list[5]
            keyword = s_split_list[6]
            doc_info_dict[doc_id] = {'picture_nums': s_split_list[3], 'publish_time_type': pubulish_time_type}
            line = f.readline()

    content_list = df.values
    user_id_list = []
    news_id_list = []
    time_list = []
    internet_env_list = []
    f5_times_list = []
    pos_list = []
    is_click_list = []
    duration_list = []
    age_list = []
    sex_list = []
    device_list = []
    ope_list = []
    picture_nums_list = []
    publish_time_type_list = []
    cou = 1
    for content in content_list:
        s = content[0]
        if cou % 5000000 == 0:
            print(cou)
        cou = cou + 1
        s_split_list = s.split('\t')
        user_id = s_split_list[0]
        doc_id = s_split_list[1]
        if user_id not in user_info_dict.keys():
            continue
        user_id_list.append(user_id)
        news_id_list.append(s_split_list[1])
        time_list.append(s_split_list[2])
        internet_env_list.append(s_split_list[3])
        f5_times_list.append(int(s_split_list[4]))
        pos_list.append(s_split_list[5])
        is_click_list.append(s_split_list[6])
        duration_list.append(int(s_split_list[7]))
        age_list.append(user_info_dict[user_id]['age'])
        sex_list.append(user_info_dict[user_id]['sex'])
        device_list.append(user_info_dict[user_id]['device'])
        ope_list.append(user_info_dict[user_id]['os'])
    data = {'label': is_click_list, 'I1': f5_times_list, 'I2': picture_nums_list, 'C1': user_id_list, 'C2': news_id_list, 'C3': internet_env_list, 'C4': age_list, 'C5': sex_list, 'C6': device_list, 'C7': ope_list, 'C8': publish_time_type_list}
    df_data = pd.DataFrame(data)
    df_data.to_csv('../CTR2021/train_data.csv')
    print("write data success!")

def process_test():
    user_info_dict = {}
    doc_info_dict = {}
    with open('../CTR2021/user_info.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            flag = False
            s_split_list = line.split('\t')
            q = 0
            if(len(s_split_list) < 7):
                flag = True
            if flag:
                line = f.readline()
                continue
            user_id = s_split_list[0]
            age_str = s_split_list[5]
            sex_str = s_split_list[6]
            age_list = age_str.split(',')
            max_pro = "0.000000"
            age_index = -1
            if age_str == '':
                age_index = np.random.randint(4)
            else:
                for i in range(4):
                    if age_list[i][-8:] > max_pro:
                        max_pro = age_list[i][-8:]
                        age_index = i
            sex_list = sex_str.split(',')
            if len(sex_list) == 1:
                if sex_list[0][0] == 'm':
                    if sex_list[0][-9:-1] < '0.500000':
                        sex = 0
                    else:
                        sex = 1
                else:
                    if sex_list[0][-9:-1] < '0.500000':
                        sex = 1
                    else:
                        sex = 0
            else:
                if sex_list[0][-8:] > sex_list[1][-9:-1]:
                    sex = 0
                elif sex_list[0][-8:] < sex_list[1][-9:-1]:
                    sex = 1
                else:
                    sex = np.random.randint(2)
            user_info_dict[user_id] = {'device': s_split_list[1], 'os': s_split_list[2], 'provice': s_split_list[3],
                                       'city': s_split_list[4], 'age': age_index, 'sex': sex}
            line = f.readline()

    with open('../CTR2021/news_info.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            flag = False
            s_split_list = line.split('\t')
            if len(s_split_list) < 7:
                flag = True
            for l in s_split_list:
                if l == "":
                    flag = True
                    break
            if flag:
                line = f.readline()
                continue
            doc_id = s_split_list[0]
            doc_title = s_split_list[1]
            publish_time = s_split_list[2]
            publish_time = int(publish_time[:-2]) // 1000
            local_publish_time = time.localtime(publish_time)
            dt = time.strftime("%Y-%m-%d %H:%M:%S", local_publish_time)
            timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
            publish_hour = timeArray.tm_hour
            pubulish_time_type = publish_hour // 6
            first_class = s_split_list[4]
            second_class = s_split_list[5]
            keyword = s_split_list[6]
            doc_info_dict[doc_id] = {'picture_nums': s_split_list[3], 'publish_time_type': pubulish_time_type}
            line = f.readline()
    user_id_list = []
    news_id_list = []
    time_list = []
    internet_env_list = []
    f5_times_list = []
    pos_list = []
    duration_list = []
    age_list = []
    sex_list = []
    device_list = []
    ope_list = []
    label_list = []
    picture_nums_list = []
    publish_time_type_list = []
    with open('../CTR2021/test_info.txt', encoding='utf-8') as f1:
        cou = 0
        s = f1.readline()
        while s:
            s = s[:-1]
            cou = cou + 1
            s_split_list = s.split('\t')
            user_id = s_split_list[1]
            if doc_id not in doc_info_dict.keys():
                s = f1.readline()
                continue
            label_list.append(0)
            user_id_list.append(user_id)
            news_id_list.append(s_split_list[2])
            time_list.append(s_split_list[3])
            internet_env_list.append(s_split_list[4])
            f5_times_list.append(int(s_split_list[5]))
            age_list.append(user_info_dict[user_id]['age'])
            sex_list.append(user_info_dict[user_id]['sex'])
            device_list.append(user_info_dict[user_id]['device'])
            ope_list.append(user_info_dict[user_id]['os'])
            picture_nums_list.append(doc_info_dict[doc_id]["picture_nums"])
            publish_time_type_list.append(doc_info_dict[doc_id]["publish_time_type"])
            s = f1.readline()
    data = {'label': label_list, 'I1': f5_times_list, 'I2': picture_nums_list, 'C1': user_id_list, 'C2': news_id_list, 'C3': internet_env_list, 'C4': age_list, 'C5': sex_list, 'C6': device_list, 'C7': ope_list, 'C8': publish_time_type_list}
    df_data = pd.DataFrame(data)
    df_data.to_csv('../CTR2021/test_data.csv')

if __name__ == '__main__':
    # pre_process_test()
    pre_process_train()
    # convert_chinese_english()
    # pre_process_train()