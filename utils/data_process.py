import pandas as pd
import numpy as np
import tensorflow as tf
def process_train():
    df = pd.read_csv('../CTR2021/train_info.txt')
    user_info_dict = {}
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
    cou = 1
    for content in content_list:
        s = content[0]
        if cou % 5000000 == 0:
            print(cou)
        cou = cou + 1
        s_split_list = s.split('\t')
        user_id = s_split_list[0]
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
    data = {'label': is_click_list, 'I1': f5_times_list, 'C1': user_id_list, 'C2': news_id_list, 'C3': internet_env_list, 'C4': age_list, 'C5': sex_list, 'C6': device_list, 'C7': ope_list}
    df_data = pd.DataFrame(data)
    df_data.to_csv('../CTR2021/train_data.csv')
    print("write data success!")

def process_test():
    user_info_dict = {}
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
    with open('../CTR2021/test_info.txt', encoding='utf-8') as f1:
        cou = 0
        s = f1.readline()
        while s:
            s = s[:-1]
            cou = cou + 1
            s_split_list = s.split('\t')
            user_id = s_split_list[1]
            if user_id not in user_info_dict.keys():
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
            s = f1.readline()
    data = {'label': label_list, 'I1': f5_times_list, 'C1': user_id_list, 'C2': news_id_list, 'C3': internet_env_list, 'C4': age_list, 'C5': sex_list,  'C7': ope_list}
    df_data = pd.DataFrame(data)
    df_data.to_csv('../CTR2021/test_data.csv')

if __name__ == '__main__':
    process_train()