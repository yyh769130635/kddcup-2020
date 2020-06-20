# -*- coding: utf-8 -*-
import pandas as pd  
from tqdm import tqdm  
from collections import defaultdict  
import math  
import time
import numpy as np

def get_sim_item(df_, user_col, item_col, use_iif=False): 
    
    df = df_.copy()
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))
    
    user_time_ = df.groupby(user_col)['time'].agg(list).reset_index() # 引入时间因素
    user_time_dict = dict(zip(user_time_[user_col], user_time_['time']))
    
    sim_item = {}  
    item_cnt = defaultdict(int)  # 商品被点击次数
    for user, items in tqdm(user_item_dict.items()):  
        for loc1, item in enumerate(items):  
            item_cnt[item] += 1  
            sim_item.setdefault(item, {})  
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:  
                    continue  
                t1 = user_time_dict[user][loc1] # 点击时间提取
                t2 = user_time_dict[user][loc2]
                sim_item[item].setdefault(relate_item, 0)  
                if not use_iif:  
                    if loc1-loc2>0:
                        sim_item[item][relate_item] += 1 * 0.7 * (0.8**(loc1-loc2-1)) * (1 - (t1 - t2) * 10000) / math.log(1 + len(items)) # 逆向
                    else:
                        sim_item[item][relate_item] += 1 * 1.0 * (0.8**(loc2-loc1-1)) * (1 - (t2 - t1) * 10000) / math.log(1 + len(items)) # 正向
                else:  
                    sim_item[item][relate_item] += 1 / math.log(1 + len(items))  

    sim_item_corr = sim_item.copy() # 引入AB的各种被点击次数  
    for i, related_items in tqdm(sim_item.items()):  
        for j, cij in related_items.items():  
            sim_item_corr[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)  
  
    return sim_item_corr, user_item_dict  
  

def recommend(sim_item_corr, user_item_dict, user_id, top_k, item_num,query_time_index):
    '''
    input:item_sim_list, user_item, uid, 500, 50
    # 用户历史序列中的所有商品均有关联商品,整合这些关联商品,进行相似性排序
    '''  
    rank = {}  
    interacted_items = user_item_dict[user_id] 
    interacted_items = interacted_items[::-1]
    str_time=int(query_time_index)
    
    for loc, i in enumerate(interacted_items):  
        for j, wij in sorted(sim_item_corr[i].items(), reverse=True):#[0:top_k]:  
            if j not in interacted_items: 

                rank.setdefault(j, 0)  
                #temp=huancun[huancun['item_id']==j]
                if j in tiaozheng_list:
                    # tag=tiaozheng_list.index(j)
                    # tag=tiaozheng_list[j]
                    # temp= huancun.loc[tag]
                    # temp=huancun[huancun['item_id']==j]
                #if len(temp)>0:
                    # rank[j] += wij * (0.7**loc) *4.5* temp[str_time].values[0]
                    temp=tiaozheng_list[j]
                    rank[j] += wij * (0.7 ** loc) * np.log2(1.4+7 * temp[str_time-1])
                #    xiuzheng=huancun[huancun['item_id']==j][str(query_time_index)].values[0]
                #    rank[j] += wij * (0.7**loc) * xiuzheng
                else:
                    rank[j] += wij * (0.7**loc) 
    #for item_id,score in rank.items():
        #if (item_count[item_count['item_id']==item_id]['user_id'].values[0] >50) :
    #    if len(huancun[huancun['item_id']==item_id])>0:
    #        rank[item_id] *=huancun[huancun['item_id']==item_id][str(query_time_index)].values[0]
              
    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]  


# fill user to 50 items
def get_predict(df, pred_col, top_fill):  
    top_fill = [int(t) for t in top_fill.split(',')]  
    scores = [-1 * i for i in range(1, len(top_fill) + 1)]  
    ids = list(df['user_id'].unique())  
    fill_df = pd.DataFrame(ids * len(top_fill), columns=['user_id'])  
    fill_df.sort_values('user_id', inplace=True)  
    fill_df['item_id'] = top_fill * len(ids)  
    fill_df[pred_col] = scores * len(ids)  
    df = df.append(fill_df)  
    df.sort_values(pred_col, ascending=False, inplace=True)  
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')  
    df['rank'] = df.groupby('user_id')[pred_col].rank(method='first', ascending=False)  
    df = df[df['rank'] <= 50]  
    df = df.groupby('user_id')['item_id'].apply(lambda x: ','.join([str(i) for i in x])).str.split(',', expand=True).reset_index()  
    return df  


def get_time_index(unix_time):
    return int((unix_time-0.98375572)/0.00005428)+1

# def dic(huancun2):
#     rank = {}
#     i = 0
#     for j in huancun2:
#         rank.setdefault(j, i)
#         i += 1
#     return rank


if __name__ == "__main__":
    start=time.time()
    now_phase =3
    train_path = './underexpose_train'
    test_path = './underexpose_test'

    huancun_path = './'
    huancun = pd.read_csv(huancun_path + '/huancun4.csv', header=None,
                          names=['item_id', '1', '2', '3', '4', '5', '6', '7'])

    # 这里可能要修改
    # tiaozheng_list = huancun['item_id'].values.tolist()
    # huancun2 = huancun['item_id']
    # tiaozheng_list=dic(huancun2)
    ##将huancun从dataframe保存为字典工快速查询
    tiaozheng_list = huancun.set_index('item_id').T.to_dict('list')

    # print(tiaozheng_list)


    recom_item = []
    whole_click = pd.DataFrame()
    click_train=[]
    click_test=[]
    qtime=[]
    all_test=[]

    for c in range(now_phase + 1):
        middle1=time.time()
        print('phase : ', c)
        temp_click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c),
                                       header=None,  names=['user_id', 'item_id', 'time'])
        temp_click_test = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(c,c),
                                      header=None,  names=['user_id', 'item_id', 'time'])
        temp_qtime = pd.read_csv(test_path+'/underexpose_test_click-{}/underexpose_test_qtime-{}.csv'.format(c,c),
                                 names=['user_id','query_time'])
        temp_all_click = temp_click_train.append(temp_click_test)
    
        #click_train=click_train.append(temp_click_test)
        #click_test=click_test.append(temp_all_click)
        #qtime=qtime.append(temp_qtime)
    
        whole_click = whole_click.append(temp_all_click)
        whole_click = whole_click.drop_duplicates(subset=['user_id','item_id','time'],keep='last')
        whole_click = whole_click.sort_values('time')

        print("---------get_sim_item-------------")
        item_sim_list, user_item = get_sim_item(whole_click, 'user_id', 'item_id', use_iif=False)

        item_count=whole_click.groupby('item_id').count().reset_index()

        #print("------------temp_click_test['user_id'].unique()------------")

        for i in tqdm(temp_click_test['user_id'].unique()):
            query_time=temp_qtime[temp_qtime['user_id']==i]['query_time'].values[0]
            query_time_index=get_time_index(query_time)
            if query_time_index>7:
                query_time_index=7
            if query_time_index<1:
                query_time_index=1
            #time_start=query_time-0.00005
            #time_end=query_time+0.00005
            #temp_whole_click=whole_click[whole_click['time']>=time_start]
            #temp_whole_click=temp_whole_click[temp_whole_click['time']<time_end]
            # start=time.time()
            rank_item = recommend(item_sim_list, user_item, i, 50, 50,query_time_index)
            # end=time.time()
            # Time=end-start
            # print("user {} recommend time: {}s".format(i, Time))
            for j in rank_item:
                recom_item.append([i, j[0], j[1]])


        middle2 = time.time()
        print('phase :{} comsuming time is: {}s '.format(c, (middle2-middle1)))
                
    # find most popular items
    top50_click = whole_click['item_id'].value_counts().index[:50].values
    top50_click = ','.join([str(i) for i in top50_click])

    recom_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim'])
    result = get_predict(recom_df, 'sim', top50_click)
    result.to_csv('baseline-04-30.csv', index=False, header=None)
    end = time.time()
    print("total time is %f s"%(end-start))
    print("finish !")