# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:23:53 2020

@author: yyh76
"""

from __future__ import division
from __future__ import print_function
import pandas as pd  
from tqdm import tqdm  
from collections import defaultdict  
import math  


import datetime
import json
import sys
import time
import numpy as np
  
def get_sim_item(df_, user_col, item_col, use_iif=False): 
    
    df = df_.copy()
    # 把原先的click表 同用户id进行group,创建一个list，保存每个用户买过的物品
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    # 转换成字典  这时候是一个用户 -- {商品集合} 的倒查表
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))
    # 把原先的click表 同用户id进行group,创建一个list，保存用户每次点击物品的时间
    user_time_ = df.groupby(user_col)['time'].agg(list).reset_index()
    #用户--{time}的倒查表
    user_time_dict = dict(zip(user_time_[user_col], user_time_['time']))
    ## 物品之间的相似度
    sim_item = {}  
    item_cnt = defaultdict(int)  # 商品被点击次数，默认值为0
    for user, items in tqdm(user_item_dict.items()):
        #loc1表示当前物品在用户点击过商品中出现的次序， item表示商品id
        for loc1, item in enumerate(items):  
            item_cnt[item] += 1  
            sim_item.setdefault(item, {})  
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:#和自己就跳过
                    continue  
                t1 = user_time_dict[user][loc1] # 点击item的时间
                t2 = user_time_dict[user][loc2] # 点击relate_item的时间
                sim_item[item].setdefault(relate_item, 0)  
                if not use_iif:  
                    if loc1-loc2>0:#比较点击的前后顺序，隔的远，相关性低，0.8可调，log消除用户活跃度影响
                        sim_item[item][relate_item] += 1 * 0.7 * (canshu2**(loc1-loc2-1)) * (1 - (t1 - t2) * 10000) / math.log(1 + len(items)) # 逆向
                    else:
                        sim_item[item][relate_item] += 1 * 1.0 * (canshu2**(loc2-loc1-1)) * (1 - (t2 - t1) * 10000) / math.log(1 + len(items)) # 正向
                else:  
                    sim_item[item][relate_item] += 1 / math.log(1 + len(items))  

    sim_item_corr = sim_item.copy() # 引入AB的各种被点击次数  
    for i, related_items in tqdm(sim_item.items()):  
        for j, cij in related_items.items():  
            sim_item_corr[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)  #这里要改一下，解决哈利波特，商品热门的影响
  #1、返回物品相似度的矩阵（这里是字典），2、返回用户-点击过的物品倒查表
    return sim_item_corr, user_item_dict  
  
def recommend(sim_item_corr, user_item_dict, user_id, top_k, item_num):  
    '''
    input:item_sim_list, user_item, uid, 500, 50
    # 用户历史序列中的所有商品均有关联商品,整合这些关联商品,进行相似性排序
    '''
    rank = {}
    #user_item_dict，用户 -- {点击过的商品}
    interacted_items = user_item_dict[user_id]
    #倒序，最新点击的在前面
    interacted_items = interacted_items[::-1]
    # 遍历该用户购买过的所有item
    for loc, i in enumerate(interacted_items):
        #对于商品i，做推荐，取前top_k个
        # print(sorted(sim_item_corr[i].items(), reverse=True)[0:top_k])
        #按商品编号排序,取前top_k个（似乎没什么用）
        for j, wij in sorted(sim_item_corr[i].items(), reverse=True)[0:top_k]:
            # j 代表候选item名称 wij代表相关度
            # 去掉已经购买过的
            if j not in interacted_items:  
                rank.setdefault(j, 0)  
                # rank[j] += wij * (0.7**loc)
                rank[j] += wij * (canshu1 ** loc)
    #按照相似度排序，取前item_num个，有可能不满
    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]

# fill user to 50 items  
def get_predict(df, pred_col, top_fill):
    #最热门的50个商品
    top_fill = [int(t) for t in top_fill.split(',')]

    scores = [-1 * i for i in range(1, len(top_fill) + 1)]  
    ids = list(df['user_id'].unique())  
    fill_df = pd.DataFrame(ids * len(top_fill), columns=['user_id'])  
    fill_df.sort_values('user_id', inplace=True)  
    fill_df['item_id'] = top_fill * len(ids)  
    fill_df[pred_col] = scores * len(ids)

    df = df.append(fill_df)

    #按照'sim'排序，用排序后的数据集代替原来的数据集
    df.sort_values(pred_col, ascending=False, inplace=True)
    ##但这里并没有加入时间因素。。。
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    #rank函数返回从小到大排序的下标
    df['rank'] = df.groupby('user_id')[pred_col].rank(method='first', ascending=False)  
    df = df[df['rank'] <= 50]  #取前50个
    df = df.groupby('user_id')['item_id'].apply(lambda x: ','.join([str(i) for i in x])).str.split(',', expand=True).reset_index()  
    return df  


now_phase = 4
train_path = './underexpose_train'
test_path = './fake_file'
global canshu1
global canshu2
count=0
for canshu1 in np.arange(0.6,0.82,0.02):
    for canshu2 in np.arange(0.7,0.92,0.02):
        recom_item = []
        print("**************************",count,"*********************************")
        count+=1
        whole_click = pd.DataFrame()
        for c in range(now_phase + 1):
            print('phase:', c)
            click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c),
                                      header=None, names=['user_id', 'item_id', 'time'])
            # click_test = pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(c),
            #                          header=None,  names=['user_id', 'item_id', 'time'])
            click_test = pd.read_csv(test_path + '/fake_test_click-{}.csv'.format(c),
                                     header=None, names=['user_id', 'item_id', 'time'])
            # 合并train 和 test
            all_click = click_train.append(click_test)
            # 合并0-3
            whole_click = whole_click.append(all_click)
            # 去重，保持最新的点击时间
            whole_click = whole_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
            # 按照时间排序
            whole_click = whole_click.sort_values('time')
            # 1、返回物品相似度的矩阵（这里是字典），2、返回用户-点击过的物品倒查表
            item_sim_list, user_item = get_sim_item(whole_click, 'user_id', 'item_id', use_iif=False)

            # unique 唯一值，对每个用户做分析
            for i in tqdm(click_test['user_id'].unique()):
                # recommend 50 items,排序过的,物品以及相似度
                rank_item = recommend(item_sim_list, user_item, i, 500, 50)
                for j in rank_item:
                    # 增加三元组 'user_id', 'item_id', 'sim' 用户 商品 契合度
                    recom_item.append([i, j[0], j[1]])
                    ###进入下一次迭代

        # find most popular items
        # 统计物品出现的个数，并选取前50个最热门的商品
        top50_click = whole_click['item_id'].value_counts().index[:50].values
        top50_click = ','.join([str(i) for i in top50_click])

        recom_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim'])
        # 没有满的补充
        result = get_predict(recom_df, 'sim', top50_click)
        result.to_csv('./result/baseline-{}-{}.csv'.format(canshu1,canshu2), index=False, header=None)