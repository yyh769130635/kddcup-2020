# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 21:27:21 2020

@author: yyh76
"""

#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
  
import pandas as pd  
from tqdm import tqdm  
from collections import defaultdict  
import math  
  
  
def get_sim_item(df, user_col, item_col, use_iif):  
    #把原先的click表 同用户id进行group,创建一个集合，保存每个用户买过的物品
    user_item_ = df.groupby(user_col)[item_col].agg(set).reset_index() 
    #转换成字典  这时候是一个用户 -- {商品集合} 的倒查表
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))  
    #这个list中记录了item之间的相似性
    sim_item = {}  
    #用来保存所有物品出现过的次数的
    item_cnt = defaultdict(int)  
    #这一段像是在计算item的同现性 用这个来近似item相似性
    #如果在整个click表中 两个item同现次数高 则 相关度/相似性高
    #从倒查表中按行取pair
    for user, items in tqdm(user_item_dict.items()):  
        #对item集合中的每一个item
        for i in items:  
            #在总的item dict中 这个item计数一次 统计item出现次数 也就是流行度
            item_cnt[i] += 1  
            #这变成了一个嵌套的list
            sim_item.setdefault(i, {})  
            for relate_item in items:  
                if i == relate_item:  
                    continue  
                sim_item[i].setdefault(relate_item, 0)  
                #是否使用use——iif 消除用户活跃的影响
                if not use_iif:  
                    sim_item[i][relate_item] += 1  
                else:  
                    sim_item[i][relate_item] += 1 / math.log(1 + len(items))  
    
    #消除商品流行度的影响
    sim_item_corr = sim_item.copy()  
    for i, related_items in tqdm(sim_item.items()):  
        for j, cij in related_items.items():  
            sim_item_corr[i][j] = cij/math.sqrt(item_cnt[i]*item_cnt[j])  
  
    return sim_item_corr, user_item_dict  
  
  
def recommend(sim_item_corr, user_item_dict, user_id, top_k, item_num):  
    rank = {} 
    #当前用户涉及到的items 
    interacted_items = user_item_dict[user_id] 
    #遍历该用户购买过的所有item 
    for i in interacted_items:  
        #对每一个item 从item相关表中找到最相似的前500个
        for j, wij in sorted(sim_item_corr[i].items(), reverse=True)[0:top_k]:  
            #j 代表候选item名称 wij代表相关度
            #去掉已经购买过的
            if j not in interacted_items:  
                rank.setdefault(j, 0)  
                rank[j] += wij  
    #从所有牵涉到的商品中 再找出前50个
    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]  
  
    
## fill user to 50 items  
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
  
  
if __name__ == "__main__": 
    #设定phase
    now_phase = 3  
    train_path = './underexpose_train'  
    test_path = './underexpose_test'  
    recom_item = []  
    
    whole_click = pd.DataFrame()  
    #最外层循环 不同phase
    for c in range(now_phase + 1):  
        print('phase:', c)  
        #获取文件
        click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
        click_test = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(c,c), header=None,  names=['user_id', 'item_id', 'time'])  
        #合并train 和 test
        all_click = click_train.append(click_test)  
        #合并0-3
        whole_click = whole_click.append(all_click)  
        #计算item相似度 user——item是用户点击商品的倒查表
        item_sim_list, user_item = get_sim_item(all_click, 'user_id', 'item_id', use_iif=True)  
        
        #对于test数据集里的每一个 user 遍历
        for i in tqdm(click_test['user_id'].unique()):  
            rank_item = recommend(item_sim_list, user_item, i, 500, 50)  
            for j in rank_item:  
                #增加三元组 'user_id', 'item_id', 'sim' 用户 商品 契合度
                recom_item.append([i, j[0], j[1]])  
    
    # find most popular items  最流行的items
    top50_click = whole_click['item_id'].value_counts().index[:50].values  
    top50_click = ','.join([str(i) for i in top50_click])  
  
    recom_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim'])  
    result = get_predict(recom_df, 'sim', top50_click)  
    result.to_csv('baseline.csv', index=False, header=None)