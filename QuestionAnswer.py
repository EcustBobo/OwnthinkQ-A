import os
import re
import random
from collections import defaultdict
from KB import Query
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# from b4k.bert4keras.questionTypeClass import QuestionTypeClass
from simbert.simbert_base import generateSimSentence
import time
from b4k.bert4keras.entityRecognize import NER
import requests
# from queue import Queue
try:
    import simplejson as json
except:
    import json
    
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dir_path = os.getcwd()

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.6
# config.gpu_options.allow_growth = True
# session=tf.Session(config=config)
# KTF.set_session(session)

# print(gen_sim_value('姚明','姚沁蕾'))
class AnswerByOwnthink(object):
    def __init__(self,):
        self.ques = ''
        self.ans_list = []
        self.ner = NER()
        self.query = Query()
        self.sim = generateSimSentence()
        # self.quesClass = QuestionTypeClass()
        self.prop_dict = dict()         # 存放识别得到的属性名
        self.searchPath = 'none->none'  # 程序最终查询路径
        self.attr_dict = defaultdict(list)  # 用以处理爸爸、妈妈、父亲相似度计算短板问题，存放映射
        self.stopFile = dir_path + '/src/qa/data/stopwords.txt'    # 加载停用词文件
        self.attrFile = dir_path + '/src/qa/data/attr_mapping.txt'    # 映射文件
        self.fuzzyFile = dir_path + '/src/qa/data/fuzzy.txt'
        self.stopWords = [line.strip() for line in open(self.stopFile,'r',encoding='utf-8').readlines()]
        self.fuzzy_dict = dict() 
        self.merge = MergeProperty()

    def searchQustion(self,question:str):
        entity_dict,self.prop_dict = self.getEntProp2dict(question)     # 获取实体和属性识别结果，根据SENT、OENT排序
        # question_type = self.quesClass.question_type_predict(question)
        entity_len = len(entity_dict)
        prop_len = len(self.prop_dict)
        self.ques = self.removeStopWordsList(question)      # 去除停用词    
        if entity_len == 0:                 # 实体不存在的情况
            res = '未识别到实体，请重新输入您的问题'
            return res,'none->none'
        elif entity_len == 1:               # 单个实体，单跳和多跳
            if prop_len == 1:
                entity = list(entity_dict.keys())[0]
                prop =  list(self.prop_dict.keys())[0]
                if list(entity_dict.values())[0] == 'SENT':            # sp_o查询
                    start_time = time.time()
                    res,searchPath,sim = self.SPO(entity,prop)
                    end_time = time.time()
                    print('spo查询耗时：%.2f'%(end_time-start_time))
                    if res != 'none':                                   # sp_o直接查询得到结果，直接返回
                        return res,searchPath
                    else:                                               # sp_o得不到结果，则使用相似度匹配法
                        start_time = time.time()
                        res,searchPath,sim = self.O_By_SP(entity)                   
                        end_time = time.time()
                        print('spo相似度匹配查询耗时：%.2f'%(end_time-start_time))
                        return res,searchPath
                else:                                                               # po_s查询
                    res,searchPath,sim = self.POS(entity,prop)                  
                    if res != 'none':                                   #PO_S直接查询，若有结果，直接返回
                        return res,searchPath
                    else:                                               #PO_S无结果，使用相似度计算法进行
                        res,searchPath,sim = self.S_By_PO(entity)
                        return res,searchPath
            
            elif prop_len == 2:                                     # spp_o查询
                if int(list(self.prop_dict.values())[0]) < int(list(self.prop_dict.values())[1]):   # 判断两个属性的前后顺序
                    prop_01 = list(self.prop_dict.keys())[0]
                    prop_02 = list(self.prop_dict.keys())[1]
                else:
                    prop_01 = list(self.prop_dict.keys())[1]
                    prop_02 = list(self.prop_dict.keys())[0]
                entity = list(entity_dict.keys())[0]
                res,searchPath,sim = self.SPPO(entity,prop_01,prop_02)
                return res,searchPath

            else:                                               # 属性为0或者属性超过两个的情况
                entity = list(entity_dict.keys())[0]
                if entity_dict[entity] == 'SENT':      # 单实体前向查询
                    res_01,searchPath_01,sim_01 = self.O_By_SP(entity)    # 单跳 返回结果 查询路径和相似度
                    if float(sim_01) >=0.99 :                      # 单条关系查询路径相似度高达0.99，直接返回
                        return res_01,searchPath_01
                    ans_list = [res_01,searchPath_01,sim_01]
                    self.ans_list = ans_list
                    res_02,searchPath_02,sim_02 = self.O_By_SPP(entity)   # 多跳 返回结果 查询路径和相似度
                    if float(sim_01) > float(sim_02) or float(sim_02) <= 0.7:
                        res = res_01
                        searchPath = searchPath_01
                    else:
                        res = res_02
                        searchPath = searchPath_02
                    return res,searchPath                  
                elif entity_dict[entity] == 'OENT':   # 单实体，反向查询，返回结果 查询路径和相似度
                    res,searchPath,sim = self.S_By_PO(entity)
                    return res,searchPath
                else:
                    res = '输入的问句中实体信息不全，请根据实体识别结果输入完整实体信息'
                    return res,self.searchPath
        elif entity_len == 2:
            res,searchPath,sim = self.P_By_SO(entity_dict)                 # 关系查询，返回结果和查询路径
            return res,searchPath

        else:
            res = '暂不支持三个实体以上的问答，请重新输入问句'
            return res,self.searchPath

    ## 给定实体和属性，查询属性值
    def SPO(self,entity,prop):
        print('搜索类型：SPO')
        res = list((self.query.SP_O(entity,prop)).values())      # 直接通过实体名+属性值查询
        ans = []
        if res:
            # print(res)
            ans.append(res[0])
            searchPath = entity + '->' + prop 
            sim_val = 1.00
            return ans[0],searchPath,sim_val
        else:
            data_list_dict = self.query.Q_Z_ByEntity(entity)     # 获取与实体相关的歧义关系列表[{ }]
            if data_list_dict:                      # 歧义关系存在的情况
                ans = ['none']                      # 要返回的列表结果
                searchPath = 'none->none'
                sim_val = 0                        # 相似度的值
                data_info = self.merge.mergeProperty(data_list_dict)
                entity_data_info_list = list(data_info.keys())
                for ent in entity_data_info_list:
                    if prop in data_info[ent]:
                        ans[0]=','.join(data_info[ent][prop])
                        searchPath = ent + '->' + prop
                        sim_val = 1.00
                        return ans[0],searchPath,sim_val
                return ans[0],searchPath,sim_val
            else:
                return 'none','none->none',0.00            
        
    # 给定P和O,查询S
    def POS(self,entity,prop):
        print('搜索类型：POS')
        res = list((self.query.OP_S(entity,prop)).values())  # 根据属性和属性值，neo4j直接查询对应的实体
        ans = []
        if res:
            temp_str = ','.join(res)
            ans.append(temp_str)
            searchPath = prop+ '->' + entity
            sim_val = 1.00
            return ans[0],searchPath,sim_val
        else:
            return 'none','none->none',0.00

    # 给定S、P、P，查询O,多跳
    def SPPO(self,entity,prop_01,prop_02):
        print('搜索类型：SPPO')
        ans_1,searchPath_01,sim_01 = self.SPO(entity,prop_01)       # 单步查询 
        if ans_1 != 'none':
            ans_2,searchPath_02,sim_02 = self.SPO(ans_1,prop_02)    # 二次查询
            if ans_2 != 'none':
                searchPath = searchPath_01+'->'+searchPath_02
                return ans_2,searchPath,sim_02
            else:        
                ans_list = [ans_1,searchPath_01,sim_01]
                self.ans_list = ans_list                            # 存放SPP_O的结果，做比较用
                ans_2,searchPath_02,sim_02 = self.O_By_SPP(entity)
                if ans_2 != 'none':
                    return ans_2,searchPath_02,sim_02
                else:
                    return 'none','none->none',0.00
        else:
            ans,searchPath,sim = self.O_By_SP(entity)
            ans_list = [ans,searchPath,sim]
            self.ans_list = ans_list
            if ans !='none':
               ans_2,searchPath_02,sim_02 = self.O_By_SPP(entity)  
               if ans_2 != 'none':
                   return ans_2,searchPath_02,sim_02
               else:
                   return 'none','none->none',0.00
            else:
                return 'none','none->none',0.00


    # 通过相似度计算sp_o类型问题的查询
    def O_By_SP(self,entity):
        print('搜索类型：O_SP')
        data_list_dict = self.query.Q_Z_ByEntity(entity)     # 获取与实体相关的歧义关系列表[{ }]
        self.attr_dict = self.getSimilarityAttr()
        searchPath = 'none->none'
        sim_val = 0.00
        if data_list_dict:                      # 歧义关系存在的情况
            ans = ['none']                      # 要返回的列表结果
            sim_val = 0.00                         # 相似度的值
            data_info = self.merge.mergeProperty(data_list_dict)
            entity_data_info_list = list(data_info.keys())
            print(entity_data_info_list)
            for ent in entity_data_info_list:
                prop_list = list(data_info[ent].keys())
                ques = [self.ques]
                for r_name in prop_list[1:]:
                    que = entity + r_name
                    ques.append(que)
                sim_val_list = self.sim.gen_all_sim_value(ques)     # 批量计算
                temp_sim_val = max(sim_val_list)
                comp_sim_val ='%.4f'%(float(temp_sim_val))
                if float(comp_sim_val) > float(sim_val):
                    r_name_index = sim_val_list.index(temp_sim_val) + 1
                    r_name = prop_list[r_name_index]
                    ans[0] = ','.join(data_info[ent][r_name])
                    searchPath = ent + '->' + r_name
                    sim_val = comp_sim_val                    
                    print('相似问题顺序：%s,相似度%s'%(ques[r_name_index],sim_val))
            return ans[0],searchPath,sim_val

        else:                                    # 歧义关系不存在
            ans = ['none']
            sim_val = 0.00                                  
            # entity_rela_dict = defaultdict(list)    # 构造字典列表
            S_P_list = self.query.S_P(entity)
            rela_info = self.merge.merge_S_P_Property(S_P_list)
            if rela_info:
                ques = [self.ques]
                prop_list = list(rela_info.keys())
                for s_prop in prop_list:
                    que = entity + s_prop
                    ques.append(que)
                sim_val_list = self.sim.gen_all_sim_value(ques)
                temp_sim_val = max(sim_val_list)
                comp_sim_val ='%.4f'%(float(temp_sim_val))
                if float(comp_sim_val) > float(sim_val):
                    r_name_index = sim_val_list.index(temp_sim_val)
                    r_name = prop_list[r_name_index]
                    ans[0] = ','.join(rela_info[r_name])
                    searchPath = entity + '->' + r_name
                    sim_val = comp_sim_val                    
                    print('相似问题顺序：%s,相似度%s'%(ques[r_name_index+1],sim_val))                   
                return ans[0] , searchPath,sim_val                       
                
            else:
                return '实体无关系词，查询结果：无','none->none',0.00
    # 通过相似度计算spp_o类型问题的查询
    def O_By_SPP(self,entity):
        print('搜索类型：O_SPP')
        ans = ['none']
        searchPath = 'none'
        # if float(self.ans_list[-1]) != 0.00 and len(self.ans_list[0])<5:
        if float(self.ans_list[-1]) != 0.00:
            S_P_list = self.query.S_P(self.ans_list[0])
            rela_info = self.merge.merge_S_P_Property(S_P_list)
            search_path_01 = self.ans_list[1].replace('->','')
            search_path_01 = re.sub(u"\\[.*?\\]",'',search_path_01) 
            sim_val = 0
            ques = [self.ques]
            prop_list = list(rela_info.keys())
            for sp_prop in prop_list:
                que = search_path_01 + sp_prop
                ques.append(que)
            sim_val_list = self.sim.gen_all_sim_value(ques)
            if not sim_val_list:
                return 'none','none->none',0.00
            temp_sim_val = max(sim_val_list)
            comp_sim_val ='%.4f'%(float(temp_sim_val))
            if float(comp_sim_val) > float(sim_val):
                r_name_index = sim_val_list.index(temp_sim_val)
                r_name = prop_list[r_name_index]
                ans[0] = ','.join(rela_info[r_name])
                searchPath = self.ans_list[1] + '->' + r_name
                sim_val = comp_sim_val                    
                print('相似问题顺序：%s,相似度%s'%(ques[r_name_index+1],sim_val))                 
            return ans[0],searchPath,sim_val
        else:
            return 'none','none->none', 0.00

    # 查询实体的关系
    def P_By_SO(self,entity_dict:dict):
        print('搜索类型：P_SO')
        entity_01,entity_02 = list(entity_dict.keys())[0],list(entity_dict.keys())[0]
        for key,index in entity_dict.items():
            if index =='SENT':
                entity_01 = key
            elif index == 'OENT':
                entity_02 = key
        sim_val = 0.00
        data_list_dict = self.query.Q_Z_ByEntity(entity_01)
        res = ''
        # data_info = self.merge.mergeProperty(data_list_dict)
        if data_list_dict:
            for dict_data in data_list_dict:
                # m_name = dict_data['m.name']    # 实体名称
                r2_name = dict_data['r2.name']  # 关系名称
                n2_name = dict_data['n2.name']  # 对应实体属性值
                if n2_name == entity_02:
                    res = "{}:{}:{}".format(entity_01,r2_name,entity_02)
                    searchPath = dict_data['n.name'] + '-[r]-' +entity_02
                    sim_val = 1.00
                    break
        if res !='':
            return res, searchPath,sim_val 
        else:
            res = self.query.P_By_SO(entity_01,entity_02) 
            if not res:
                res,searchPath,sim_val = self.O_By_SP(entity_02)
            else:
                searchPath = res
                sim_val = 1.00
        return res, searchPath,sim_val

    # 相似度计算，根据关系和实体查询实体
    def S_By_PO(self,entity):
        print('搜索类型：S_PO')
        data_dict_list = self.query.OP_S_ByEntity(entity)    # 先查找与实体相关的信息
        sim_val = 0.00
        ans = ['none']
        if data_dict_list:
            for data_dict in data_dict_list:
                r_name = data_dict['r.name']
                que = r_name + entity
                temp_sim_val = self.sim.gen_sim_value(self.ques,que)
                if float(temp_sim_val) > float(sim_val):
                    sim_val = temp_sim_val
                    ans[0] = data_dict['m.name']
                    searchPath = r_name + '->' + entity
                    print('相似问题：%s，相似度:%s,查询路径：%s'%(que,sim_val,searchPath))
                if float(temp_sim_val) == float(sim_val) and data_dict['m.name'] not in ans[0]:
                    ans[0] = ans[0] + ',' + data_dict['m.name']
                    searchPath = r_name + '->' + entity
            return ans[0],searchPath,sim_val
        else:
            ans[0] = '未能查到实体%s的相关信息，请重新输入'%(entity)
            return ans[0],'none->none', 0.00

    # 返回实体和属性字典，实体以SENT,OENT作为键，实体名作为，属性以属性值作为键
    def getEntProp2dict(self,question):
        ent = dict()
        prop = dict()
        # ner = NamedEntityRecognizer()
        start_time = time.time()
        ans = self.ner.predict(question)
        # ans = predict(question)
        if not ans:
            ans = ans = self.ner.predict(question + '?')
            # ans = predict(question + '？')
        end_time = time.time()
        print('实体识别耗时：%s'%(end_time-start_time))
        print('实体属性识别结果：%s'%(ans))
        if ans:
            for val in ans:
                if val[1] == 'SENT' or val[1] == 'OENT':
                    ent[val[0]]=val[1]
                elif val[1] == 'PROP':
                    pos = question.find(val[0])
                    prop[val[0]] = pos
        return ent,prop


    # 加载停用词词典
    def removeStopWordsList(self,question:str):
        for i in  range(len(self.stopWords)):
            if self.stopWords[i] in question:
                question = question.replace(self.stopWords[i],'')
        return question

    # 属性映射计算相似度
    def getSimilarityAttr(self,):
        with open(self.attrFile,'r',encoding='utf-8') as f:
            for line in f.readlines():
                line = line.replace('\n','')
                line_list = line.split(' ')
                for i in line_list:
                    for j in line_list:
                        if i != j:
                            self.attr_dict[i].append(j)
            # print(self.attr_dict)
            return self.attr_dict

    # elasticsearch模糊查询
    def getSimEntFromElastic(self,question):
        es_url = os.getenv('NEO4J_BASE')
        if es_url =='http://keylab.jios.org:7474':
            es_url = "http://localhost:9200/"
        else:
            es_url = "http://192.168.0.55:9200/"
        ent,prop = self.getEntProp2dict(question)
        ent_list = list(ent.keys())
        ents = ent_list[0]
        self.loadFuzzyDict(self.fuzzyFile)
        if ents in self.fuzzy_dict:
            entity = self.fuzzy_dict[ents]
            question = question.replace(ents,entity)
            return entity,question
        self.query = json.dumps({"self.query": { "bool":{"must":[{"match":{"Entity":ents}}]}},"from":0,"size":10},ensure_ascii=False)
        self.query = self.query.encode('utf-8')
        url_01 = es_url + 'node_实体' + '/' + 'Entity' + '/_search'
        start_time = time.time()
        response = requests.get(url_01,headers={"Content-Type":"application/json"}, data = self.query)
        end_time = time.time()
        print('ES查询耗时：%s'%(end_time-start_time))
        res = json.loads(response.content)
        if res['hits']['hits']:
            ans = res['hits']['hits'][0]['_source']['Entity']
            question = question.replace(ents,ans)
            return ans,question
        else:
            return 'none',question
    
    def loadFuzzyDict(self,file):
        with open(file,'r',encoding='utf-8') as f:
            for line in f.readlines():
                line = line.replace('\n','')
                line_list = line.split(':')
                self.fuzzy_dict[line_list[0]]=line_list[1]


class MergeProperty(object):
    def __init__(self,):
        self.dict = {}
    def mergeProperty(self,dataDict:list):
        res_dict = dict()
        if not dataDict:
            return res_dict
        for qiyiDict in dataDict:
            entity = qiyiDict['n.name']
            weight = qiyiDict['weight']
            r2_name = qiyiDict['r2.name']
            n2_name = qiyiDict['n2.name']
            temp_dict =dict()
            temp_list = []
            if entity not in res_dict:
                temp_dict['weight'] = [weight]
                temp_dict[r2_name] = [n2_name]
                res_dict[entity]=temp_dict
            else:
                if r2_name in res_dict[entity]:
                    temp_list = res_dict[entity][r2_name]
                    if n2_name not in temp_list:
                        temp_list.append(n2_name)
                        res_dict[entity][r2_name] = temp_list
                else:
                    temp_list.append(n2_name)
                    res_dict[entity][r2_name] = temp_list

        return res_dict
    def merge_S_P_Property(self,spDict:list):
        res_dict = dict()
        if not spDict:
            return res_dict
        for p_dict in spDict:
            r_name = p_dict['r.name']
            m_name = p_dict['m.name']
            if r_name not in res_dict:
                res_dict[r_name]=[m_name]
            else:
                res_dict[r_name] = [m_name] + res_dict[r_name] 
        return res_dict
'''     
    # 推荐问题
    def generateSimQestion(self,searchPath:str):
        ent_attr_list = searchPath.split('->')
        ent_attr_list[0] = re.sub(u"\\[.*?\\]",'',ent_attr_list[0]) 
        # print(ent_attr_list)
        sim_str = self.sim.wordVec2word(ent_attr_list[1])
        # print('相似属性：%s'%(sim_str))
        ques_str = ent_attr_list[0] + '的' + sim_str + '是？'
        sim_ques_list = random.sample(self.genSim.gen_synonyms(ques_str),1)
        return sim_ques_list  
'''


if __name__ == "__main__":
    test = AnswerByOwnthink()
    while True: 
        try:
            ans,searchPath = 'none','none'
            print('请输入您要查询的问题')
            question = input('question:')
            if question == 'break':
                break
            start_time = time.time()
            ans,searchPath = test.searchQustion(question)
            end_time = time.time()
            print('查询路径：%s'%(searchPath))
            print('问题“%s”的答案是：“%s”'%(question,ans))
            # print('查询耗时：%s'%(end_time-start_time))
            times = int((end_time-start_time)*1000)
            print(times)
            all_use_time = int(float('%.3f'%(end_time-start_time))*1000)
            print('查询耗时：%s'%(str(all_use_time)+'ms'))
            if ans == '实体无关系词，查询结果：无':
                res,que = test.getSimEntFromElastic(question)
                if res !='none':
                    print('您输入的问句中实体不存在，您是否想查询：%s'%(que))
        except Exception as e :
            print(e)
            print('输入异常，请重新输入')
