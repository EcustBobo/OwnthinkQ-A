import os
import re
import random
from collections import defaultdict
from KB import Query
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from b4k.bert4keras.questionTypeClass import QuestionTypeClass
from simbert.simbert_base import generateSimSentence
import time
from b4k.bert4keras.entityRecognize import NER
import requests
try:
    import simplejson as json
except:
    import json


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dir_path = os.getcwd()

class AnswerByOwnthink(object):
    def __init__(self,):
        self.ques = ''
        self.ans_list = []  # 存放一次查询的结果
        self.ner = NER()                        # 实体识别
        self.query = Query()                    # neo4j查询
        self.sim = generateSimSentence()        # 相似度计算
        self.quesClass = QuestionTypeClass()    # 分类模型，对问题进行分类
        self.prop_dict = dict()         # 存放识别得到的属性名
        self.searchPath = 'none->none'  # 程序最终查询路径
        self.attr_dict = defaultdict(list)  # 用以处理爸爸、妈妈、父亲相似度计算短板问题，存放映射
        self.stopFile = dir_path + '/src/qa/data/stopwords.txt'    # 加载停用词文件
        self.attrFile = dir_path + '/src/qa/data/attr_mapping.txt'    # 映射文件
        self.fuzzyFile = dir_path + '/src/qa/data/fuzzy.txt'        # 不完全实体的映射匹配
        self.stopWords = [line.strip() for line in open(self.stopFile,'r',encoding='utf-8').readlines()]    # 构建停用词
        self.fuzzy_dict = dict()                    # 模糊匹配字典
        self.merge = MergeProperty()                # 属性融合

    def searchQustion(self,question:str):
        entity_dict,self.prop_dict = self.getEntProp2dict(question)     # 获取实体和属性识别结果，根据SENT、OENT排序
        question_type = self.quesClass.question_type_predict(question)  # ['SP_O','SPP_O','SO_P','PO_S','OP_S']，进行分类
        entity_len = len(entity_dict)
        prop_len = len(self.prop_dict)
        self.ques = self.removeStopWordsList(question)      # 去除停用词    
        if entity_len == 0:                 # 实体不存在的情况
            res = '未识别到实体，请重新输入您的问题'
            return res,'none->none',0.00,question_type
        # sp_o查询
        if question_type == '0':
            entity = list(entity_dict.keys())[0]
            if prop_len == 1:
                prop =  list(self.prop_dict.keys())[0]
                res,searchPath,sim = self.SPO(entity,prop)      
                if res!= 'none':                                # sp_o直接查询得到结果，直接返回
                    return res,searchPath,sim,question_type
                else:                                           # sp_o得不到结果，则使用相似度匹配法
                    res,searchPath,sim = self.O_By_SP(entity)
                    return res,searchPath,sim,question_type
            else:                                               # 找不到属性，则使用相似度匹配法
                res,searchPath,sim = self.O_By_SP(entity)
                return res,searchPath,sim,question_type        
        # spp_o查询
        elif question_type == '1':
            entity = list(entity_dict.keys())[0]
            if prop_len == 2:                                     # spp_o查询
                if int(list(self.prop_dict.values())[0]) < int(list(self.prop_dict.values())[1]):   # 判断两个属性的前后顺序
                    prop_01 = list(self.prop_dict.keys())[0]
                    prop_02 = list(self.prop_dict.keys())[1]
                else:
                    prop_01 = list(self.prop_dict.keys())[1]
                    prop_02 = list(self.prop_dict.keys())[0]            
                res,searchPath,sim = self.SPPO(entity,prop_01,prop_02)
                if res !='none':
                    return res,searchPath,sim,question_type
                else:
                    res,searchPath,sim = self.O_By_SPP(entity)          # 直接查询无结果，使用相似度匹配算法
                    return res,searchPath,question_type                    
            else:
                res,searchPath,sim = self.O_By_SPP(entity)          # 直接查询无结果，使用相似度匹配算法
                return res,searchPath,sim,question_type
        # SO_P关系查询
        elif question_type == '2':
            if entity_len == 2:
                res,searchPath,sim = self.P_By_SO(entity_dict)                 # 关系查询，返回结果和查询路径
                return res,searchPath,sim,question_type
            else:
                res = '识别到问题类型未SO->P，但实体数目低于或超过了2个，请重新输入。'
                return res, 'none->none',0.00,question_type
        # PO_S查询、 OP_S查询
        elif question_type == '3' or question_type == '4':
            entity = list(entity_dict.keys())[0]
            if prop_len == 1:
                prop =  list(self.prop_dict.keys())[0]
                res,searchPath,sim = self.POS(entity,prop)
                if res!= 'none':                                            # PO_S直接查询，若有结果，直接返回
                    return res,searchPath,sim,question_type
                else:                                                       # PO_S无结果，使用相似度计算法进行
                    res,searchPath,sim = self.S_By_PO(entity)
                    return res,searchPath,sim,question_type
            else:
                res,searchPath,sim = self.S_By_PO(entity)
                return res,searchPath,sim,question_type                 

    ## 给定实体和属性，查询属性值
    def SPO(self,entity,prop):
        print('搜索类型：SPO')
        res = list((self.query.SP_O(entity,prop)).values())      # 直接通过实体名+属性值查询
        ans = []
        if res:
            # print(res)
            
            ans.append(','.join(res))
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
    # 给定P和O,查询S
    def POS(self,entity,prop):
        print('搜索类型：POS')
        res = []
        neo4j_ans = dict()
        neo4j_ans = self.query.OP_S(prop,entity)
        if neo4j_ans:
            res = list(neo4j_ans.values())  # 根据属性和属性值，neo4j直接查询对应的实体
        ans = []
        if res:
            temp_str = ','.join(res)
            ans.append(temp_str)
            searchPath = prop+ '->' + entity
            sim_val = 1.00
            return ans[0],searchPath,sim_val
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
        res_01,searchPath_01,sim_01 = self.O_By_SP(entity)
        self.ans_list = [res_01,searchPath_01,sim_01]
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
                # print('相似问题顺序：%s,相似度%s'%(ques[r_name_index+1],sim_val))                 
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
                    # print('相似问题：%s，相似度:%s,查询路径：%s'%(que,sim_val,searchPath))
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
        start_time = time.time()
        ans = self.ner.predict(question)
        if not ans:
            ans = ans = self.ner.predict(question + '?')
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
        # print('ES查询耗时：%s'%(end_time-start_time))
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


class GiveFlaskWebData(object):
    def __init__(self,):
        self.qa = AnswerByOwnthink()
        self.query = Query()

    def getWebTypeData(self, question: str):
        data = []
        link = []
        start_time = time.time()
        ans, searchPath, sim, question_type = self.qa.searchQustion(question)
        end_time = time.time()
        timeUsed = str(int(float('%.3f' % (end_time - start_time)) * 1000)) + 'ms'
        ques_type_list =  ['SP->O','SPP->O','SO->P','PO->S','OP->S']
        question_type = ques_type_list[int(question_type)]
        answer = {
            '查询路径': question_type,
            '返回答案': ans,
            '答案得分': sim,
            '用时':timeUsed
        }
        if ans == '未识别到实体，请重新输入您的问题' or ans == '识别到问题类型未SO->P，但实体数目低于或超过了2个，请重新输入。':
            return answer, data, link
        elif ans == '实体无关系词，查询结果：无':
            res,que = self.qa.getSimEntFromElastic(question)
            if res != 'none':
                ans = '您输入的问句中实体不存在，您是否想查询：%s'%(que)
                answer = {
                    '查询路径': question_type,
                    '返回答案': ans,
                    '答案得分': sim,
                    '用时':timeUsed
                    }
                return answer, data, link
            else:
                return answer, data, link
        
        if question_type == 'SP->O':
            ent_attr_list = searchPath.split('->')
            res_list = self.query.S_P(ent_attr_list[0])
            idx = 1
            temp_dict = {
                # 'id':idx,
                'name': ent_attr_list[0],
                'symbolSize': 90,
                'category': 0
            }
            data.append(temp_dict)
            r_name_list = []
            m_name_list = [ent_attr_list[0]]
            for ans_dict in res_list:
                r_name = ans_dict['r.name']
                m_name = ans_dict['m.name']
                if r_name not in r_name_list and len(r_name_list) <= 20 and m_name not in m_name_list:
                    r_name_list.append(r_name)
                    idx += 1
                    m_name = m_name.replace('\n',',')
                    temp_dict = {
                        # 'id':idx,
                        'name': m_name,
                        'symbolSize': 75,
                        'category': 1
                    }
                    if m_name in ans:
                        temp_dict['category'] = 2
                        temp_dict['name'] = ans
                        m_name = ans
                    if temp_dict not in data:
                        data.append(temp_dict)
                    link_dict = {
                            'source': ent_attr_list[0],
                            'target': m_name,
                            'name': r_name
                        }
                    if link_dict not in link:

                        link.append(link_dict)
            
            temp_dict = {
                # 'id':idx,
                'name': ans,
                'symbolSize': 75,
                'category': 2
            }
            if temp_dict not in data:
                data.append(temp_dict)
            link_dict = {
                    'source': ent_attr_list[0],
                    'target': ans,
                    'name': ent_attr_list[1]
                }
            if link_dict not in link:
                link.append(link_dict)
            return answer, data, link
        elif question_type == 'SPP->O':
            ent_attr_list = searchPath.split('->')
            ent_o1 = ent_attr_list[2]
            res_list = self.query.S_P(ent_attr_list[0])
            temp_dict = {
                'name': ent_attr_list[0],
                'symbolSize': 90,
                'category': 0
            }
            data.append(temp_dict)
            r_name_list = []
            for ans_dict in res_list:
                r_name = ans_dict['r.name']
                m_name = ans_dict['m.name']
                if r_name not in r_name_list and len(r_name_list) <=20 and m_name != ans:
                    r_name_list.append(r_name)
                    m_name = m_name.replace('\n',',')
                    temp_dict = {
                        'name': m_name,
                        'symbolSize': 75,
                        'category': 1
                    }
                    if m_name == ent_o1:
                        temp_dict['category'] = 2
                    if temp_dict not in data :
                        data.append(temp_dict)
                    link_dict = {
                            'source': ent_attr_list[0],
                            'target': m_name,
                            'name': r_name
                        }
                    if link_dict not in link:
                        link.append(link_dict)
            temp_dict = {
                'name':ans,
                'symbolSize': 90,
                'category': 2
            }
            if temp_dict not in data:
                data.append(temp_dict)
            link_dict = {
            'source': ent_attr_list[2],
            'target': ans,
            'name': ent_attr_list[-1]
            }
            if link_dict not in link:
                link.append(link_dict)
            return answer, data, link
        elif question_type == 'PO->S' or question_type == 'OP->S':
            ent_attr_list = searchPath.split('->')
            res_list = self.query.OP_S_ByEntity(ent_attr_list[-1])
            temp_dict = {
                'name': ent_attr_list[-1],
                'symbolSize': 90,
                'category': 0
            }
            data.append(temp_dict)
            r_name_list = []
            for ans_dict in res_list:
                r_name = ans_dict['r.name']
                m_name = ans_dict['m.name']
                if r_name not in r_name_list and len(r_name_list) <=20 and m_name != ent_attr_list[-1]:
                    r_name_list.append(r_name)
                    m_name = m_name.replace('\n',',')
                    temp_dict = {
                        'name': m_name,
                        'symbolSize': 75,
                        'category': 1
                    }
                    if m_name in  ans:
                        temp_dict['category'] = 2
                        temp_dict['name'] = ans
                        m_name = ans
                    if temp_dict not in data:
                        data.append(temp_dict)
                    link_dict = {
                        'source': m_name,
                        'target': ent_attr_list[-1], 
                        'name':r_name
                    }
                    if link_dict not in link:

                        link.append(link_dict)
            
            return answer, data, link
        elif question_type == 'SO->P':
            if '-[r]-' in searchPath:
                ent_attr_list = searchPath.split('-[r]-')
                res_list = self.query.S_P(ent_attr_list[0])
                temp_dict = {
                    'name': ent_attr_list[0],
                    'symbolSize': 90,
                    'category': 0
                }
                data.append(temp_dict)
                r_name_list = []
                for ans_dict in res_list:
                    r_name = ans_dict['r.name']
                    m_name = ans_dict['m.name']
                    if r_name not in r_name_list and len(r_name_list) <=20:
                        r_name_list.append(r_name)
                        temp_dict = {
                            'name': m_name,
                            'symbolSize': 75,
                            'category': 1
                        }
                        if m_name == ent_attr_list[-1]:
                            temp_dict['category'] = 2
                        if temp_dict not in data:
                            data.append(temp_dict)
                        link_dict = {
                                'source': ent_attr_list[0],
                                'target': m_name,
                                'name': r_name
                            }
                        if link_dict not in link:
                            link.append(link_dict)
                ans_spo_list = ans.split(':')
                temp_dict = {
                    'name': ent_attr_list[-1],
                    'symbolSize': 75,
                    'category': 2
                }
                link_dict = {
                    'source': ent_attr_list[0],
                    'target': ent_attr_list[-1],
                    'name': ans_spo_list[1]
                    }
                if temp_dict not in data:
                    data.append(temp_dict)        
                if link_dict not in link:
                    link.append(link_dict)
                return answer,data,link
            else:
                ent_attr_list = searchPath.split('-')
                temp_dict = {
                    'name': ent_attr_list[0],
                    'symbolSize': 70,
                    'category': 0
                }
                data.append(temp_dict)
                temp_dict = {
                    'name': ent_attr_list[-1],
                    'symbolSize': 70,
                    'category': 0
                }
                data.append(temp_dict)
                searchPath = searchPath.replace(ent_attr_list[0],'').replace(ent_attr_list[-1],'')
                link_dict = {
                    'source': ent_attr_list[0],
                    'target': ent_attr_list[-1],
                    'name': searchPath
                }
                link.append(link_dict)
            return answer,data,link


