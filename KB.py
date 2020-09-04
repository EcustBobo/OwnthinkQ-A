from neo4j import GraphDatabase
import time
from py2neo import Node, Relationship, Graph
import os
import re
# import chardet

neo4j_url = os.getenv('NEO4J_BASE')
if neo4j_url == 'http://keylab.jios.org:7474':
    neo4j_url = "127.0.0.1:7474"
    toInt = 'toInteger'
else:
    neo4j_url = "192.168.0.55:7474"
    toInt = 'toInt'
print('连接neo4j数据库')
session = Graph(neo4j_url, username="neo4j", password="950415")
print('neo4j数据库连接完成,使用服务的地址为：%s' % (neo4j_url))


class Query(object):
    def __init__(self,):
        self.string = 'none'

    # 直接查询，根据实体和属性查询属性值，带方向
    def SP_O(self, s, p):
        string = "MATCH (a:Entity{name:'%s'})-[r:Relation{name:'%s'}]->(b:Entity) RETURN b.name" % (
            s, p)
        rel = session.run(string).data()
        ans = dict()
        for i in range(0, len(rel)):
            temp = rel[i]
            ans[i] = temp['b.name']
        return ans
    # 直接查询，根据两个属性值直接查一步关系

    def SO_P(self, s, o):
        string = "MATCH (a:Entity{name:'%s'})-[r:Relation]->(b:Entity{name:'%s'}) RETURN r.name " % (
            s, o)
        rel = session.run(string).data()
        ans = dict()
        for i in range(0, len(rel)):
            temp = rel[i]
            ans[i] = temp['r.name']
        return ans
    # 关系查询，查询与实体相关的关系，以列表形式返回

    def S_P(self, s):
        string = "MATCH (n:Entity{name:'%s'})-[r:Relation]->(m:Entity) RETURN r.name,m.name" % (
            s)
        rel = session.run(string).data()
        try:
            rel = session.run(string).data()
        except:
            pass
        return rel

    # 直接查询，根据实体+属性1+属性2，直接查询结果，字典形式返回
    def SPP_O(self, s1, p1, p2):
        string = "MATCH (a:Entity{name:'%s'})-[r1:Relation{name:'%s'}]->(b:Entity), (b:Entity)-[r2:Relation{name:'%s'}]->(c:Entity) RETURN c.name" % (
            s1, p1, p2)
        rel = session.run(string).data()
        ans = dict()
        for i in range(0, len(rel)):
            temp = rel[i]
            ans[i] = temp['c.name']
        # print(ans)
        return ans

    # 查询实体相关的歧义关系，带有方向
    def qiyiByEntity(self, s):
        p_qiyi = '歧义关系'
        string = "MATCH (a:Entity{name:'%s'})-[r:Relation{name:'%s'}]->(b:Entity) RETURN b.name" % (
            s, p_qiyi)
        rel = session.run(string).data()
        qiyi_entity_list = []
        for i in range(len(rel)):
            temp = rel[i]
            qiyi_entity_list.append(temp['b.name'])
        if s not in qiyi_entity_list:
            qiyi_entity_list.insert(0, s)
        return qiyi_entity_list

    # 查询实体的歧义实体，并根据歧义实体的权重对实体进行排序，返回歧义实体的所有属性和属性值
    def Q_Z_ByEntity(self, s):
        rel = []
        # string = "match (m:Entity)-[r:Relation{name:'歧义关系'}]->(n:Entity)-[r1:Relation{name:'歧义权重'}]->(n1:Entity) where m.name='%s' return m.name,r.name,n.name, %s(n1.name) as weight order by  %s(n1.name) desc"%(s,toInt,toInt)
        string = "match (m:Entity)-[r:Relation{name:'歧义关系'}]->(n:Entity)-[r1:Relation{name:'歧义权重'}]->(n1:Entity), (n:Entity)-[r2:Relation]->(n2) where m.name='%s' return m.name,r.name,n.name, %s(n1.name) as weight,r2.name, n2.name order by  %s(n1.name) desc" % (s, toInt, toInt)
        try:
            rel = session.run(string).data()
        except:
            pass
        # print(rel)
        return rel
    # 属性值的前属性查询，返回与属性值相关的属性和实体

    def OP_S_ByEntity(self, s):
        rel = []
        string = "match (m:Entity)-[r:Relation]->(n:Entity{name:'%s'}) return m.name,r.name" % (
            s)
        try:
            rel = session.run(string).data()
        except:
            pass
        return rel
    # 根据属性和属性值，直接查询实体s,字典形式返回

    def OP_S(self, p, o):
        rel = []
        string = "match (m:Entity)-[r:Relation{name:'%s'}]->(n:Entity{name:'%s'}) return m.name" % (
            p, o)
        rel = session.run(string).data()
        ans = dict()
        for i in range(0, len(rel)):
            temp = rel[i]
            ans[i] = temp['m.name']
        return ans

    # 最小路径查询，查询两个节点之间五步关系的最小路径
    def P_By_SO(self, s, o):
        rel = []
        string = "match (m:Entity{name:'%s'}),(n:Entity{name:'%s'}),p=shortestpath((m)-[*..5]-(n)) return p" % (
            s, o)
        rel = session.run(string).data()
        ans = rel[0]['p']
        ans = str(ans).replace("[:Relation {name: '", '[').replace("'}]", ']')
        ans_list = ans.split('-')
        for i in range(len(ans_list)):
            if 'u' in ans_list[i]:
                ans_list[i] = ans_list[i].encode().decode('unicode_escape')
        res = '-'.join(ans_list)
        return res


# q = Query()
