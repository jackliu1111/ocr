import re
def label(result):
    print(result)
    #print(result)
    '''
    采用正则表达式。
    规则为：
    1.直接匹配姓名、性别、民族、出生、住址。若成功，则直接return字典
    2.若1失败，匹配看是否是国徽面
    3.若2失败，可能是【姓名、性别、民族、出生、住址】几个关键字出现问题，尝试在falsedict中匹配，
    4.若3失败，return {}
    '''

    txt = ''
    for i in range(len(result)):
        txt += result[i][0]
    txt = txt.replace(' ', '')
    dict = {}
    falsedict = {'姓名': ['姓名'], '性别': ['性别'], '民族': ['民族'], '出生': ['出生'], '住址': ['住址', '佳址']}       #关键字纠错

    patten = r"姓名" + r"([\u4e00-\u9fa5]+?)" + r"性别" + r"([\u4e00-\u9fa5]{1}?)" + "民族" + r"([\u4e00-\u9fa5]{1,5}?)" + \
             "出生" + "(.*)" + "住址" + "(.*)" + "公民身份号码" + r"(\d{18}|(\d{17})X)"
    patten1 = re.compile(patten)
    m = patten1.search(txt)
    if m != None:          #直接匹配
        dict['姓名'] = m.group(1)
        dict['性别'] = m.group(2)
        dict['民族'] = m.group(3)
        dict['出生'] = m.group(4)
        dict['住址'] = m.group(5)
        dict['公民身份号码'] = m.group(6)
        dict['类型'] = '身份证反面'
    else:                #匹配是否为国徽
        patten = r"(.*)中华人民共和国|签发机关(.*)"
        patten1 = re.compile(patten)
        m = patten1.search(txt)
        if m != None:
            dict['类型'] = '身份证正面'
        else:            #开始纠错
            index = 3
            for i in range(len(result)):
                label = result[i][0].replace(' ', '')
                for j in falsedict['住址']:
                    patten = j + "(.*)"
                    patten1 = re.compile(patten)
                    m = patten1.search(label)
                    if m != None:
                        index = i
                        dict['住址'] = m.group(1)
                        break
                if '住址' in dict.keys():
                    patten = r"([\u4e00-\u9fa5]*)" + r'(\d{18}|(\d{17})X)'
                    patten1 = re.compile(patten)
                    m = patten1.search(label)
                    if m != None:
                        dict['公民身份号码'] = m.group(2)
                        break
                    elif m == None and i > index:
                        dict['住址'] += label
                for j in falsedict['姓名']:
                    patten = j + "(.*)"
                    patten1 = re.compile(patten)
                    m = patten1.search(label)
                    if m != None:
                        dict['姓名'] = m.group(1)
                        break
                for j in falsedict['性别']:
                    for k in falsedict['民族']:
                        patten = j + "(.*)" + k + "(.*)"
                        patten1 = re.compile(patten)
                        m = patten1.search(label)
                        if m != None:
                            dict['性别'] = m.group(1)
                            dict['民族'] = m.group(2)
                            break
                for j in falsedict['出生']:
                    patten = j + "(.*)"
                    patten1 = re.compile(patten)
                    m = patten1.search(label)
                    if m != None:
                        dict['出生'] = m.group(1)
                        break
            # if len(dict) == 6:
            #     dict['类型'] = '身份证反面'
            # else:
            #     return {}
    return dict
