import numpy as np
import copy
import time
from operator import itemgetter

goal = {}

def loc(vec, num):  # 定位num位置
    rows = vec.shape[0]  
    columns= vec.shape[1]  
    for i in range(rows):
        for j in range(columns):
            if num == vec[i][j]:
                return i, j

def get_directions(vec):  # 删除不能移动的位置
    rows = vec.shape[0]  
    columns= vec.shape[1]  
    (x, y) = loc(vec, 0) 
    action = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    if x == 0:  # 不能上移
        action.remove((-1, 0))
    if y == 0:  # 不能左移
        action.remove((0, -1))
    if x == rows - 1:  # 不能下移
        action.remove((1, 0))
    if y == columns - 1:  # 不能右移
        action.remove((0, 1))
    return list(action)


def exchange(vec, action):  # 移动元素
    (x, y) = loc(vec, 0)  # 获取0的位置
    (a, b) = action
    s = copy.deepcopy(vec)
    s[x][y] = vec[x+a][y+b] 
    s[x+a][y+b] = 0
    return s


def get_distance(vec1, vec2):  # vec1目标矩阵，vec2当前矩阵 曼哈顿距离
    rows = vec1.shape[0]  
    columns= vec1.shape[1]  
    dis = 0
    for i in range(rows):
        for j in range(columns):
            if vec1[i][j] != vec2[i][j] and vec2[i][j] != 0:
                k, m = loc(vec1, vec2[i][j])
                d = abs(i - k) + abs(j - m)
                dis += d
    return dis


def expand(now, directions, step): 
    rs = [] 
    for action in directions:
        tmp = {}
        tmp['parent'] = now
        tmp['vec'] = (exchange(now['vec'], action))
        tmp['distance'] = get_distance(goal['vec'], tmp['vec'])
        tmp['step'] = step + 1
        # f=g+h
        tmp['distance'] = tmp['distance'] + tmp['step']
        tmp['action'] = get_directions(tmp['vec'])
        rs.append(tmp)
    return rs

def sort_list(node_list):  # 对列表从大到小排序
    return sorted(node_list, key=itemgetter('distance'), reverse=True)

def get_parent(node):
    return node['parent']

def show_vec(vec):
    rows = vec.shape[0]  
    columns = vec.shape[1]  
    for i in range(rows):
        for j in range(columns):
            if(vec[i][j]==0):
                print(' ',end=' ')
            else:
                print(vec[i][j],end=' ')
        print('')

def transform(list_):
    return np.array([list_[0:3],list_[3:6],list_[6:9]])

def judge(list1,list2):
    tmp1=0
    tmp2=0
    for i in range(len(list1)):
        for j in range(i):
            if(list1[j]<list1[i] and list1[j]!=0):
                tmp1+=1
            if(list2[j]<list2[i] and list2[j]!=0):
                tmp2+=1
    if((tmp1+tmp2)%2==0):
        return True
    return False



def main():
    start_time = time.perf_counter() 
    open_list = []
    close_list = []
    print('请输入初始状态：(0-8)的任意序列')
    A=list(map(int,input().split(',')))

    #A=[2,1,3,6,4,5,0,7,8]
    B=[1,2,3,8,0,4,7,6,5]
    if(not judge(A,B)):
        print('no solution!')
        print('初始状态：')
        show_vec(transform(A))
        print('目标状态')
        show_vec(transform(B))
        return
    goal['vec'] = transform(B)

    origin = {}
    origin['vec'] = transform(A)
    origin['distance'] = get_distance(goal['vec'], origin['vec'])
    origin['step'] = 0
    origin['action'] = get_directions(origin['vec'])
    origin['parent'] = {}

    if (origin['vec'] == goal['vec']).all():
        print('初始矩阵与目标矩阵相同')
        return

    open_list.append(origin)

    while open_list:

        children = []

        node = open_list.pop()  
        close_list.append(node)  

        if (node['vec'] == goal['vec']).all(): 
            end_time = time.perf_counter()  
            print('Size of the search nodes:' +str(len(open_list)+len(close_list)) + '\n')
            print('size of close_list：' + str(len(close_list)) + '\n')
            print('size of open_list：' + str(len(open_list)) + '\n')
            print('running time：' + str(end_time - start_time) + '\n')
            print('path length:' + str(node['distance']) + '\n')
            print('solution: ' + '\n')
            i = 0
            ways = []
            while close_list:
                ways.append(node['vec'])
                node = get_parent(node)
                if(node['vec'] == origin['vec']).all():
                    ways.append(node['vec'])
                    break
            while ways:
                print('step:'+str(i))
                print('------------------------')
                show_vec(ways.pop())
                print('------------------------')
                i += 1
            return

        children = expand(node, node['action'], node['step'])
        for child in children:
            close_flag = False
            open_flag = False
            for i in range(len(open_list)):
                if (child['vec'] == open_list[i]['vec']).all():
                    open_flag = True
                    j=i
                    break
            for i in range(len(close_list)):
                if(child['vec'] == close_list[i]).all():
                    close_flag = True
                    break
            if open_flag == True: 
                if child['distance'] < open_list[j]['distance']:
                    
                    open_list[j]=(child)
            elif close_flag == False: 
                open_list.append(child)


        open_list = sort_list(open_list)  # 排序


if __name__ == '__main__':
    main()
