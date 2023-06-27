# 1 快排
# 2 编辑距离
# 3 开根号
# 4 最长回文子串
# 5 找到第K大的数
# 6 单词拆分
# 7 硬币
# 8 全排列
# 9 链表相关：1 创建链表，2 反转链表，3合并两个有序列表
# 10 三个数之和为0
# 11 最大连续乘积
# 12 不同路径
# 13 最长不重复子串
# 14 最长公共子序列
# 15 实现dropout

import copy
import random
import re

import numpy as np
import torch
import random
import math

def a(s):
    # 最长不重复子串
    max_=0
    res=[]

    for i in range(len(s)):
        if s[i]  in res:
            idx=res.index(s[i])
            res=res[idx+1:] if idx+1<len(res) else []
        res.append(s[i])
        max_=max(max_,len(res))
    return max_

print(a('adfaw2'))



class al:
    def dropout_15(self, x, p=.5):
        '''实现dropout'''
        n = len(x)
        c_0 = int(p * n)

        res = [0] * c_0 + [1] * (n - c_0)
        random.shuffle(res)
        return [i * j for i, j in zip(x, res)]
    def longestCommonSubsequence_14(self,text1: str, text2: str) -> int:
        '''
        最长公共子序列:给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0
        rouge-l的计算，分子就是最长公共子序列
        '''

        dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]

        for i in range(1, len(text1) + 1):
            for j in range(1, len(text2) + 1):
                if text1[i - 1] == text2[j - 1]:

                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    def lengthOfLongestSubstring_13(self,s: str) -> int:
        '''
        请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
        方法：维护一个list，如果新加的存在，就去掉重复之前的数据
        '''
        # index=0
        res_list = []
        res = 0
        for i in range(len(s)):

            if s[i]  in res_list:
                idx = res_list.index(s[i])
                res_list = res_list[idx + 1:] if idx + 1 < len(res_list) else []

            res_list.append(s[i])
            res = max(len(res_list), res)
        return res

    def uniquePaths_12(self,m: int, n: int) -> int:
        '''
        一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
        机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
        问总共有多少条不同的路径？
        '''
        dp = [[0] * n for i in range(m)]
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[-1][-1]


    # print(a(3, 7))
    # print(uniquePaths(3, 3))

    def maxProduct_11(self,nums) -> int:
        '''
        乘积最大子数组:给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
        两个数组：以i结尾的最大连续乘积和最小连续乘积
        '''

        res_max = nums[0]
        min_dp = [0] * len(nums)
        max_dp = [0] * len(nums)

        min_dp[0] = max_dp[0] = nums[0]

        for i in range(1, len(nums)):
            v = (nums[i] * max_dp[i - 1], nums[i] * min_dp[i - 1], nums[i])
            min_dp[i] = min(v)
            max_dp[i] = max(v)
            res_max = max(max_dp[i], res_max)
        return res_max


    def threeSum_10(self,nums) :
        '''
        三数之和
        给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请
        你返回所有和为 0 且不重复的三元组。
        注意：答案中不可以包含重复的三元组。
        nums = [-1,0,1,2,-1,-4] 输出：[[-1,-1,2],[-1,0,1]]
        重复的情况：（1）target 相等（2）left+1=left and right+1=right

        做法：排序 ；nums[i]是第一个数；取两个数和为-nums[i]
        '''
        nums=sorted(nums)

        res=[]


        for i in range(len(nums)):

            if i>0 and nums[i]==nums[i-1]:
                continue

            left=i+1
            right=len(nums)-1

            while left<right:

                v=nums[left]+nums[right]
                if v==-nums[i]:
                    res.append([nums[i],nums[left],nums[right]])

                    left+=1
                    right-=1

                    while left<right and nums[left]==nums[left-1]:
                        left+=1
                elif v>-nums[i]:
                    right-=1
                else:
                    left+=1

        return res






    # print(threeSum([0,0,0]))


    def min_path_10(nums,w=np.inf):
        '''
        :param nums: 仓库的坐标
        定位：最短路径问题，使用维特比方法
        '''

        # 原点到每个仓库的距离
        state=[np.sqrt((i[0]**2+i[1]**2)) for i in nums]

        # 仓库间距离
        nums=torch.tensor(nums)
        c1 = torch.unsqueeze(nums, dim=1)
        c2 = torch.unsqueeze(nums, dim=0)
        num_distance=torch.sum((c1 - c2) ** 2, dim=-1) ** 0.5
        num_distance=num_distance.tolist()

        # 路径：元素为到达每个点的最短路径，每层更新，返回
        path=[[[i]]for i in range(len(nums))]

        path_all=[path]#保存每层路径，有 W时需要

        # 权重：每层每个节点的权重和（满足条件下最小的）
        dp_w=[[0]*len(nums) for i in range(len(nums))]
        dp_w[0]=state

        #'---每层---'
        for layer in range(1,len(nums)):
            # print('----第%d层-----'%(layer))
            new_path=[0]*len(nums) #每层更新一次路径

            for i in range(len(nums)): #计算一个节点
                weight_path_dict={}# 可达路径：'权重'：[路径]
                for p in range(len(path)):#遍历每个path
                    for p_i in path[p]:# 可能多个最短路径
                        if i not in p_i:
                            weight=dp_w[layer-1][p]+num_distance[i][p]
                            if weight not in weight_path_dict:
                                weight_path_dict[weight]=[p_i+[i]] # ac 权重
                            else:

                                weight_path_dict[weight]=weight_path_dict[weight]+[p_i+[i]]


                if len(weight_path_dict):
                    min_weight=sorted(weight_path_dict.items(),key=lambda x:x[0])[0][0]#最小权重

                    # 权重更新
                    dp_w[layer][i]=min_weight
                    new_path[i]=weight_path_dict[min_weight]#最小权重对应的路径

                else:
                    dp_w[layer][i]=np.inf
                    new_path[i] =[]
            # 路径更新
            path_all.append(new_path)
            path=new_path

        # 有W限制对结果进行筛选
        for i in range(len(dp_w)-1,-1,-1):
            if min(dp_w[i])<=w:
                res_path=path_all[i][np.argmin(dp_w[i])]
                for p in res_path:
                    print ('最短路径：',p)
                return res_path

        # 没有w
        # for i  in path[np.argmin(dp_w[-1])]:
        #     print('最短路径:',i)
        # return path[np.argmin(dp_w[-1])]
    #
    # if __name__ == '__main__':
    #
    #     # 1 每个仓库经过一次，要求库存总量最大，就是用最少的时间，也就是最少的路径
    #     min_path([[0,1],[0,3],[0,2],[0,4],[0,5]])
    #     min_path([[1, 4], [4, 4], [1, 1], [4,1], [2, 2]])
    #     # 2 A，B
    #     min_path([[1,1],[1,2]])
    #     # 3 最大里程限制，早停。对输出结果进行筛选
    #     min_path([[0,1],[0,2],[0,3]],w=2)


    def reverseList(head):
        # 链表反转
        '''
        head遍历：
        temp=head.next
        head=temp
        用后一项指向前一项：
        head.next=P
        P=head
        上面两个步骤都引入了一个新的变量，每次都是用一个新的变量存head.next
        '''
        P=None
        while head:
            temp=head.next

            head.next=P
            P=head

            head=temp
        return P


    def mergeTwoLists_9(list1,list2):
        '''
        合并有序列表:
        1 输入的是两个链表的头结点
        2 输入是链表的头结点
        3 创建一个节点，
        4 返回创建节点的头节点
        '''
        # 合并有序列表
        # list1,list2:listnode
        res=cur=ListNode(-1)

        while list1 and list2:
            if list1.val<list2.val:
                res.next=list1
                list1=list1.next
            else:
                res.next=list2
                list2 = list2.next

            res=res.next
            # print(res.val)
        if list1:
            res.next=list1
            # print(res.val)
        elif list2 :
            res.next = list2


        return cur.next
    # print(text(l1 = [1,2,4], l2 = [1,3,4,5,7,9]))


    # 链表节点类
    class ListNode:
        def __init__(self, val):
            self.val = val
            self.next = None


    class LinkedList:
        def __init__(self,head=None):
            self.head = head

        def append(self,new_node):
            current=self.head
            if not current:
                self.head=new_node#如果head没有
            else:#加到链尾
                while current.next:
                    current=current.next
                current.next=new_node

    Node11=ListNode(1)
    Node12=ListNode(2)
    Node13=ListNode(4)

    Node21=ListNode(1)
    Node22=ListNode(3)
    Node23=ListNode(4)
    Node24=ListNode(5)
    Node25=ListNode(7)

    list1 = LinkedList()
    list1.append(Node11)
    list1.append(Node12)
    list1.append(Node13)



    list2 = LinkedList()
    list2.append(Node21)
    list2.append(Node22)
    list2.append(Node23)
    list2.append(Node24)
    list2.append(Node25)



    def singleNumber(nums) -> int:
        '''
        给你一个 非空 整数数组 nums ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
        你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。
        '''
        for i in range(len(nums)-1,-1,-1):

            a=nums.pop()
            if a not in nums:
                return a
            else:nums.insert(0,a)



    # print(singleNumber([1,0,1]))

    def permute_8(nums) :
        '''
        给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
        输入：nums = [1,2,3]
        输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
        '''
        # 方法一：动态规划，每次加一个元素,注意深拷贝，列表的index删除和插入
        if len(nums)<1:return []
        res=[[nums[0]]]#[[1]]

        i=1
        while i<len(nums):
            res_new = []
            for v in res:
                for j in range(len(v)):
                    if j==0:
                        v.insert(j,nums[i])
                        res_new.append(copy.deepcopy(v))
                        v.pop(j)
                    v.insert(j+1, nums[i])
                    res_new.append(copy.deepcopy(v))
                    v.pop(j+1)
            i+=1
            res=res_new

        return res




    def  waysToChange_7( n: int):
        '''硬币。给定数量不限的硬币，币值为25分、10分、5分和1分，计算n分有几种表示法(给结果对1000000007求余)'''
        # 方法一：动态规划
        nums=[25,10,5,1]
        dp=[0]*(n+1)
        dp[0]=1

        for j in nums:
            for i in range(1,n+1):
                if i>=j:
                    dp[i]=(dp[i]+dp[i-j])%1000000007

        return dp[-1]


    # print(waysToChange_7(5))

    def wordBreak_6(s,wordDict):
        '''
        给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
        注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
        '''
        dp=[False]*(len(s)+1)
        w_l=[len(i)for i in wordDict if len(i)<=len(s)]
        if not len(w_l):return False

        for i in range(min(w_l),max(w_l)+1):
            if s[:i] in wordDict:
                dp[i]=True

        for i in range(min(w_l),len(s)+1):
            if dp[i]:
                for j in range(min(w_l),max(w_l)+1):

                    if i+j<=len(s) and s[i:i+j] in wordDict:
                        dp[i+j]=True
        return dp[-1]




    # print('F',wordBreak(s = "catsandog",wordDict=["cats", "dog", "sand", "and", "cat"]))
    # print('T',wordBreak(s = "a", wordDict =  ["a"]))
    # print('F',wordBreak(s = "a", wordDict =  ["aa","aaa","aaaa","aaaaa","aaaaaa"]))
    # print('T',wordBreak(s = "carts", wordDict =  ["car", "cart", "ts", "and", "cat"]))
    # print('T',wordBreak(s = "leetcode", wordDict =  ["leet", "code"]))
    # print('T',wordBreak(s = "applepenapple", wordDict = ["apple", "pen"]))

    def findKthLargest_5(nums, k: int) :
        '''
        给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
        请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
        你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。
        示例 1:
        输入: [3,2,1,5,6,4], k = 2
        输出: 5
        示例 2:
        输入: [3,2,3,1,2,4,5,5,6], k = 4
        输出: 4
        '''
        # 方法一：快排
        mid=nums[0]
        left=[]
        right=[]

        for i in range(1,len(nums)):
            if nums[i]>mid:
                right.append(nums[i])
            else:left.append(nums[i])


        if len(right)==k-1:
            return mid
        elif len(right)>k-1:
            return findKthLargest_5(right,k)
        else:
            k=k-1-len(right)

            return findKthLargest_5(left,k)



    # print(findKthLargest([2,1], k = 2))


    def longestPalindrome_4( s: str) -> str:
        '''
        输入：s = "babad"
        输出："bab"
        解释："aba" 同样是符合题意的答案。
        示例 2：
        输入：s = "cbbd"
        输出："bb"
        '''

        if  len(s)<2:return s
        mid=0
        res=s[mid]
        while mid<len(s):
            word = s[mid]

            right=mid+1
            left = mid - 1 if mid > 0 else 0

            while right<len(s) and s[right]==s[mid]:
                word=word+s[right]
                right+=1

            while left>=0 and right<len(s) and s[left]==s[right]:
                word=s[left]+word+s[right]
                right+=1
                left-=1
            res=word if  len(word)>len(res) else res
            mid+=1
        return res



    # print(longestPalindrome_4('cbbd'))


    def sqrt_3(n):
        # 开根号
        x0=0.25
        x1=x0/2+n/(2*x0)
        while abs(x1-x0)>0.0001:
            x0=x1
            x1=x0/2+n/(2*x0)

        return x1
    # print(sqrt_3(2))

    def leven_distance_2(s1,s2):
        l1=len(s1)
        l2=len(s2)
        dp=[[0]*(l2+1)for _ in range(l1+1) ]

        for i in range(l1+1):
            dp[i][0]=i
        for j in range(l2+1):
            dp[0][j]=j

        for i in range(1,l1+1):
            for j in range(1,l2+1):
                dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+int(s1[i-1]!=s2[j-1]))


        return dp[-1][-1]

    # print(leven_distance_2('abs','an'))


    def quick_sort_1(nums):
        # 快排
        if not len(nums):
            return []

        mid=nums[0]
        right=[]
        left=[]
        for i in range(1,len(nums)):
            if nums[i]>mid:
               right.append(nums[i])
            else:left.append(nums[i])

        return quick_sort_1(left)+[mid]+quick_sort_1(right)

    # print(quick_sort_1([1]))


