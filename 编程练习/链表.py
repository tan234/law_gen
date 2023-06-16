

class ListNode(object):
    def __init__(self,val):
        self.val=val
        self.next=None

def mergeListNode(list1,list2):
    pass





node1=ListNode(1)
node2=ListNode(2)
node3=ListNode(4)

node4=ListNode(2)
node5=ListNode(9)

node1.next=node2
node2.next=node3

node4.next=node5
# res=reverseList(node1)
res=mergeListNode(node1,node4)
while res:
    print(res.val)
    res=res.next