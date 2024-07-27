import io
from BSTSimPieceSegment import BSTSimPieceSegment

    

class BST:
    def __init__(self):
        self.root = None
        self.minNode = None

    @property
    def count(self):
        count = 0
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            count += 1
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return count

    def insert(self, node):
        if self.root is None:
            self.root = node
            self.minNode = self.root
        else:
            self._insert(self.root, node)
            if node.get_a_min<self.minNode.get_a_min:
                self.minNode = node


    def _insert(self, node, new_node):
        if node.get_a_min < new_node.get_a_min:
            if node.right is None:
                node.right = new_node
                #self.mergeNode(node)
            else:
                self._insert(node.right, new_node)
        else:
            if node.left is None:
                node.left = new_node
                #self.mergeNode(node)
            else:
                self._insert(node.left, new_node)

    def mergeNode(self, node):
        """
        合并节点

        Args:
            node: 要合并的节点

        Returns:
            合并后的节点
        """

        # 获取节点的左右孩子
        left_child = node.left
        right_child = node.right

        # 计算左右孩子的交集
        left_intersection = self.intersection(node, left_child)
        right_intersection = self.intersection(node, right_child)
        if(left_intersection==0 and right_intersection==0):
            return node

        # 比较交集大小
        if left_intersection >= right_intersection:
            node.init_timestamp.extend(node.left.init_timestamp)
            node.a_min = node.a_min
            node.a_max = node.left.a_max

            node.left = None

        else:
            node.init_timestamp.extend(node.right.init_timestamp)
            node.a_min = node.right.a_min
            node.a_max = node.a_max

            node.right = None

        return node


    def intersection(self, node1, node2):
        """
        计算两个节点的交集

        Args:
            node1: 第一个节点,较小的那个
            node2: 第二个节点，较大的那个

        Returns:
            交集的大小
        """

        if node1 is None or node2 is None:
            return 0
        if node1.a_max<=node2.a_min or node2.a_max < node1.a_min:
            return 0
        return (node1.a_max -  node2.a_min)
    
    def inorder_traversal(self, node):

        if node is None:
            return

        self.inorder_traversal(node.left)

        print(f"Node: {self.format_sim_piece_segment(node)}")

        self.inorder_traversal(node.right)

    def format_sim_piece_segment(self, node):
        return f"SimPieceSegment(init_timestamp={node.init_timestamp}, a_min={node.a_min}, a_max={node.a_max})"
    
    def to_list(self):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        result = []
        def inorder(node):
            if node:
                inorder(node.left)
                result.append(node)
                inorder(node.right)
        
        inorder(self.root)
        return result
    
    def to_list_non_recursive(self):
        stack = []
        result = []
        current = self.root

        while current or stack:
            # 把当前节点和所有左子节点入栈
            while current:
                stack.append(current)
                current = current.left

            # 当前节点为空，说明左边已经访问完，弹栈访问节点
            current = stack.pop()
            result.append(current)

            # 转到右子树
            current = current.right

        return result


    def mergeAll(self):
        res = []
        segments = self.to_list()

        i = 0
        while(i <len(segments)-1):
            for j in range(i + 1, len(segments)):
                if segments[i].a_max>segments[j].a_min:
                    if segments[i].a_max<=segments[j].a_max:
                        res.append(BSTSimPieceSegment(segments[j].a_min, segments[i].a_max, segments[i].init_timestamp.extend(segments[j].init_timestamp)))
                    else:
                        res.append(BSTSimPieceSegment(segments[j].a_min, segments[j].a_max, segments[i].init_timestamp.extend(segments[j].init_timestamp)))
                else:
                    break
        
        return res



if __name__ == "__main__":
    bst = BST()
    bst.insert(BSTSimPieceSegment(1, 3.1, 4.0))
    bst.insert(BSTSimPieceSegment(5, 1.0, 2.0))
    bst.insert(BSTSimPieceSegment(10, 2.1, 3.5))
    bst.insert(BSTSimPieceSegment(17, 1.5, 2.7))

    bst.inorder_traversal(bst.root)
