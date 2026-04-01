def bubble_sort(arr):
    """
    冒泡排序算法
    
    参数:
        arr: 待排序的列表
        
    返回:
        排序后的列表（原地排序，原列表会被修改）
    """
    n = len(arr)
    
    # 遍历所有数组元素
    for i in range(n):
        # 最后i个元素已经排好序，不需要再比较
        swapped = False
        
        for j in range(0, n - i - 1):
            # 如果当前元素大于下一个元素，则交换它们
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # 如果在这一轮中没有发生交换，说明数组已经有序
        if not swapped:
            break
    
    return arr


def bubble_sort_desc(arr):
    """
    冒泡排序（降序版本）
    
    参数:
        arr: 待排序的列表
        
    返回:
        降序排序后的列表
    """
    n = len(arr)
    
    for i in range(n):
        swapped = False
        
        for j in range(0, n - i - 1):
            # 降序排序：如果当前元素小于下一个元素，则交换它们
            if arr[j] < arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        if not swapped:
            break
    
    return arr


def bubble_sort_optimized(arr, reverse=False):
    """
    优化的冒泡排序算法
    
    参数:
        arr: 待排序的列表
        reverse: 是否降序排序，默认为False（升序）
        
    返回:
        排序后的列表
    """
    n = len(arr)
    
    for i in range(n):
        swapped = False
        
        for j in range(0, n - i - 1):
            # 根据reverse参数决定排序方向
            if (not reverse and arr[j] > arr[j + 1]) or (reverse and arr[j] < arr[j + 1]):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        if not swapped:
            break
    
    return arr


# 测试代码
if __name__ == "__main__":
    # 测试升序排序
    test_arr1 = [64, 34, 25, 12, 22, 11, 90]
    print("原始数组:", test_arr1)
    sorted_arr1 = bubble_sort(test_arr1.copy())
    print("升序排序后:", sorted_arr1)
    
    # 测试降序排序
    test_arr2 = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr2 = bubble_sort_desc(test_arr2.copy())
    print("降序排序后:", sorted_arr2)
    
    # 测试优化版本
    test_arr3 = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr3 = bubble_sort_optimized(test_arr3.copy(), reverse=False)
    print("优化版本升序:", sorted_arr3)
    
    test_arr4 = [64, 34, 25, 12, 22, 11, 90]
    sorted_arr4 = bubble_sort_optimized(test_arr4.copy(), reverse=True)
    print("优化版本降序:", sorted_arr4)
    
    # 测试已排序数组
    test_arr5 = [1, 2, 3, 4, 5]
    sorted_arr5 = bubble_sort(test_arr5.copy())
    print("已排序数组:", test_arr5, "->", sorted_arr5)
    
    # 测试空数组
    test_arr6 = []
    sorted_arr6 = bubble_sort(test_arr6.copy())
    print("空数组:", test_arr6, "->", sorted_arr6)
    
    # 测试单个元素数组
    test_arr7 = [42]
    sorted_arr7 = bubble_sort(test_arr7.copy())
    print("单个元素数组:", test_arr7, "->", sorted_arr7)