def feature_engineering(obj):
    sepal_area = obj.sepal_length + obj.sepal_width
    petal_area = obj.petal_length + obj.petal_width
    
    return sepal_area, petal_area

def feature_engineering_list(instance):
    sepal = instance.data[0] + instance.data[1]
    petal = instance.data[2] + instance.data[3]

    return sepal, petal