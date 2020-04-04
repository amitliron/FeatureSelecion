from FeatureSelection import filter_methods
from FeatureSelection import embedded_methods

def filter_and_wrapper(df, X, y):
    from FeatureSelection import filter_methods
    filter_res = filter_methods.filter_methods(df, X, y)

    print(filter_res[0][2])

    None


def hybrid_methods(df, X, y):
    res1 = filter_and_wrapper(df, X, y)
    res = []
    res.append(('hybrid_methods', 'filter\nand\nwrapper', res1))
    return res