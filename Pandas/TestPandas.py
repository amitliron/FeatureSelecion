import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.DataFrame(np.array(([1, 1, np.nan], [np.nan, np.nan, np.nan],[np.nan, 1, 1],[np.nan, np.nan, np.nan],[np.nan, 1, np.nan])),
                  index=['test1', 'test2', 'test3', 'test4', 'test5'],
                  columns=['check1', 'check2', 'check3'])
df.fillna(value=0, inplace=True)
mlb = MultiLabelBinarizer()
plt.scatter(*np.where(df)[::-1])
plt.xticks(range(df.shape[1]), df.columns)
plt.yticks(range(df.shape[0]), df.index)
plt.show()
None


# df = pd.DataFrame(np.array(([1, 2, 3], [4, 5, 6])),
#                   index=['mouse', 'rabbit'],
#                   columns=['one', 'two', 'three'])
#
# print(df)
#
# print(df.filter(items=['one', 'three']))
#
# df = pd.DataFrame({'temp_c': [17.0, 25.0]}, index=['Portland', 'Berkeley'])
#
# print(df)
#
# print(df.assign(temp_f=lambda x: x.temp_c * 9 / 5 + 32))


#
# df = pd.DataFrame({'A': list('aaabbccc'), 'B': ['x', 'x', np.nan, np.nan, np.nan, np.nan, 'x', 'x'], 'C': ['x', 'x', np.nan, np.nan, np.nan, np.nan, 'x', 'x']})
# print(df)
# print(df.groupby('A').count())
# print(df.groupby('A').size())

#
# df = pd.DataFrame({'f_idx' : [1, 1, 2, 1, 2,3, 1], 'day': [1, 1, 7, 3, 3, 1, 3], 'test_result':['OK', 'NULL', 'NULL', 'NULL', 'NULL', 'OK', 'OK']})
#
# print(df)
#
# #res1 = df.groupby('f_idx').groupby('day').mean()
# #res1 = df.groupby(['f_idx', 'day']).apply(lambda x: x.test_result == "OK")
# #res1 = (df.assign(p_res=(df['test_result'] == 'OK').view('i1')).groupby(['f_idx','day']))['p_res'].mean().reset_index(name='p_res')
# res1 = (df.assign(p_res=(df['test_result'] == 'OK').view('i1')).groupby(['f_idx','day']))['p_res'].mean().mean(level=1).reset_index(name='p_res')
#
# print(res1)
#
# #print(res.describe())
