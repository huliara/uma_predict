import dask.dataframe as dd
test_array = [[1.0] * 1788] * 1064240

df = dd.from_array(test_array, columns=['a'] * 1788)

print(df)
