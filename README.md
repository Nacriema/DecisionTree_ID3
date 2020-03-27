# DecisionTree_ID3
Learn and My own Implementation of ID3


## Stopping criteria
* Max depth of the tree
* The minimum number of samples in a split to be considered.
* Minimum gain for splitting.

## Simple test on daramnang.csv

|TT |Mau_toc|Chieu_cao|Can_nang|Dung_thuoc?|Ket_qua|
|---|-------|---------|--------|-----------|-------|
|1  |Den    |Tam_thuoc|Nhe     |Khong      |Bi_ram |
|2  |Den    |Cao      |Vua_phai|Co         |Khong  |
|3  |Ram    |Thap     |Vua_phai|Co         |Khong  |
|4  |Den    |Thap     |Vua_phai|Khong      |Bi_ram |
|5  |Bac    |Tam_thuoc|Nang    |Khong      |Bi_ram |
|6  |Ram    |Cao      |Nang    |Khong      |Khong  |
|7  |Ram    |Tam_thuoc|Nang    |Khong      |Khong  |
|8  |Den    |Thap     |Nhe     |Co         |Khong  |


```python
python id3_v1.py
```

Result:

```python
['Bi_ram', 'Khong', 'Khong', 'Bi_ram', 'Bi_ram', 'Khong', 'Khong', 'Khong']
```
It means 100% accuracy on training set.

And we check with another table

```python
my_test_data = [{'TT': 1, 'Mau_toc': 'Den', 'Chieu_cao': 'Tam_thuong', 'Can_nang': 'Nhe', 'Dung_thuoc?': 'Khong'},
                    {'TT': 2, 'Mau_toc': 'Den', 'Chieu_cao': 'Tam_thuong', 'Can_nang': 'Nhe', 'Dung_thuoc?': 'Co'},
                    {'TT': 3, 'Mau_toc': 'Bac', 'Chieu_cao': 'Tam_thuong', 'Can_nang': 'Vua_phai', 'Dung_thuoc?': 'Co'},
                    {'TT': 4, 'Mau_toc': 'Ram', 'Chieu_cao': 'Thap', 'Can_nang': 'Nhe', 'Dung_thuoc?': 'Co'}, ]

df2 = pd.DataFrame(my_test_data)
print(tree.predict(df2))
```

Then we get predict result: 
```python
Du doan cho bo du lieu moi: 
['Bi_ram', 'Khong', 'Khong', 'Bi_ram']
```