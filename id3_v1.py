'''
Những kỹ thuật code cần chú ý: pandas.read_csv() Source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
Pandas Dataframe Source: https://www.geeksforgeeks.org/python-pandas-dataframe/
Pandas.Dataframe.loc xem phần xử lý với index là integer labels Source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html
Pandas.Dataframe.count Source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.count.html
Pandas.Series.value_counts
Pandas.DataFrame.mode
Khi muon lay list thong tin cua header trong Pandas thi dung: list(my_dataframe)  Source: https://stackoverflow.com/questions/19482970/get-list-from-pandas-dataframe-column-headers

Ta nghi cho ids co the chinh sua thanh dang list ngay tu dau thay vi no la range roi chuyen thanh list,
voi lai cai y ong no bao chi so trong pandas chay tu 1 tro di... cung ko dung lam, minh thay chay tu 0 cung duoc

Du me, cai ham kinh nhat la cai split cua ong ni

'''

from __future__ import print_function
import numpy as np
import pandas as pd


class TreeNode(object):
    '''
    Một node trong Decision tree
    '''
    def __init__(self, ids=None, children=[], entropy=0, depth=0):
        '''
        :param ids: chỉ số của data trong node
        :param children:
        :param entropy: là entropy tại node đó, sẽ tìm và điền vào sau
        :param depth: khoảng cách đến node root
        '''
        self.ids = ids   # Chi so cua data trong node nay
        self.entropy = entropy
        self.depth = depth
        self.split_attribute = None  #Chỉ tên thuộc tính mà node ni lưu giữ
        self.children = children  # Danh sách node con của nó
        self.order = None   # Chua biet no là gì
        self.label = None  # Nhãn của node nếu như nó là node lá

    def set_properties(self, split_attribute, order):
        '''
        Hàm thay đổi thuộc tính của node, bao gồm
        :param split_attribute: Tên thuộc tính của node đang giữ
        :param order: Chưa biết là gì
        :return:
        '''
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label


def entropy(freq):
    '''
    Tính toán theo công thức H(T) = H(P_T) = -Xichma(pi*log2(pi))
    :param freq: Là mảng np với các phần tử bên trong là tần suất xuất hiện của từng class trong output
    Ví dụ trong bài toán weather thì nó sẽ là số yes và no
    :return: H(P) - Entropy cua no
    Chu y la loai bo cai distribution bang 0 tai log(0) ko xac dinh
    '''
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0 / float(freq_0.sum())
    return -np.sum(prob_0*np.log2(prob_0))


class DecisionTreeID3(object):
    def __init__(self, max_depth=10, min_samples_split=2, min_gain=1e-4):
        self.root = None
        self.max_depth = max_depth
        self.min_sample_split = min_samples_split
        self.Ntrain = 0
        self.min_gain = min_gain

    def fit(self, data, target):
        '''
        Hàm thực thi chính của chúng ta đây, đầu vào của nó là
        :param data:
        :param target:
        :return:
        '''
        self.Ntrain = data.count()[0]  # Số mẫu để train
        self.data = data     # Dữ liệu X đem xuống
        self.attributes = list(data)  # Lay thong tin cua cac header trong X, tra ra: ['id', 'outlook', 'temperature', 'humidity', 'wind']
        self.target = target    # Target la cai Dataframe y gom integer index va [yes, no, ...,]
        self.labels = target.unique()  # Khi lam the thi no chuyen thanh list ['no', 'yes']

        ids = range(self.Ntrain)   # Mang chi so, no tra ra kieu range(0, 14)
        # Khai bao node root
        self.root = TreeNode(ids=ids, entropy=self._entropy(ids), depth=0)
        queue = [self.root]  # Day chinh la cai queue, chua biet co lam theo cai tim kiem gi day
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)
                if not node.children:  # Neu nhu la node la
                    self._set_label(node)
                queue += node.children  # nguoc lai la node thuong
            else:
                # Neu no khong thoa dieu kien rang buoc lon ban dau thi cho no la node la
                self._set_label(node)



    def predict(self, new_data):
        '''
        :param new_data: la mot data frame moi, moi hang la mot datapoint
        :return: nhan duoc du doan cho moi dong
        '''

        npoints = new_data.count()[0]
        labels = [None] * npoints
        for n in range(npoints):
            x = new_data.iloc[n, :] # Mot diem, tuc mot dong trong do
            # Bat dau tu node root va travel de quy khi chua gap node la
            node = self.root
            while node.children:
                node = node.children[node.order.index(x[node.split_attribute])]
            labels[n] = node.label
        return labels


    def _entropy(self, ids):
        '''
        Tinh entropy cua mot node voi chi so index ids tinh theo cai bang dau vao cua chung ta
        :param ids:
        :return:
        '''
        if len(ids) == 0:
            return 0
        ids = [_ + 1 for _ in ids]  # pandas series index starts from 1 (Neu nhu vua khoi tao thi no la 1 den 15)
        freq = np.array(self.target[ids].value_counts())  # A, cai target[id] voi id la cai mang thi no cung tra ra cai mang dem so phan tu yes, no trong do thoi
        return entropy(freq)

    def _set_label(self, node):
        '''
        Danh nhan cho node, trong truong hop no la node la, vi du trong du lieu nay thi no la yes hoac no
        Don gian la chon cai nao co ti le vote cao nhat, chu y trong truong hop nay no
        :param node:
        :return:
        '''
        target_ids = [_ + 1 for _ in node.ids]
        node.set_label(self.target[target_ids].mode()[0])   # Ben trong la cai label yes hoac no co ti le cao nhat trong danh sach ids

    def _split(self, node):
        '''
        Ham de phan chia thanh cac nhanh con tai mot node
        :param node:
        :return:
        '''
        ids = node.ids
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()
            if len(values) == 1: continue  # entropy = 0
            splits = []
            for val in values:
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id - 1 for sub_id in sub_ids])
            # Khong split mot node khi ma node co qua it diem
            if min(map(len, splits)) < self.min_sample_split: continue

            # information gaim
            HxS = 0
            for split in splits:
                HxS += len(split)*self._entropy(split)/len(ids)
            gain = node.entropy - HxS
            if gain < self.min_gain: continue # dung neu nhu small gain
            if gain > best_gain:
                best_gain = gain
                best_splits = splits
                best_attribute = att
                order = values
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids=split,
                                entropy=self._entropy(split), depth=node.depth + 1) for split in best_splits]
        return child_nodes


print(__doc__)

if __name__ == '__main__':
    df = pd.read_csv('daramnang.csv')
    print('Train data: ')
    print(df.to_string())
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    tree = DecisionTreeID3(max_depth=2, min_samples_split=1)
    tree.fit(X, y)
    # Su dung chinh data train de test
    print('Kiem tra lai cay minh tao duoc dua tren bo du lieu cu: ')
    print(tree.predict(X))
    # Tu tao ra bo du lieu test xem thu
    my_test_data = [{'TT': 1, 'Mau_toc': 'Den', 'Chieu_cao': 'Tam_thuong', 'Can_nang': 'Nhe', 'Dung_thuoc?': 'Khong'},
                    {'TT': 2, 'Mau_toc': 'Den', 'Chieu_cao': 'Tam_thuong', 'Can_nang': 'Nhe', 'Dung_thuoc?': 'Co'},
                    {'TT': 3, 'Mau_toc': 'Bac', 'Chieu_cao': 'Tam_thuong', 'Can_nang': 'Vua_phai', 'Dung_thuoc?': 'Co'},
                    {'TT': 4, 'Mau_toc': 'Ram', 'Chieu_cao': 'Thap', 'Can_nang': 'Nhe', 'Dung_thuoc?': 'Co'}, ]

    df2 = pd.DataFrame(my_test_data)
    print('Du doan cho bo du lieu moi: ')
    print(tree.predict(df2))
