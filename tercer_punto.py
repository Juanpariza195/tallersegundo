# partir la data en dos

data_train = data[:980]
data_test = data[980:]

x = np.array(data_train.drop(['Survived'], 1))
y = np.array(data_train.Survived) # 0 Murió 1 vivió


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Survived'], 1))
y_test_out = np.array(data_test.Survived) # 0 Murió 1 vivió