
class MyIterable():
    def __init__(self):
        self.data = [1, 2, 3, 4, 5]
    def __iter__(self):
        return MyIterator(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MyIterator():
    def __init__(self, data):
        self.data = data
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter >= len(self.data):
            raise StopIteration()
        data = self.data[self.counter]
        self.counter +=1
        return data


my_iterable = MyIterable()

for d in my_iterable:
    print(d)

print(my_iterable[0])
