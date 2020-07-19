def test_func(*args, **kwargs) -> None:
    """testなfunc
    """
    for x in args:
        print("x", x)
    for y in kwargs:
        print("y", y)


def keyword_func(*, x: str, y: str):
    """[summary]

    Args:
        x (str): [description]
        y (str): [description]
    """
    print(x, y)


def annotation_func(x: str, y: str) -> str:
    """[summary]

    Args:
        x (str): [description]
        y (str): [description]

    Raises:
        BaseException: [description]

    Returns:
        str: [description]
    """
    try:
        return x + y
    except BaseException:
        return x + y
    finally:
        raise BaseException


for i in range(10):
    print(i)
else:
    print("done")

for i, s in enumerate("hogehoge"):
    print(i, s)

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = [2, 4, 6, 20]
print(a + b)
print(b[:-2])

dic = dict(hoge=1, fuga=2, piyo="piyo")
print(dic)
print(dic.pop("hoge"))

for v in dic.items():
    print(v)

setA = set(a)
setB = set(b)

print(setA | setB)
print(setA - setB)
print(setA & setB)
print(setA ^ setB)


generator10 = (x for x in range(10))

print([generator10])

print(test_func("hoge", "fuga", "piyo", test="test"))

keyword_func(x="1", y="2")

# annotation_func(1, 3)
