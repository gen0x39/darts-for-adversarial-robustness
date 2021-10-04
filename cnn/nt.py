from collections import namedtuple
Car = namedtuple('Car' , 'color mileage')   # Car = namedtuple('Car', ['color', 'mileage']) と同じ
my_car = Car('red', 3812.4)