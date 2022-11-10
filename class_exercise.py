#%%
import csv

class Item:
    pay_rate = 0.8
    item_lst = []
    def __init__(self,name:str, price:float, quantity=0) -> None:
        assert isinstance(name, str), f"name: {name} is not a string."
        assert price>=0, f"price: {price} is less than 0."
        assert quantity>=0, f"quantity: {quantity} is less than 0."
        self.name = name
        self.__price = price
        self.quantity = quantity
        Item.item_lst.append(self)

    def calc_tot_price(self):
        return self.__price*self.quantity

    @classmethod
    def read_csv(cls):
        with open('items.csv', 'r') as f:
            reader =csv.DictReader(f)
            items = list(reader)
        for item in items:
            Item(name=item.get('name'), price = float(item.get('price')), quantity=int(item.get('quantity')))

    @property
    def price(self):
        return self.__price

    def apply_increment(self, increment_value):
        self.__price = self.__price+self.__price*increment_value

    def apply_discount(self):
        self.__price=self.__price*self.pay_rate


    def __str__(self) -> str:
        rep = f"Item\n name: {self.name}\n price: {self.__price: ,}\n quantity: {self.quantity: ,}"
        return rep

    def __repr__(self) -> str:
        rep = f"{self.__class__.__name__}('{self.name}', {self.__price}, {self.quantity})"
        return rep

# %%
Item.read_csv()
# print(Item.item_lst)
# %%
print([item for item in Item.item_lst])
# %%
class Phone(Item):
    def __init__(self, name: str, price: float, quantity=0, broken_phone=0) -> None:
        super().__init__(name, price, quantity)
        assert broken_phone>=0, f"Broken phone: {broken_phone} is less than 0."
        self.broken_phone = broken_phone

# %%
phone1 = Phone('Iphone 12', 1200, 3,1)
# %%
print(phone1.__str__)
# %%
print(phone1.item_lst)
# %%
item1 = Item('Myitem', 400, 5)
# %%
item1.price
# %%
