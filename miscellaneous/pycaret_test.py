# %%
import pandas as pd
from pycaret.classification import *

# %%
df = pd.read_csv("HeartDiseaseTrain-Test.csv")
cat_features = list(df.dtypes[df.dtypes == "object"].index)
# %%
experiment = setup(df, target="target", categorical_features=cat_features)
# %%
best_model = compare_models()
# %%
predict_model(best_model)
# %%
save_model(best_model, model_name="extra_tree")

# %%
import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv("laptop_training.csv")
fig = px.scatter(
    df,
    x="chargetime",
    y="usetime",
)
fig.show()
# %%
from pyearth import Earth
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["chargetime"], df["usetime"], test_size=0.2, random_state=42
)

# Create and fit the MARS model
model = Earth()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the results
plt.scatter(X_test, y_test, color="black", label="data")
plt.plot(X_test, y_pred, color="red", linewidth=3, label="MARS model")
plt.title("MARS Regression")
plt.legend()
plt.show()
# %%
import pandas as pd
import numpy as np

stocks = pd.read_csv(
    "https://s3.amazonaws.com/hr-testcases-us-east-1/329/assets/bigger_data_set.txt",
    header=None,
)

stock_dict = {
    stock.split(" ")[0]: [float(val) for val in stock.split(" ")[1:]]
    for stock in stocks[0].to_list()
}

stocks_df = pd.DataFrame(stock_dict)


def buying_pow(available_stocks, m):
    buying_power = dict(
        sorted(
            {
                k: (m // v)
                for k, v in available_stocks.to_dict().items()
                if (m // v) != 0
            }.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    return buying_power


# we have $100
def buy_stocks(available_stocks, m):
    buying_power = buying_pow(available_stocks, m)
    bought_stocks = {k: 0 for k in buying_power.keys()}
    money_left = m
    while money_left > 0:
        for k in buying_power.keys():
            if money_left - available_stocks[k] > 0:
                bought_stocks[k] += 1
                money_left -= available_stocks[k]
        buying_power = buying_pow(available_stocks, money_left)
        if not buying_power:
            break
    return bought_stocks, money_left


# %%
def stock_pred(m, k, d):
    for i in range(1, d + 1):
        if i == 1:
            available_stocks = stocks_df.iloc[i].sort_values()
            bought_stocks, money = buy_stocks(available_stocks, m)
        elif i == 5:
            # so basic logic here is every 5 days we look to sell our stocks depending on how it has been trending if the  price of a stock on the 5th day is above what it was bought for we sell it all and claim our profit. if it's not hold it for another 5 days if the overall slope in the last 5 days is positive and in the next 5 days if the over slope is negative we sell
            pass


stock_pred(60, 10, 3)
# %%
bought_stocks, money = buy_stocks(stocks_df.iloc[0], 60)


# %%
# so we buy on the first day check to see if we have enough to buy on the second
#
#
class Dogs:
    def __init__(self, name, age, weight):
        self.name = name
        self.age = age
        self.weight = weight

    def bark(self):
        print(f"{self.name} speak!")

    def eat(self):
        if self.weight < 15:
            print(f"{self.name} you are underweight please eat!")
        else:
            print(f"eat up!")

    def __str__(self) -> str:
        return f"This dog's name is {self.name}, it is {self.age} years old and weighs {self.weight} lbs."


roxie = Dogs("Roxie", 5, 13)

roxie.bark()
roxie.eat()

float()
# %%
import os

default_home = os.path.join(os.getcwd(), ".cache")

hf_cache_home = os.path.join(default_home, "huggingface")

os.environ["HF_HOME"] = hf_cache_home
# default_cache_path = os.path.join(hf_cache_home, "hub")
# default_assets_cache_path = os.path.join(hf_cache_home, "assets")
# default_token_cache_path = os.path.join(hf_cache_home, "token")
# os.environ["HF_HUB_PATH"] = default_cache_path
# os.environ["HF_ASSETS_PATH"] = default_assets_cache_path
# os.environ["HF_TOKEN_PATH"] = default_token_cache_path
# HF_HUB_PATH = os.getenv("HF_HUB_PATH")
# HF_ASSETS_PATH = os.getenv("HF_ASSETS_PATH")
# HF_TOKEN_PATH = os.getenv("HF_TOKEN_PATH")

access_token = "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"
from huggingface_hub import login

login(access_token)

from transformers import HfAgent


# %%
# Starcoder
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
# StarcoderBase
# agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoderbase")
# OpenAssistant
# agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")

tmp = agent.run("Draw me a picture of rivers and lakes.")

# %%
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")

llm.invoke("how can langsmith help with testing?")
# %%
import string


class Solution:
    def convertToTitle(self, columnNumber: int) -> str:
        if columnNumber < 1:
            return "Error: try a number greater than 0."
        letters = list(string.ascii_uppercase)
        if columnNumber <= 26:
            return letters[columnNumber - 1]
        elif columnNumber > 26:
            new_index = columnNumber - 26
            # for i in letters:
            #     for j in letters:


sol = Solution()
sol.convertToTitle(27)
# %%
