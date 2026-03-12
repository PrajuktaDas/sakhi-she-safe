import pandas as pd
import networkx as nx

# load dataset
df = pd.read_csv("data/processed/safety_scores.csv")

# create graph
G = nx.Graph()

# add cities as nodes
for index, row in df.iterrows():
    G.add_node(row["city"], risk=row["final_risk_index"])

# get unique cities
cities = df["city"].unique()

# create edges between cities
for i in range(len(cities) - 1):

    risk1 = df[df["city"] == cities[i]]["final_risk_index"].values[0]
    risk2 = df[df["city"] == cities[i+1]]["final_risk_index"].values[0]

    edge_weight = (risk1 + risk2) / 2

    G.add_edge(cities[i], cities[i+1], weight=edge_weight)


# function to find safest route
def find_safest_route(start, end):

    try:
        route = nx.shortest_path(G, source=start, target=end, weight="weight")
        return route

    except:
        return None


# quick test
if __name__ == "__main__":
    print(find_safest_route("Rajasthan_City_1", "Gujarat_City_5"))





