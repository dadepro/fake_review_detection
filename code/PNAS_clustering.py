import pandas as pd
import networkx as nx
import scipy

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

################################# DATASETS
# reviews
reviews = pd.read_csv('UCSD_home_and_kitchen_reviews.csv.gz')

print("Number of products, reviews, and reviewers in reviews dataset:", \
				len(reviews.product_ID.unique()),\
				reviews.shape[0],\
				len(reviews.reviewer_ID.unique()))

# UCSD product level data
df_ucsd = pd.read_csv('UCSD_product_level_data.csv.gz')

# our data
df_ours = pd.read_csv('product_level_data_without_img_feats.csv.gz')

############################ FUNCTIONS

def scaling_data(df, features):
	scaler = StandardScaler()
	X = scaler.fit_transform(df[features])
	return X

def weighted_projected_graph(B, nodes, ratio=False):
    if B.is_directed():
        pred = B.pred
        G = nx.DiGraph()
    else:
        pred = B.adj
        G = nx.Graph()
    G.graph.update(B.graph)
    G.add_nodes_from((n, B.nodes[n]) for n in nodes)
    n_top = float(len(B) - len(nodes))
    nodes_checked = []
    for u in nodes:
        nodes_checked.append(u)
        unbrs = set(B[u])
        nbrs2 = {n for nbr in unbrs for n in B[nbr]} - set(nodes_checked)
        for v in nbrs2:
            vnbrs = set(pred[v])
            common = unbrs & vnbrs
            if not ratio:
                weight = len(common)
            else:
                weight = len(common) / n_top
            G.add_edge(u, v, weight=weight)
    return G

def obtain_network_features(reviews):

	# initializing the product-level data
	df = pd.DataFrame({"product_ID": reviews.product_ID.unique()})

	# building the bipartite product-reviewer graph
	B = nx.Graph()
	B.add_nodes_from(reviews.reviewer_ID, bipartite=0)
	B.add_nodes_from(reviews.product_ID, bipartite=1)
	B.add_edges_from([(row['reviewer_ID'], row['product_ID']) for idx, row in reviews.iterrows()])

	# building the product projected graph
	P = weighted_projected_graph(B, reviews.product_ID.unique())

	w_degree_cent = nx.degree(P, weight='weight')
	eig_cent = nx.eigenvector_centrality(P, max_iter=500)
	pr = nx.pagerank(P, alpha=0.85)
	cc = nx.clustering(P)

	# creating the features data
	df['pagerank'] = [pr[i] for i in df.product_ID]
	df['eigenvector_cent'] = [eig_cent[i] for i in df.product_ID]
	df['clustering_coef'] = [cc[i] for i in df.product_ID]
	df['w_degree'] = [w_degree_cent[i] for i in df.product_ID]

	return df

def classification_results(df_train, df_test, features):

	X_train = df_train[features].values
	y_train = df_train['fake'].values
	X_test = df_test[features].values

	# scaler = StandardScaler()
	# X_train = scaler.fit_transform(X_train)
	# X_test = scaler.transform(X_test)
	print("Shape of train and test:",X_train.shape, X_test.shape)

	model = RandomForestClassifier(random_state=42, 
	                               n_estimators=1200,
	                               min_samples_leaf=3,
	                               min_samples_split=6,
	                               max_features='auto',
	                               max_depth=40,
	                               bootstrap=True,
	                               n_jobs=-1)
	model.fit(X_train, y_train)
	y_prob_pred = model.predict_proba(X_test)[:,1]
	print(sum(y_prob_pred >= 0.5), sum(y_prob_pred >= 0.6), sum(y_prob_pred >= 0.7))

	df_test['p_fake'] = y_prob_pred
	return df_test

################################## CLUSTERING
review_features = ['tfidf_review_body', 'n_of_reviews','avg_review_rating',
                   'avg_days_between_reviews', 'stdev_days_between_reviews',
                   'max_days_between_reviews', 'min_days_between_reviews', 
                   'share_helpful_reviews', 'share_1star', 'share_5star', 'share_photo', 'std_review_len']
network_features = ['pagerank', 'w_degree', 'clustering_coef', 'eigenvector_cent']

features_to_use = review_features + network_features

X = scaling_data(df_ucsd, features_to_use)
k = 20
method = KMeans(n_clusters=k, random_state=42).fit(X)
labels = method.labels_
df_ucsd['cluster_ID'] = labels + 1
print(df_ucsd.groupby('cluster_ID')['product_ID'].count())

################################# CLASSIFICATION ON CLUSTERS
frames = []
for i in range(k):

	print("================ CLUSTER {}====================".format(i+1))
	# obtain the network features
	df_network = obtain_network_features(reviews.loc[reviews.product_ID.isin(df_ucsd.loc[df_ucsd.cluster_ID == i+1,'product_ID'].values), :])

	# obtain all features
	df = df_network[['product_ID'] + network_features].merge(df_ucsd[review_features+['product_ID']], on='product_ID', how='inner')

	# classify
	df_with_p_fake = classification_results(df_ours, df, features=features_to_use)

	# append the data
	frames.append(df_with_p_fake)

# combining all clusters in one df
clusters = pd.concat(frames, axis=0, ignore_index=True)
clusters = clusters.merge(df_ucsd[['product_ID', 'cluster_ID']], on='product_ID', how='inner')

################################ RESULTS

clusters_pt = clusters.pivot_table(index='cluster_ID', aggfunc={'clustering_coef': 'mean',
																'eigenvector_cent': 'mean',
																'share_photo': 'mean',
																'w_degree': 'mean',
																'n_of_reviews': 'mean',
																'max_days_between_reviews':'mean',
																'pagerank':'mean',
																'share_5star':'mean',
																'tfidf_review_body':'mean', 'avg_days_between_reviews':'mean', 'stdev_days_between_reviews':'mean', 'avg_review_rating':'mean', 
																'std_review_len':'mean', 'share_1star':'mean', 'share_helpful_reviews':'mean',
																'min_days_between_reviews':'mean', 'product_ID':'count', 'p_fake':lambda x:(x>=0.5).sum(),})
clusters_pt[review_features + network_features] = scipy.stats.zscore(clusters_pt[review_features + network_features])
clusters_pt = clusters_pt.reindex(['clustering_coef','eigenvector_cent',
									'share_photo','w_degree','n_of_reviews','max_days_between_reviews',
									'pagerank','share_5star','tfidf_review_body','avg_days_between_reviews',
									'stdev_days_between_reviews','avg_review_rating','std_review_len',
									'share_1star','share_helpful_reviews','min_days_between_reviews','product_ID','p_fake'], axis=1)
clusters_pt

