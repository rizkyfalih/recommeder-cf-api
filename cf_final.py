import numpy as np
import pandas as pd 
from numpy import nan
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from correlation_pearson.code import CorrelationPearson
pearson = CorrelationPearson()
import operator

from sqlalchemy import create_engine
import pymysql
import mysql.connector

# Database config
mydb = mysql.connector.connect(
  host="103.129.222.66",
  port= 3306,
  user="mylearn1_mylearn1",
  password="W7e3l7:5zK!dOF",
  database="mylearn1_mylearning"
)

mycursor = mydb.cursor()


db_connection_str = 'mysql+pymysql://mylearn1_mylearn1:W7e3l7:5zK!dOF@103.129.222.66:3306/mylearn1_mylearning'
db_connection = create_engine(db_connection_str)

sql = """SELECT us.user_id, us.content_id, c.title, c.content_img, c.description, r.rating, b.bookmarked, t.timespent, us.total_selection 
FROM user_selection us 
LEFT OUTER JOIN ratings r ON r.user_id = us.user_id AND r.content_id = us.content_id
LEFT OUTER JOIN bookmarks b ON b.user_id = us.user_id AND b.content_id = us.content_id
LEFT OUTER JOIN timespents t ON t.user_id = us.user_id AND t.content_id = us.content_id
LEFT OUTER JOIN contents c ON c.id = us.content_id"""

def recommendation(id): 
    # Import data
    #raw = pd.read_csv('WLO_raw(1).csv', delimiter=';')
    raw = pd.read_sql(sql, con=db_connection)
    raw['user_id'] = raw['user_id'].astype(np.int64)
    raw['content_id'] = raw['content_id'].astype(np.int64)
    raw['rating'] = raw['rating'].astype(np.float64)
    raw['bookmarked']= raw['bookmarked'].astype(np.float64)
    raw['timespent'] = raw['timespent'].astype(np.float64)
    raw['total_selection'] = raw['total_selection'].astype(np.int64)

    # Cleaning/preprocessing
    # raw = raw[pd.notnull(raw['rating'])]
    # raw = raw[pd.notnull(raw['timespent'])]
    # where_are_NaNs = np.isnan(raw)
    # raw[where_are_NaNs] = 0
    raw.fillna(value=0, inplace=True)
    
    # Variables
    E = raw['rating']
    a = raw['bookmarked']
    t = raw['timespent']
    c = raw['total_selection']
    b = np.exp(-t)
    
    # Normalize
    normalize = lambda x : ((x - np.min(x))/(np.max(x)-np.min(x)))
    A = normalize(a)
    B = normalize(b)
    C = normalize(c)
    
    # Weight Learning Object
    e = E
    i = A + ((2)*(B)) + ((2)*(C)*(E))
    S = (1/2)*(e+i)
    
    user_id = raw[['user_id']]
    content_id = raw[['content_id']]
    d = {
        'A' : A, 'B' : B, 'C' : C,
        'implicit' : i, 'explicit' : e, 'S' : S
    }
    
    data = pd.DataFrame(data = d)
    
    joinraw = raw.join(data)
    weight = joinraw[['user_id', 'content_id', 'S']]
    
    # LLOR matrix
    llor = weight.pivot_table(index='user_id', columns='content_id', values='S').fillna(0)
    llor_matrix = csr_matrix(llor.values)
    llor_array = llor.values
    
    # Similarities
    dict_x={}
    for i in range(len(llor_array)):
        dict_x[i]={}
        for j in range(len(llor_array)):
            if i==j:
                continue
            else:
                dict_x[i][j]= pearson.result(llor_array[i],llor_array[j])
    
    # Prediction
    dict_x={}
    final_score=[]
    final_seq=[]
    k=10
    for i,value_i in enumerate(list(llor.index)):
    #    print("=========INI USER ID: ",value_i,"=================")
        dict_x[i]={}
        temp={}
        for j,value_j in enumerate(list(llor.index)):
            if i==j:
                continue
            else:
                temp[j]= pearson.result(llor_array[i],llor_array[j])
        tmp = {key: temp[key] if not np.isnan(temp[key]) else 0 for key in temp}
        tmp = dict(sorted(tmp.items(), key=operator.itemgetter(1),reverse=True)[:k])
        pearsonDF = pd.DataFrame.from_dict(tmp, orient='index')
        pearsonDF.columns = ['similarityIndex']
        pearsonDF['user_id'] = pearsonDF.index
        pearsonDF.index = range(len(pearsonDF))
        mean_rating = [llor_array[y].mean() for y in list(pearsonDF['user_id'])]
        pearsonDF['ave_rating'] = mean_rating
        topUsersRating=pearsonDF.merge(weight, left_on='user_id', right_on='user_id', how='inner')
        topUsersRating['weight'] = topUsersRating['S'] - topUsersRating['ave_rating']
        topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['weight']
        tempTopUsersRating = topUsersRating.groupby('content_id').sum()[['similarityIndex','weightedRating']]
        tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
        recommendation_df = pd.DataFrame()
        recommendation_df['recommendation score'] = llor_array[i].mean()+(tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex'])
        recommendation_df['content_id'] = tempTopUsersRating.index
        recommendation_df = recommendation_df.sort_values(by='recommendation score', ascending=False)
        for index, row in recommendation_df.iterrows():
            final_score.append([value_i,row['content_id'],row['recommendation score']])
        final_seq.append([value_i,list(recommendation_df["content_id"])])
    #    print(recommendation_df)
    
    # Recommendation result table
    final_score_df = pd.DataFrame(final_score,columns=["user_id","content_id","Recommendation Score"])
    
    # LLOP matrix
    llop = final_score_df.pivot_table(index='user_id', columns='content_id', values='Recommendation Score').fillna(0)
    
    # Data Final
    final_seq_df = pd.DataFrame(final_seq,columns=["user_id","Recommendation Sequence"])
    #print(final_seq_df)
    return final_seq_df["Recommendation Sequence"][final_seq_df.index[final_seq_df['user_id'] == id]].item()

def get_content(id):
    list_id = recommendation(id)
    list_content = []
    for i in range(len(list_id)):
        query = """SELECT * FROM contents WHERE id= {} """.format(list_id[i])
        content_data = mycursor.execute(query)
        content_data = mycursor.fetchall()
        content ={
            "id": str(content_data[0][0]),
            "title": str(content_data[0][2]),
            "content_img": str(content_data[0][3]),
            "description": str(content_data[0][4]),
#            "category": content_data[0][5],
#            "tag": content_data[0][6],
#            "body": content_data[0][7],
#            "file": content_data[0][8],
#            "video": content_data[0][9],
            "created_at": str(content_data[0][10]),
#            "updated_at": content_data[0][11],
            "rating": str(content_data[0][12])
#            "total_selection": content_data[0][13],
        }
        list_content.append(content)
        
    return list_content

# a = get_content(28)

# query = """SELECT * FROM contents WHERE id= {} """.format(21)
# content_data = mycursor.execute(query)
# content_data = mycursor.fetchall()