# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:56:34 2019

@author: ACER
"""

import json, falcon
from waitress import serve
from cf_final import recommendation
from cf_final import get_content

class Recommender():
    def on_post(self, req, res):
        data = json.loads(req.stream.read())
        id = data['id_user']
        
        prediction = get_content(id)
        print(prediction)
        output = {
            'recommendation': prediction
        }
        res.body = json.dumps(output)

app = api = falcon.API()
api.add_route('/recommend', Recommender())

#serve(app, host="127.0.0.1", port=8001)
