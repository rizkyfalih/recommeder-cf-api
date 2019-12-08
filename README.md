# Recommender CF API

## Libraries
* json
* falcon
* gunicorn

## Commands
To run the server
1. Open the terminal
2. Go to this folder directory
3. Write 'gunicorn app:app' or 'sh start.sh' on terminal

To use API on Postman
1. Open Postman
2. Select POST
3. Write 'http://IP/recommend' on url
4. Select Body
5. Select raw(json)
6. Input JSON data
7. Click send

## To change the port
```
gunicorn app:app -b IP:Port
```

## Example template for Emotion (input)
```
{
	"id_user" : 21
}
```
