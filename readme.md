# Chest X-ray Disease Detection 

We have created a web application for ease of access and faster interpretation of Chest X-ray images by saving the model and using flask web server to render the saved model for the given input image. The server is run at the local host port 5000 for testing the system. 

The flask server renders the html page with input forms for uploading the image file. The uploaded image is saved in uploads folder in the server and the saved model is used for running inference on it using the input image after pre-processing it for exact input shape parameters. The model gives out the prediction which then posted back to the server for displaying the result prediction. 
