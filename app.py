import base64
import io
from flask import Flask, request, render_template, jsonify
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


app = Flask(__name__)

loaded_model = tf.keras.models.load_model("Backend/real_estate_model.keras")

housing_data = pd.read_csv('Backend/processed.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    input_data = {}

    input_data['year'] = int(request.form.get('year', 2025))
    input_data['location'] = request.form.get('location', 'Pal')
    input_data['sqft'] = int(request.form.get('sqft', 2000))
    input_data['amenities'] = float(request.form.get('amenities', 7.5))
    input_data['real_estate_type'] = request.form.get('real_estate_type', 'House')

    areas_data = {
        "Area1": {"ranking": 1, "locations": ["Adajan", "Pal", "Vesu"]},
        "Area2": {"ranking": 2, "locations": ["Athwa", "Ghod Dod Road", "City Light"]},
        "Area3": {"ranking": 3, "locations": ["Piplod", "Varachha", "Althan"]},
        "Area4": {"ranking": 4, "locations": ["Sarthana", "Katargam", "Udhna"]},
        "Area5": {"ranking": 5, "locations": ["Sachin", "Dindoli", "Bhestan"]}
    }
    
    location = input_data['location']
    area_rank = None
    for area, details in areas_data.items():
        if location in details['locations']:
            area_rank = details['ranking']
            break
        
    year_range = housing_data['year'].max() - housing_data['year'].min()
    hf_slope = (housing_data['HF'].max() - housing_data['HF'].min()) / year_range
    input_data['HF'] = hf_slope * (input_data['year'] - housing_data['year'].min()) + housing_data['HF'].min()
    
    input_data['location'] = area_rank
    
    real_estate_type_priority = {
        "Apartment": 1,
        "House": 2,
        "Villa": 3
    }
    input_data['real_estate_type'] = real_estate_type_priority.get(input_data['real_estate_type'], 0)
    
    input_df = pd.DataFrame([input_data])
    
    input_df.fillna(0, inplace=True)  

    scaler = StandardScaler()
    scaler.fit(housing_data.drop(columns=['price']))  
    user_input_scaled = scaler.transform(input_df) 
    prediction = loaded_model.predict(user_input_scaled)

    print("Predicted Price:", prediction)

    input_layer_weights = loaded_model.layers[0].get_weights()[0]

    feature_importance = abs(input_layer_weights).sum(axis=1)

    feature_names = ['year', 'location', 'sqft', 'amenities', 'real_estate_type', 'HF']

    img_bytes = io.BytesIO()
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance Plot')
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')

    return jsonify({
        "predicted_price": prediction.tolist()[0][0], 
        "plot": img_base64,
    })

if __name__ == '__main__':
    app.run(debug=True)
