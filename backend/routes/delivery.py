from flask import Blueprint, jsonify, request, current_app
from datetime import datetime
import googlemaps
from config import Config

delivery_bp = Blueprint('delivery', __name__)

def get_gmaps_client():
    return googlemaps.Client(key=current_app.config['GOOGLE_MAPS_API_KEY'])

@delivery_bp.route('/api/delivery/optimize-route', methods=['POST'])
def optimize_route():
    data = request.get_json()
    delivery_points = data.get('delivery_points', [])
    
    if not delivery_points:
        return jsonify({'error': 'No delivery points provided'}), 400
    
    try:
        # Get optimized route using Google Maps Directions API
        gmaps = get_gmaps_client()
        result = gmaps.directions(
            origin=delivery_points[0],
            destination=delivery_points[-1],
            waypoints=delivery_points[1:-1] if len(delivery_points) > 2 else None,
            optimize_waypoints=True,
            mode="driving"
        )
        
        if not result:
            return jsonify({'error': 'Could not find route'}), 404
            
        # Process the route
        route = result[0]
        optimized_points = []
        
        for leg in route['legs']:
            optimized_points.append({
                'location': leg['start_location'],
                'address': leg['start_address'],
                'duration': leg['duration']['text'],
                'distance': leg['distance']['text']
            })
            
        # Add the final destination
        optimized_points.append({
            'location': route['legs'][-1]['end_location'],
            'address': route['legs'][-1]['end_address'],
            'duration': route['legs'][-1]['duration']['text'],
            'distance': route['legs'][-1]['distance']['text']
        })
        
        return jsonify({
            'optimized_route': optimized_points,
            'total_duration': route['legs'][-1]['duration']['text'],
            'total_distance': route['legs'][-1]['distance']['text']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@delivery_bp.route('/api/delivery/update-location', methods=['POST'])
def update_location():
    data = request.get_json()
    delivery_id = data.get('delivery_id')
    location = data.get('location')
    
    if not delivery_id or not location:
        return jsonify({'error': 'Missing delivery_id or location'}), 400
        
    try:
        # In a real application, you would update the delivery location in your database
        # and broadcast the update to connected clients using WebSocket
        
        return jsonify({
            'status': 'success',
            'delivery_id': delivery_id,
            'location': location,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@delivery_bp.route('/api/delivery/active-deliveries', methods=['GET'])
def get_active_deliveries():
    try:
        # In a real application, you would fetch active deliveries from your database
        # This is a sample response
        active_deliveries = [
            {
                'id': '1',
                'status': 'in_progress',
                'current_location': {'lat': 20.5937, 'lng': 78.9629},
                'destination': {'lat': 20.6037, 'lng': 78.9729},
                'eta': '15 mins',
                'customer_name': 'John Doe',
                'address': '123 Main St, City'
            }
        ]
        
        return jsonify({'active_deliveries': active_deliveries})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500 