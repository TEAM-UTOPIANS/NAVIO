from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import sys
import os
from datetime import datetime, timedelta
import jwt
from functools import wraps
from models import db, User, DeliveryRoute
from config import Config
from routeoptimize import RouteOptimizer
from dotenv import load_dotenv
import math
import requests
from typing import List, Dict, Tuple
import json

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, 
    static_folder='../frontend',
    template_folder='../frontend'
)
CORS(app, supports_credentials=True)

# Load configuration
app.config.from_object(Config)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'navio_secret_key_2024')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI', 'sqlite:///navio.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

# Initialize database
db.init_app(app)

# Initialize route optimizer
route_optimizer = RouteOptimizer()

# Create database tables
with app.app_context():
    db.create_all()

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({
                'status': 'error',
                'message': 'Token is missing'
            }), 401
        
        try:
            # Decode the token using the same secret key
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            
            # Check if token is expired
            exp = payload.get('exp')
            if exp and datetime.utcnow().timestamp() > exp:
                return jsonify({
                    'status': 'error',
                    'message': 'Token has expired'
                }), 401
            
            # Get user from database
            user = User.query.get(payload.get('user_id'))
            if not user:
                return jsonify({
                    'status': 'error',
                    'message': 'User not found'
                }), 401
                
            # Add user to request context
            request.user = user
            return f(user, *args, **kwargs)
            
        except jwt.ExpiredSignatureError:
            return jsonify({
                'status': 'error',
                'message': 'Token has expired'
            }), 401
        except jwt.InvalidTokenError:
            return jsonify({
                'status': 'error',
                'message': 'Invalid token'
            }), 401
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 401
            
    return decorated

# Serve static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# Serve index.html for root URL
@app.route('/')
def index():
    return render_template('index.html')

# Serve home.html for /home route
@app.route('/home')
def home():
    return render_template('home.html')

# Serve register-form.html
@app.route('/register-form')
def register_form():
    return render_template('register-form.html')

# Serve login.html
@app.route('/login')
def login():
    return render_template('login.html')

# Serve routes.html
@app.route('/routes')
def routes():
    return render_template('routes.html')

# API Routes
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.json
        
        # Check if user already exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'message': 'User already exists!'}), 400
        
        # Create new user
        user = User(
            email=data['email'],
            full_name=data['fullName'],
            company=data['company']
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({'message': 'User registered successfully!'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 400

@app.route('/api/login', methods=['POST'])
def login_api():
    try:
        data = request.json
        user = User.query.filter_by(email=data['email']).first()
        
        if not user or not user.check_password(data['password']):
            return jsonify({'message': 'Invalid credentials!'}), 401
        
        # Generate JWT token with consistent secret key
        token = jwt.encode({
            'user_id': user.id,
            'email': user.email,
            'exp': datetime.utcnow() + timedelta(days=1)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'status': 'success',
            'message': 'Login successful!',
            'token': token,
            'user': {
                'id': user.id,
                'email': user.email,
                'full_name': user.full_name,
                'company': user.company
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

# Delivery Routes API
@app.route('/api/routes', methods=['GET'])
@token_required
def get_routes(current_user):
    routes = DeliveryRoute.query.filter_by(user_id=current_user.id).order_by(DeliveryRoute.created_at.desc()).all()
    return jsonify([route.to_dict() for route in routes])

@app.route('/api/routes', methods=['POST'])
@token_required
def create_route(current_user):
    try:
        data = request.json
        route = DeliveryRoute(
            user_id=current_user.id,
            name=data['name'],
            start_location=data['startLocation'],
            end_location=data['endLocation'],
            waypoints=data.get('waypoints', []),
            total_distance=data.get('totalDistance'),
            estimated_time=data.get('estimatedTime'),
            status='pending'
        )
        
        db.session.add(route)
        db.session.commit()
        
        return jsonify(route.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 400

@app.route('/api/routes/<int:route_id>', methods=['GET'])
@token_required
def get_route(current_user, route_id):
    route = DeliveryRoute.query.filter_by(id=route_id, user_id=current_user.id).first()
    if not route:
        return jsonify({'message': 'Route not found'}), 404
    return jsonify(route.to_dict())

@app.route('/api/routes/<int:route_id>', methods=['PUT'])
@token_required
def update_route(current_user, route_id):
    try:
        route = DeliveryRoute.query.filter_by(id=route_id, user_id=current_user.id).first()
        if not route:
            return jsonify({'message': 'Route not found'}), 404
        
        data = request.json
        if 'status' in data:
            route.status = data['status']
            if data['status'] == 'completed':
                route.completed_at = datetime.utcnow()
        
        if 'name' in data:
            route.name = data['name']
        
        db.session.commit()
        return jsonify(route.to_dict())
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 400

@app.route('/api/routes/<int:route_id>', methods=['DELETE'])
@token_required
def delete_route(current_user, route_id):
    try:
        route = DeliveryRoute.query.filter_by(id=route_id, user_id=current_user.id).first()
        if not route:
            return jsonify({'message': 'Route not found'}), 404
        
        db.session.delete(route)
        db.session.commit()
        return jsonify({'message': 'Route deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 400

@app.route('/api/optimize/route', methods=['POST'])
@token_required
def optimize_route(current_user):
    try:
        data = request.get_json()
        start_location = data.get('startLocation')
        waypoints = data.get('waypoints', [])
        priority = data.get('priority', 'balanced')
        is_round_trip = data.get('isRoundTrip', False)
        preferences = data.get('preferences', {})

        if not start_location:
            return jsonify({'error': 'Start location is required'}), 400

        # Combine start location and waypoints
        locations = [start_location] + waypoints

        # Optimize route using RouteOptimizer
        result = route_optimizer.optimize_route(
            locations=locations,
            priority=priority,
            is_round_trip=is_round_trip,
            preferences=preferences
        )

        # Save route to database
        route = DeliveryRoute(
            user_id=current_user.id,
            start_location=start_location,
            waypoints=json.dumps(waypoints),
            total_distance=result['total_distance'],
            estimated_time=result['estimated_time'],
            is_round_trip=is_round_trip,
            preferences=json.dumps(preferences)
        )
        db.session.add(route)
        db.session.commit()

        return jsonify({
            'optimized_order': result['path'],
            'total_distance': result['total_distance'],
            'estimated_time': result['estimated_time'],
            'directions': result['directions'],
            'is_round_trip': is_round_trip
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/config/maps-key', methods=['GET'])
def get_maps_key():
    """Endpoint to get Google Maps API key"""
    try:
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not api_key:
            api_key = 'AIzaSyA6sNWP3otKrfEc5_YcCwwPVs_DaYZhKNc'  # Fallback to hardcoded key
            
        return jsonify({
            'apiKey': api_key,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

def haversine(coord1, coord2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    R = 6371  # Earth's radius in kilometers
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def geocode_address(address):
    """
    Convert an address to coordinates using Google Maps Geocoding API
    """
    try:
        response = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={
                "address": address,
                "key": GOOGLE_MAPS_API_KEY
            }
        )
        data = response.json()
        if data["status"] == "OK":
            loc = data["results"][0]["geometry"]["location"]
            return (loc["lat"], loc["lng"])
        return None
    except Exception as e:
        print(f"Geocoding error: {str(e)}")
        return None

def calculate_distance(loc1, loc2):
    """
    Calculate distance between two locations using either Google Maps API or Haversine formula
    """
    try:
        # First try to get coordinates for both locations
        coord1 = geocode_address(loc1)
        coord2 = geocode_address(loc2)
        
        if coord1 and coord2:
            # Use Haversine formula for direct distance
            return haversine(coord1, coord2)
        
        # Fallback to Google Maps Distance Matrix API
        url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        params = {
            'origins': loc1,
            'destinations': loc2,
            'key': GOOGLE_MAPS_API_KEY,
            'mode': 'driving'
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data['status'] != 'OK':
            raise Exception(f"Google Maps API error: {data['status']}")

        element = data['rows'][0]['elements'][0]
        if element['status'] != 'OK':
            raise Exception(f"Could not calculate distance: {element['status']}")

        return element['distance']['value'] / 1000  # Convert meters to kilometers
    except Exception as e:
        print(f"Distance calculation error: {str(e)}")
        # Fallback to a default distance if all methods fail
        return 10.0

def calculate_time(distance, preferences):
    """
    Calculate travel time using Google Maps Distance Matrix API with traffic consideration.
    """
    api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
    if not api_key:
        raise ValueError("Google Maps API key not found in environment variables")

    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        'origins': preferences.get('startLocation'),
        'destinations': preferences.get('endLocation'),
        'key': api_key,
        'mode': 'driving',
        'departure_time': 'now' if preferences.get('considerTraffic') else None,
        'traffic_model': 'best_guess' if preferences.get('considerTraffic') else None,
        'avoid': 'tolls' if preferences.get('avoidTolls') else None
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data['status'] != 'OK':
        raise Exception(f"Google Maps API error: {data['status']}")

    element = data['rows'][0]['elements'][0]
    if element['status'] != 'OK':
        raise Exception(f"Could not calculate time: {element['status']}")

    # Get duration in traffic if available, otherwise use normal duration
    duration = element.get('duration_in_traffic', element['duration'])
    return duration['value'] / 60  # Convert seconds to minutes

class Graph:
    def __init__(self):
        self.nodes = {}
    
    def add_node(self, node):
        if node not in self.nodes:
            self.nodes[node] = {}
    
    def add_edge(self, from_node, to_node, weight):
        self.add_node(from_node)
        self.add_node(to_node)
        self.nodes[from_node][to_node] = weight
        self.nodes[to_node][from_node] = weight  # For undirected graph

    def dijkstra(self, start_node):
        distances = {node: float('infinity') for node in self.nodes}
        previous = {node: None for node in self.nodes}
        distances[start_node] = 0
        unvisited = set(self.nodes.keys())

        while unvisited:
            # Find node with smallest distance
            current = min(unvisited, key=lambda node: distances[node])
            if distances[current] == float('infinity'):
                break

            unvisited.remove(current)

            # Update distances to neighbors
            for neighbor, weight in self.nodes[current].items():
                if neighbor not in unvisited:
                    continue
                
                distance = distances[current] + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current

        return {'distances': distances, 'previous': previous}

    def floyd_warshall(self):
        nodes = list(self.nodes.keys())
        n = len(nodes)
        dist = [[float('infinity')] * n for _ in range(n)]
        next_node = [[None] * n for _ in range(n)]

        # Initialize distances
        for i in range(n):
            dist[i][i] = 0
            for neighbor, weight in self.nodes[nodes[i]].items():
                j = nodes.index(neighbor)
                dist[i][j] = weight
                next_node[i][j] = j

        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]

        return {
            'distances': dist,
            'next_node': next_node,
            'nodes': nodes
        }

    def get_path(self, start_node, end_node, previous):
        """Get the path from start_node to end_node using the previous dictionary"""
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = previous.get(current)
        return list(reversed(path))

    def calculate_total_distance(self, path):
        """Calculate total distance of a path"""
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += self.nodes[path[i]][path[i + 1]]
        return total_distance

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on the earth."""
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def calculate_distance_matrix(locations: List[Tuple[float, float]]) -> List[List[float]]:
    """Calculate distance matrix between all locations."""
    n = len(locations)
    matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            distance = haversine_distance(
                locations[i][0], locations[i][1],
                locations[j][0], locations[j][1]
            )
            matrix[i][j] = matrix[j][i] = distance
    
    return matrix

def dijkstra_algorithm(graph: List[List[float]], start: int, end: int) -> Tuple[List[int], float]:
    """Dijkstra's algorithm for finding shortest path."""
    n = len(graph)
    distances = [float('inf')] * n
    distances[start] = 0
    previous = [-1] * n
    unvisited = set(range(n))
    
    while unvisited:
        current = min(unvisited, key=lambda x: distances[x])
        if current == end:
            break
            
        unvisited.remove(current)
        
        for neighbor in range(n):
            if neighbor in unvisited and graph[current][neighbor] > 0:
                new_distance = distances[current] + graph[current][neighbor]
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current
    
    # Reconstruct path
    path = []
    current = end
    while current != -1:
        path.append(current)
        current = previous[current]
    
    return path[::-1], distances[end]

def astar_algorithm(graph: List[List[float]], start: int, end: int, 
                   heuristic: List[float]) -> Tuple[List[int], float]:
    """A* algorithm for finding shortest path with heuristic."""
    n = len(graph)
    g_score = [float('inf')] * n
    g_score[start] = 0
    f_score = [float('inf')] * n
    f_score[start] = heuristic[start]
    previous = [-1] * n
    
    open_set = [(f_score[start], start)]
    closed_set = set()
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current == end:
            break
            
        if current in closed_set:
            continue
            
        closed_set.add(current)
        
        for neighbor in range(n):
            if graph[current][neighbor] > 0:
                tentative_g = g_score[current] + graph[current][neighbor]
                
                if tentative_g < g_score[neighbor]:
                    previous[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic[neighbor]
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # Reconstruct path
    path = []
    current = end
    while current != -1:
        path.append(current)
        current = previous[current]
    
    return path[::-1], g_score[end]

def optimize_route(locations: List[str], algorithm: str, priority: str, 
                  is_round_trip: bool, preferences: Dict) -> Dict:
    """Optimize route using selected algorithm and preferences."""
    try:
        # Geocode all locations
        coordinates = [geocode_address(loc) for loc in locations]
        
        # Calculate distance matrix
        distance_matrix = calculate_distance_matrix(coordinates)
        
        # Calculate heuristic for A* (straight-line distance to end)
        heuristic = [haversine_distance(lat, lon, coordinates[-1][0], coordinates[-1][1])
                    for lat, lon in coordinates]
        
        # Optimize route
        if algorithm == 'dijkstra':
            path, total_distance = dijkstra_algorithm(distance_matrix, 0, len(locations) - 1)
        else:  # A*
            path, total_distance = astar_algorithm(distance_matrix, 0, len(locations) - 1, heuristic)
        
        # Add return to start if round trip
        if is_round_trip:
            path.append(0)
            total_distance += distance_matrix[path[-2]][0]
        
        # Calculate estimated time (assuming average speed of 50 km/h)
        estimated_time = int((total_distance / 50) * 60)  # in minutes
        
        # Get directions from Google Maps
        directions = get_directions([locations[i] for i in path], preferences)
        
        return {
            'path': path,
            'total_distance': total_distance,
            'estimated_time': estimated_time,
            'directions': directions,
            'is_round_trip': is_round_trip
        }
    except Exception as e:
        raise Exception(f"Route optimization failed: {str(e)}")

def get_directions(locations: List[str], preferences: Dict) -> List[Dict]:
    """Get turn-by-turn directions from Google Maps Directions API."""
    try:
        origin = locations[0]
        destination = locations[-1]
        waypoints = [{'location': loc} for loc in locations[1:-1]]
        
        response = requests.get(
            'https://maps.googleapis.com/maps/api/directions/json',
            params={
                'origin': origin,
                'destination': destination,
                'waypoints': '|'.join([loc['location'] for loc in waypoints]),
                'mode': 'driving',
                'departure_time': 'now',
                'traffic_model': 'best_guess' if preferences.get('considerTraffic') else 'pessimistic',
                'avoid': '|'.join([
                    'tolls' if preferences.get('avoidTolls') else '',
                    'highways' if not preferences.get('useHighways') else ''
                ]).strip('|'),
                'key': GOOGLE_MAPS_API_KEY
            }
        )
        
        data = response.json()
        if data['status'] != 'OK':
            raise Exception(f"Directions API error: {data['status']}")
        
        # Extract and format directions
        directions = []
        for leg in data['routes'][0]['legs']:
            for step in leg['steps']:
                directions.append({
                    'instruction': step['html_instructions'],
                    'distance': step['distance']['value'] / 1000,  # convert to km
                    'duration': step['duration']['value'] / 60  # convert to minutes
                })
        
        return directions
    except Exception as e:
        raise Exception(f"Failed to get directions: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 