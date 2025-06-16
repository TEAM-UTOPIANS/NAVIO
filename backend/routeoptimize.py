import math
import heapq
from typing import List, Dict, Tuple, Set
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class RouteOptimizer:
    def __init__(self):
        # Try to get API key from environment variable
        self.api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        
        # If not found in environment, try to get it directly
        if not self.api_key:
            self.api_key = 'AIzaSyA6sNWP3otKrfEc5_YcCwwPVs_DaYZhKNc'
            
        if not self.api_key:
            raise ValueError("Google Maps API key not found")
            
        self.cache = {}  # Cache for geocoding and distance calculations

    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points on the earth."""
        R = 6371  # Earth's radius in kilometers

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def geocode_address(self, address: str) -> Tuple[float, float]:
        """Convert an address to coordinates using Google Maps Geocoding API."""
        if address in self.cache:
            return self.cache[address]

        try:
            response = requests.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={
                    "address": address,
                    "key": self.api_key
                }
            )
            data = response.json()
            if data["status"] == "OK":
                loc = data["results"][0]["geometry"]["location"]
                self.cache[address] = (loc["lat"], loc["lng"])
                return self.cache[address]
            raise Exception(f"Geocoding failed: {data['status']}")
        except Exception as e:
            raise Exception(f"Geocoding error: {str(e)}")

    def calculate_distance_matrix(self, locations: List[Tuple[float, float]]) -> List[List[float]]:
        """Calculate distance matrix between all locations."""
        n = len(locations)
        matrix = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = self.haversine_distance(
                    locations[i][0], locations[i][1],
                    locations[j][0], locations[j][1]
                )
                matrix[i][j] = matrix[j][i] = distance
        
        return matrix

    def dijkstra_algorithm(self, graph: List[List[float]], start: int) -> Tuple[List[int], List[float]]:
        """Dijkstra's algorithm for finding shortest paths from start to all points."""
        n = len(graph)
        distances = [float('inf')] * n
        distances[start] = 0
        previous = [-1] * n
        unvisited = set(range(n))
        
        while unvisited:
            current = min(unvisited, key=lambda x: distances[x])
            if distances[current] == float('inf'):
                break
                
            unvisited.remove(current)
            
            for neighbor in range(n):
                if neighbor in unvisited and graph[current][neighbor] > 0:
                    new_distance = distances[current] + graph[current][neighbor]
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current
        
        return previous, distances

    def astar_algorithm(self, graph: List[List[float]], start: int, end: int, 
                       heuristic: List[float]) -> Tuple[List[int], float]:
        """A* algorithm for finding shortest path between two points."""
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

    def optimize_route(self, locations: List[str], priority: str, 
                      is_round_trip: bool, preferences: Dict) -> Dict:
        """Optimize route using Dijkstra for initial planning and A* for real-time navigation."""
        try:
            # Geocode all locations
            coordinates = [self.geocode_address(loc) for loc in locations]
            
            # Calculate distance matrix
            distance_matrix = self.calculate_distance_matrix(coordinates)
            
            # Step 1: Use Dijkstra's algorithm for initial planning (start-to-all-points)
            previous, distances = self.dijkstra_algorithm(distance_matrix, 0)
            
            # Construct initial path using Dijkstra's results
            initial_path = [0]  # Start with warehouse
            unvisited = set(range(1, len(locations)))
            
            while unvisited:
                # Find nearest unvisited point
                next_point = min(unvisited, key=lambda x: distances[x])
                if distances[next_point] == float('inf'):
                    break
                
                # Add to path
                initial_path.append(next_point)
                unvisited.remove(next_point)
            
            # Step 2: Use A* for real-time navigation between consecutive points
            final_path = [initial_path[0]]  # Start with the first point
            total_distance = 0
            
            # Navigate between consecutive points using A*
            for i in range(len(initial_path) - 1):
                start = initial_path[i]
                end = initial_path[i + 1]
                
                # Calculate heuristic for A* (straight-line distance to end)
                heuristic = [self.haversine_distance(lat, lon, coordinates[end][0], coordinates[end][1])
                           for lat, lon in coordinates]
                
                # Use A* for real-time navigation between these two points
                segment_path, segment_distance = self.astar_algorithm(
                    distance_matrix, start, end, heuristic
                )
                
                # Add the segment path to final path (excluding the first point to avoid duplicates)
                final_path.extend(segment_path[1:])
                total_distance += segment_distance
            
            # Add return to start if round trip
            if is_round_trip:
                # Use A* for the return trip
                heuristic = [self.haversine_distance(lat, lon, coordinates[0][0], coordinates[0][1])
                           for lat, lon in coordinates]
                return_path, return_distance = self.astar_algorithm(
                    distance_matrix, final_path[-1], 0, heuristic
                )
                final_path.extend(return_path[1:])
                total_distance += return_distance
            
            # Calculate estimated time based on priority
            if priority == 'time':
                avg_speed = 40  # km/h for time priority
            elif priority == 'distance':
                avg_speed = 50  # km/h for distance priority
            else:  # balanced
                avg_speed = 45  # km/h for balanced priority
            
            estimated_time = int((total_distance / avg_speed) * 60)  # in minutes
            
            # Get directions from Google Maps
            directions = self.get_directions([locations[i] for i in final_path], preferences)
            
            return {
                'path': final_path,
                'total_distance': total_distance,
                'estimated_time': estimated_time,
                'directions': directions,
                'is_round_trip': is_round_trip,
                'initial_plan': initial_path,  # Include the initial Dijkstra plan for reference
                'optimization_method': 'Dijkstra + A*'
            }
        except Exception as e:
            raise Exception(f"Route optimization failed: {str(e)}")

    def get_directions(self, locations: List[str], preferences: Dict) -> List[Dict]:
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
                    'key': self.api_key
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