// Initialize map
let map = L.map('map').setView([20.5937, 78.9629], 5); // Centered on India
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Store locations and markers
let locations = [];
let markers = [];
let routeLine = null;
let currentLocationMarker = null;
let watchId = null;
let currentRoute = null;
let turnByTurnDirections = [];
let lastKnownPosition = null;
let routeUpdateInterval = null;

// Initialize GPS tracking
function initGPS() {
    if ("geolocation" in navigator) {
        // Get initial position
        navigator.geolocation.getCurrentPosition(
            position => {
                const { latitude, longitude } = position.coords;
                updateCurrentLocation(latitude, longitude);
                
                // Start watching position
                watchId = navigator.geolocation.watchPosition(
                    position => {
                        const { latitude, longitude } = position.coords;
                        updateCurrentLocation(latitude, longitude);
                    },
                    error => {
                        console.error('Error getting location:', error);
                        alert('Error getting your location. Please check your GPS settings.');
                    },
                    {
                        enableHighAccuracy: true,
                        maximumAge: 0,
                        timeout: 5000
                    }
                );
            },
            error => {
                console.error('Error getting location:', error);
                alert('Error getting your location. Please check your GPS settings.');
            }
        );
    } else {
        alert('Geolocation is not supported by your browser');
    }
}

// Update current location marker
function updateCurrentLocation(lat, lng) {
    lastKnownPosition = { lat, lng };
    
    if (currentLocationMarker) {
        currentLocationMarker.setLatLng([lat, lng]);
    } else {
        const currentLocationIcon = L.divIcon({
            className: 'current-location-marker',
            html: '<div class="pulse-marker"></div>',
            iconSize: [20, 20],
            iconAnchor: [10, 10]
        });

        currentLocationMarker = L.marker([lat, lng], {
            icon: currentLocationIcon,
            zIndexOffset: 1000
        }).addTo(map);
    }
    
    // Update distances and ETA for all stops
    updateDistancesToStops();
    
    // If we have a route, update it
    if (currentRoute) {
        updateRouteWithCurrentLocation();
    }
}

// Add current location to route
function addCurrentLocation() {
    if (currentLocationMarker) {
        const latlng = currentLocationMarker.getLatLng();
        const location = {
            id: locations.length + 1,
            name: 'Current Location',
            lat: latlng.lat,
            lng: latlng.lng
        };
        addLocation(location);
    } else {
        alert('Waiting for GPS signal...');
    }
}

// Stop GPS tracking
function stopGPS() {
    if (watchId) {
        navigator.geolocation.clearWatch(watchId);
        watchId = null;
    }
    if (currentLocationMarker) {
        map.removeLayer(currentLocationMarker);
        currentLocationMarker = null;
    }
}

// Initialize GPS when the page loads
initGPS();

// Add click handler to map
map.on('click', async function(e) {
    const { lat, lng } = e.latlng;
    
    try {
        // Reverse geocode to get location name
        const response = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}`);
        const data = await response.json();
        
        const location = {
            id: locations.length + 1,
            name: data.display_name,
            lat: lat,
            lng: lng
        };
        
        addLocation(location);
    } catch (error) {
        console.error('Error getting location name:', error);
        // If geocoding fails, use coordinates as name
        const location = {
            id: locations.length + 1,
            name: `${lat.toFixed(4)}, ${lng.toFixed(4)}`,
            lat: lat,
            lng: lng
        };
        addLocation(location);
    }
});

// Search functionality
async function searchLocation() {
    const searchInput = document.getElementById('searchInput');
    const searchResults = document.getElementById('searchResults');
    const query = searchInput.value.trim();
    
    if (!query) {
        return;
    }
    
    try {
        const response = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        // Clear previous results
        searchResults.innerHTML = '';
        
        if (data.length === 0) {
            searchResults.innerHTML = '<div class="search-result-item">No results found</div>';
            searchResults.style.display = 'block';
            return;
        }
        
        // Display results
        data.forEach(result => {
            const div = document.createElement('div');
            div.className = 'search-result-item';
            div.textContent = result.display_name;
            div.onclick = () => {
                const location = {
                    id: locations.length + 1,
                    name: result.display_name,
                    lat: parseFloat(result.lat),
                    lng: parseFloat(result.lon)
                };
                addLocation(location);
                searchResults.style.display = 'none';
                searchInput.value = '';
                
                // Center map on the selected location
                map.setView([location.lat, location.lng], 13);
            };
            searchResults.appendChild(div);
        });
        
        searchResults.style.display = 'block';
    } catch (error) {
        console.error('Error searching location:', error);
        searchResults.innerHTML = '<div class="search-result-item">Error searching location</div>';
        searchResults.style.display = 'block';
    }
}

// Close search results when clicking outside
document.addEventListener('click', function(e) {
    const searchResults = document.getElementById('searchResults');
    const searchContainer = document.querySelector('.search-container');
    
    if (!searchContainer.contains(e.target)) {
        searchResults.style.display = 'none';
    }
});

// Handle Enter key in search input
document.getElementById('searchInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        searchLocation();
    }
});

function addLocation(location) {
    locations.push(location);
    
    // Add marker to map
    const marker = L.marker([location.lat, location.lng])
        .bindPopup(location.name)
        .addTo(map);
    markers.push(marker);
    
    // Update location list
    updateLocationList();
}

function updateLocationList() {
    const locationList = document.getElementById('locationList');
    locationList.innerHTML = '';
    
    locations.forEach((location, index) => {
        const li = document.createElement('li');
        li.className = 'list-group-item';
        li.innerHTML = `
            <div>
                <strong>${location.name}</strong>
                <div class="location-info">${location.lat.toFixed(4)}, ${location.lng.toFixed(4)}</div>
            </div>
            <span class="btn-remove" onclick="removeLocation(${index})">×</span>
        `;
        locationList.appendChild(li);
    });
}

function removeLocation(index) {
    locations.splice(index, 1);
    map.removeLayer(markers[index]);
    markers.splice(index, 1);
    updateLocationList();
}

function clearMap() {
    locations = [];
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];
    if (routeLine) {
        map.removeLayer(routeLine);
        routeLine = null;
    }
    updateLocationList();
    document.getElementById('routeOutput').textContent = '';
}

async function optimizeRoute() {
    if (locations.length < 2) {
        alert('Please add at least 2 locations to optimize the route');
        return;
    }

    // Create graph edges based on straight-line distances
    const edges = [];
    for (let i = 0; i < locations.length; i++) {
        for (let j = i + 1; j < locations.length; j++) {
            const distance = calculateDistance(
                locations[i].lat, locations[i].lng,
                locations[j].lat, locations[j].lng
            );
            edges.push({
                from: locations[i].id,
                to: locations[j].id,
                weight: distance
            });
        }
    }

    try {
        // Get optimized route using Dijkstra's algorithm
        const response = await fetch('/api/optimize/dijkstra', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                graph: edges,
                startNode: locations[0].id
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        displayRoute(result);
    } catch (error) {
        console.error('Error optimizing route:', error);
        alert('Error optimizing route. Please try again.');
    }
}

function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Earth's radius in km
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * 
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
}

function toRad(degrees) {
    return degrees * (Math.PI/180);
}

function calculateETA(distance, averageSpeed = 50) { // 50 km/h default speed
    const timeInHours = distance / averageSpeed;
    const now = new Date();
    const eta = new Date(now.getTime() + timeInHours * 60 * 60 * 1000);
    return eta;
}

function updateDistancesToStops() {
    if (!lastKnownPosition) return;
    
    locations.forEach((location, index) => {
        const distance = calculateDistance(
            lastKnownPosition.lat,
            lastKnownPosition.lng,
            location.lat,
            location.lng
        );
        
        // Update marker popup with distance
        const popupContent = `
            <strong>${location.name}</strong><br>
            Distance: ${distance.toFixed(2)} km<br>
            ETA: ${calculateETA(distance).toLocaleTimeString()}
        `;
        markers[index].setPopupContent(popupContent);
    });
}

function generateTurnByTurnDirections(route) {
    turnByTurnDirections = [];
    for (let i = 0; i < route.length - 1; i++) {
        const current = route[i];
        const next = route[i + 1];
        
        // Calculate bearing between points
        const bearing = calculateBearing(current.lat, current.lng, next.lat, next.lng);
        const distance = calculateDistance(current.lat, current.lng, next.lat, next.lng);
        
        // Determine turn direction based on bearing
        let direction = '';
        if (bearing >= 337.5 || bearing < 22.5) direction = 'Continue straight';
        else if (bearing >= 22.5 && bearing < 67.5) direction = 'Turn slightly right';
        else if (bearing >= 67.5 && bearing < 112.5) direction = 'Turn right';
        else if (bearing >= 112.5 && bearing < 157.5) direction = 'Turn sharply right';
        else if (bearing >= 157.5 && bearing < 202.5) direction = 'Turn around';
        else if (bearing >= 202.5 && bearing < 247.5) direction = 'Turn sharply left';
        else if (bearing >= 247.5 && bearing < 292.5) direction = 'Turn left';
        else if (bearing >= 292.5 && bearing < 337.5) direction = 'Turn slightly left';
        
        turnByTurnDirections.push({
            instruction: `${direction} towards ${next.name}`,
            distance: distance.toFixed(2),
            eta: calculateETA(distance).toLocaleTimeString()
        });
    }
    
    updateDirectionsDisplay();
}

function calculateBearing(lat1, lon1, lat2, lon2) {
    const φ1 = toRad(lat1);
    const φ2 = toRad(lat2);
    const λ1 = toRad(lon1);
    const λ2 = toRad(lon2);
    
    const y = Math.sin(λ2 - λ1) * Math.cos(φ2);
    const x = Math.cos(φ1) * Math.sin(φ2) -
              Math.sin(φ1) * Math.cos(φ2) * Math.cos(λ2 - λ1);
    const θ = Math.atan2(y, x);
    
    return (toDegrees(θ) + 360) % 360;
}

function toDegrees(radians) {
    return radians * (180/Math.PI);
}

function updateDirectionsDisplay() {
    const directionsContainer = document.getElementById('directionsContainer');
    if (!directionsContainer) return;
    
    let html = '<h4>Turn-by-Turn Directions</h4><div class="directions-list">';
    turnByTurnDirections.forEach((step, index) => {
        html += `
            <div class="direction-step ${index === 0 ? 'active' : ''}">
                <div class="step-number">${index + 1}</div>
                <div class="step-content">
                    <div class="step-instruction">${step.instruction}</div>
                    <div class="step-details">
                        <span class="distance">${step.distance} km</span>
                        <span class="eta">ETA: ${step.eta}</span>
                    </div>
                </div>
            </div>
        `;
    });
    html += '</div>';
    directionsContainer.innerHTML = html;
}

function updateRouteWithCurrentLocation() {
    if (!lastKnownPosition || !currentRoute) return;
    
    // Find the closest point on the route to current location
    let closestPoint = null;
    let minDistance = Infinity;
    
    for (let i = 0; i < currentRoute.length - 1; i++) {
        const distance = calculateDistance(
            lastKnownPosition.lat,
            lastKnownPosition.lng,
            currentRoute[i].lat,
            currentRoute[i].lng
        );
        
        if (distance < minDistance) {
            minDistance = distance;
            closestPoint = i;
        }
    }
    
    // Update turn-by-turn directions starting from current position
    if (closestPoint !== null) {
        const updatedRoute = currentRoute.slice(closestPoint);
        generateTurnByTurnDirections(updatedRoute);
    }
}

function displayRoute(result) {
    // Clear previous route
    if (routeLine) {
        map.removeLayer(routeLine);
    }

    // Create route points
    const routePoints = [];
    let current = locations[0].id;
    
    while (current) {
        const location = locations.find(loc => loc.id === current);
        routePoints.push([location.lat, location.lng]);
        current = result.previous[current];
    }

    // Store current route for updates
    currentRoute = routePoints.map((point, index) => ({
        lat: point[0],
        lng: point[1],
        name: locations[index].name
    }));

    // Draw route on map
    routeLine = L.polyline(routePoints, {
        color: '#007bff',
        weight: 3,
        dashArray: '5, 5'
    }).addTo(map);

    // Generate turn-by-turn directions
    generateTurnByTurnDirections(currentRoute);

    // Start real-time route updates
    if (routeUpdateInterval) {
        clearInterval(routeUpdateInterval);
    }
    routeUpdateInterval = setInterval(updateRouteWithCurrentLocation, 5000);

    // Display route information
    const output = document.getElementById('routeOutput');
    let text = 'Optimized Route:\n\n';
    
    let totalDistance = 0;
    let currentId = locations[0].id;
    
    while (currentId) {
        const location = locations.find(loc => loc.id === currentId);
        text += `${location.name}\n`;
        
        if (result.previous[currentId]) {
            const prevLocation = locations.find(loc => loc.id === result.previous[currentId]);
            const distance = calculateDistance(
                location.lat, location.lng,
                prevLocation.lat, prevLocation.lng
            );
            totalDistance += distance;
        }
        
        currentId = result.previous[currentId];
    }
    
    text += `\nTotal Distance: ${totalDistance.toFixed(2)} km`;
    text += `\nEstimated Total Time: ${calculateETA(totalDistance).toLocaleTimeString()}`;
    output.textContent = text;

    const fuelCostElem = document.getElementById('fuelCost');
    if (fuelCostElem) {
        fuelCostElem.textContent = fuelCost.toFixed(2);
    }
} 