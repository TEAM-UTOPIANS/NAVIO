Navio - Route Optimization Platform
🌐 Live Demo
🔗 Visit: https://navio.onrender.com

Navio is a comprehensive route optimization platform that helps businesses and individuals optimize their delivery routes for maximum efficiency. The platform combines advanced algorithms with a user-friendly interface to provide intelligent route planning solutions.

Features
User Authentication: Secure registration and login system with JWT tokens

Route Management: Create, view, update, and delete delivery routes

Route Optimization: Advanced algorithms including Dijkstra, A*, and Floyd-Warshall

Real-time Maps Integration: Google Maps API integration for accurate geocoding and directions

Multi-algorithm Support: Choose from different optimization algorithms based on your needs

Round-trip Optimization: Support for both one-way and round-trip route planning

Customizable Preferences: Set delivery priorities and time constraints

Project Structure
bash
Copy
Edit
/project
├── backend/                 # Flask backend application
│   ├── app.py              # Main Flask application
│   ├── models.py           # Database models (User, DeliveryRoute)
│   ├── config.py           # Configuration settings
│   ├── routeoptimize.py    # Route optimization algorithms
│   └── routes/             # Additional route handlers
├── frontend/               # Frontend application
├── requirements.txt        # Python dependencies
└── README.md               # This file
Technology Stack
Backend
Flask: Web framework for building the API

SQLAlchemy: Database ORM

JWT: Authentication and authorization

Flask-CORS: Cross-origin resource sharing

Google Maps API: Geocoding and directions

Frontend
HTML, CSS, JavaScript

Responsive design for mobile and desktop

Installation
Prerequisites
Python 3.7+

pip package manager

Setup Instructions
Clone the repository

bash
Copy
Edit
git clone <repository-url>
cd Navio
Create and activate virtual environment

bash
Copy
Edit
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set up environment variables
Create a .env file in the root directory with the following variables:

ini
Copy
Edit
SECRET_KEY=your_secret_key_here
SQLALCHEMY_DATABASE_URI=sqlite:///navio.db
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
Run the application

bash
Copy
Edit
cd backend
python app.py
Access the application
Open your browser and navigate to http://localhost:5000
Or try the hosted version: https://navio.onrender.com

API Endpoints
Authentication
POST /api/register - User registration

POST /api/login - User login

Routes Management
GET /api/routes - Get all routes for authenticated user

POST /api/routes - Create a new route

GET /api/routes/<id> - Get specific route

PUT /api/routes/<id> - Update route

DELETE /api/routes/<id> - Delete route

Route Optimization
POST /api/optimize/route - Optimize delivery route

System
GET /api/health - Health check endpoint

GET /api/config/maps-key - Get Google Maps API key

Route Optimization Algorithms
The platform supports multiple optimization algorithms:

Dijkstra's Algorithm: Finds shortest path between nodes

A* Algorithm: Informed search algorithm with heuristics

Floyd-Warshall: All-pairs shortest path algorithm

Usage
Register/Login: Create an account or log in to access the platform

Create Routes: Add delivery locations and set preferences

Optimize Routes: Choose an algorithm and optimize your route

View Results: See optimized routes with distance and time estimates

Manage Routes: Edit, delete, or duplicate existing routes

Configuration
Environment Variables
SECRET_KEY: Secret key for JWT token generation

SQLALCHEMY_DATABASE_URI: Database connection string

GOOGLE_MAPS_API_KEY: Google Maps API key for geocoding and directions

Database
The application uses SQLite by default. For production, consider using PostgreSQL or MySQL.

Deployment
Local Development
bash
Copy
Edit
cd backend
python app.py
Production Deployment
Deployed live on Render: https://navio.onrender.com

For advanced production setups, consider using:

Gunicorn: WSGI server

Nginx: Reverse proxy

PostgreSQL: Production database

Docker: Containerization

Troubleshooting
Common Issues
Module Import Errors: Ensure all dependencies are installed

Database Errors: Check database URI and permissions

Maps API Errors: Verify Google Maps API key is valid

CORS Issues: Check CORS configuration for frontend integration

File Path Issues
If you encounter the error python: can't open file '/opt/render/project/src/app.py', this indicates a deployment configuration mismatch. The application expects the file structure to match your deployment platform's requirements.

Contributing
Fork the repository

Create a feature branch

Make your changes

Test thoroughly

Submit a pull request

License
This project is licensed under the MIT License - see the LICENSE file for details.

Support
For support and questions, please contact the development team or create an issue in the repository.

Navio – Optimizing routes, one delivery at a time. 🚚✨
🔗 Live Project Link
