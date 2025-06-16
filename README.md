# 🚚 NAVIO – Route Optimization Platform

### 👨‍💻 BY  
Mayank Karki  
Nitin Kandpal  
Swarit Kumar  

Navio is a comprehensive route optimization platform that helps businesses and individuals streamline their delivery operations. The system combines advanced pathfinding algorithms with a modern interface to provide intelligent and customizable delivery planning.

---

## 🌐 Live Demo

🔗 Visit: [https://navio.onrender.com](https://navio.onrender.com)

---

## ☁️ Deployment with Render

**Navio** is deployed using [Render](https://render.com), a powerful cloud hosting service that automates builds and deployments directly from GitHub.

### 🚀 Why Render?
- Auto-deployment from GitHub
- Free-tier for testing and demos
- HTTPS, scalability, and ease of use

---

## ⚙️ Features

- ✅ **User Authentication:** Secure login/registration using JWT
- 🛣 **Route Management:** Create, update, delete routes
- 🧠 **Route Optimization:** Uses Dijkstra, A*, Floyd-Warshall
- 🗺 **Real-Time Maps:** Google Maps API integration
- 🔀 **Multi-Algorithm Support:** Select algorithms as needed
- 🔄 **Round-Trip Optimization:** Supports one-way and round-trip planning
- ⚙️ **Custom Preferences:** Set delivery time, constraints, and priorities

---

## 🧪 Technology Stack

### 🔧 Backend
- **Flask**: Web framework  
- **SQLAlchemy**: ORM  
- **JWT**: Authentication  
- **Google Maps API**: Geocoding & directions  
- **Flask-CORS**: Cross-origin support  

### 🎨 Frontend
- **HTML, CSS, JavaScript**  
- Responsive for desktop and mobile

---

## 🗂 Project Structure

Navio/
├── backend/
│ ├── app.py # Flask entry point
│ ├── models.py # SQLAlchemy models
│ ├── config.py # App config
│ ├── routeoptimize.py # Core algorithms
│ └── routes/ # API endpoints
├── frontend/ # Web interface
├── requirements.txt # Python dependencies
└── README.md # You are here!

yaml
Copy
Edit

---

## 🔧 Installation

### 📋 Prerequisites
- Python 3.7+
- pip (Python package manager)

### 🚀 Setup Steps

```bash
git clone <repository-url>
cd Navio
Create and activate a virtual environment:

bash
Copy
Edit
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Set environment variables in a .env file:

ini
Copy
Edit
SECRET_KEY=your_secret_key
SQLALCHEMY_DATABASE_URI=sqlite:///navio.db
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
Run the backend server:

bash
Copy
Edit
cd backend
python app.py
Visit http://localhost:5000
Or try the live version: https://navio.onrender.com

🔌 API Endpoints
🔐 Authentication
POST /api/register – Register new users

POST /api/login – Login and receive token

📍 Routes Management
GET /api/routes – List all routes

POST /api/routes – Create new route

GET /api/routes/<id> – Get route by ID

PUT /api/routes/<id> – Update a route

DELETE /api/routes/<id> – Delete a route

⚙️ Optimization
POST /api/optimize/route – Optimize route based on selected algorithm

🛠 System
GET /api/health – Check system status

GET /api/config/maps-key – Get Maps API key

🧠 Optimization Algorithms
Navio supports:

Dijkstra’s Algorithm: Shortest path from source to all nodes

A*: Heuristic-based best-first search

Floyd-Warshall: All-pairs shortest path for dense graphs

📌 Usage Guide
Register/Login to the platform

Create Routes by entering delivery stops

Select an Algorithm and optimize the route

View Results with maps and estimated time

Manage Routes with edit, delete, and duplication options

🧾 Configuration
Environment Variables
SECRET_KEY: JWT security

SQLALCHEMY_DATABASE_URI: DB connection string

GOOGLE_MAPS_API_KEY: Required for map & geocoding

Database
Uses SQLite by default

For production, use PostgreSQL or MySQL

🚀 Deployment Options
🔧 Local
bash
Copy
Edit
cd backend
python app.py
☁️ Production
Deployed on Render: https://navio.onrender.com

For advanced setups, consider:

Gunicorn + Nginx

PostgreSQL

Docker support

❗ Troubleshooting
Import Errors → Check virtual environment and requirements.txt

Maps API Error → Verify key and billing in Google Cloud

Database Issues → Check file permissions and URI

CORS Problems → Update allowed origins in backend config

🤝 Contributing
We welcome all contributions!
Please:

Fork the repo

Create a branch

Submit a pull request after testing your changes

📄 License
Licensed under the MIT License. See the LICENSE file for details.

💬 Support
For issues, bugs, or questions, open an issue or reach out to the developers.

Navio – Optimizing routes, one delivery at a time. 🚚✨

🔗 Live App: https://navio.onrender.com
