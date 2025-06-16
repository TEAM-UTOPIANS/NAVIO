from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    company = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    routes = db.relationship('DeliveryRoute', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'fullName': self.full_name,
            'company': self.company,
            'createdAt': self.created_at.isoformat()
        }

class DeliveryRoute(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    start_location = db.Column(db.String(200), nullable=False)
    end_location = db.Column(db.String(200), nullable=False)
    waypoints = db.Column(db.JSON)  # Store waypoints as JSON
    total_distance = db.Column(db.Float)
    estimated_time = db.Column(db.Float)  # in minutes
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='pending')  # pending, in_progress, completed

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'startLocation': self.start_location,
            'endLocation': self.end_location,
            'waypoints': self.waypoints,
            'totalDistance': self.total_distance,
            'estimatedTime': self.estimated_time,
            'createdAt': self.created_at.isoformat(),
            'completedAt': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status
        } 