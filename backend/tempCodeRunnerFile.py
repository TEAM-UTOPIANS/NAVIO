from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import sys
import os
from datetime import datetime, timedelta
import jwt