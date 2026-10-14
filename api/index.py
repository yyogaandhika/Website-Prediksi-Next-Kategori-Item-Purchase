import sys
import os

# Menambahkan folder backend ke path eksekusi Python
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from main import app
