# Install required packages:
# pip install flask requests

# ============================================
# API Gateway (gateway.py)
# ============================================
from flask import Flask, request, jsonify
import requests

gateway = Flask(__name__)

USER_SERVICE = "http://localhost:5001"
PRODUCT_SERVICE = "http://localhost:5002"

@gateway.route('/api/users/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def user_proxy(path):
    url = f"{USER_SERVICE}/users/{path}"
    resp = requests.request(request.method, url, json=request.get_json())
    return jsonify(resp.json()), resp.status_code

@gateway.route('/api/products/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def product_proxy(path):
    url = f"{PRODUCT_SERVICE}/products/{path}"
    resp = requests.request(request.method, url, json=request.get_json())
    return jsonify(resp.json()), resp.status_code

@gateway.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "gateway"})

if __name__ == '__main__':
    gateway.run(port=5000, debug=True)


# ============================================
# User Service (user_service.py)
# ============================================
from flask import Flask, request, jsonify

user_app = Flask(__name__)

users_db = {
    "1": {"id": "1", "name": "Alice", "email": "alice@example.com"},
    "2": {"id": "2", "name": "Bob", "email": "bob@example.com"}
}

@user_app.route('/users', methods=['GET'])
def get_users():
    return jsonify(list(users_db.values()))

@user_app.route('/users/<user_id>', methods=['GET'])
def get_user(user_id):
    user = users_db.get(user_id)
    if user:
        return jsonify(user)
    return jsonify({"error": "User not found"}), 404

@user_app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    user_id = str(len(users_db) + 1)
    user = {"id": user_id, "name": data['name'], "email": data['email']}
    users_db[user_id] = user
    return jsonify(user), 201

@user_app.route('/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    if user_id not in users_db:
        return jsonify({"error": "User not found"}), 404
    data = request.get_json()
    users_db[user_id].update(data)
    return jsonify(users_db[user_id])

@user_app.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    if user_id in users_db:
        del users_db[user_id]
        return jsonify({"message": "User deleted"})
    return jsonify({"error": "User not found"}), 404

@user_app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "user-service"})

if __name__ == '__main__':
    user_app.run(port=5001, debug=True)


# ============================================
# Product Service (product_service.py)
# ============================================
from flask import Flask, request, jsonify

product_app = Flask(__name__)

products_db = {
    "1": {"id": "1", "name": "Laptop", "price": 999.99, "stock": 50},
    "2": {"id": "2", "name": "Mouse", "price": 29.99, "stock": 200}
}

@product_app.route('/products', methods=['GET'])
def get_products():
    return jsonify(list(products_db.values()))

@product_app.route('/products/<product_id>', methods=['GET'])
def get_product(product_id):
    product = products_db.get(product_id)
    if product:
        return jsonify(product)
    return jsonify({"error": "Product not found"}), 404

@product_app.route('/products', methods=['POST'])
def create_product():
    data = request.get_json()
    product_id = str(len(products_db) + 1)
    product = {
        "id": product_id,
        "name": data['name'],
        "price": data['price'],
        "stock": data.get('stock', 0)
    }
    products_db[product_id] = product
    return jsonify(product), 201

@product_app.route('/products/<product_id>', methods=['PUT'])
def update_product(product_id):
    if product_id not in products_db:
        return jsonify({"error": "Product not found"}), 404
    data = request.get_json()
    products_db[product_id].update(data)
    return jsonify(products_db[product_id])

@product_app.route('/products/<product_id>', methods=['DELETE'])
def delete_product(product_id):
    if product_id in products_db:
        del products_db[product_id]
        return jsonify({"message": "Product deleted"})
    return jsonify({"error": "Product not found"}), 404

@product_app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "product-service"})

if __name__ == '__main__':
    product_app.run(port=5002, debug=True)


# ============================================
# Usage Example (test_microservices.py)
# ============================================
import requests
import json

BASE_URL = "http://localhost:5000/api"

# Test Users
print("=== Testing User Service ===")
print("\n1. Get all users:")
resp = requests.get(f"{BASE_URL}/users/")
print(json.dumps(resp.json(), indent=2))

print("\n2. Create new user:")
new_user = {"name": "Charlie", "email": "charlie@example.com"}
resp = requests.post(f"{BASE_URL}/users/", json=new_user)
print(json.dumps(resp.json(), indent=2))

# Test Products
print("\n\n=== Testing Product Service ===")
print("\n1. Get all products:")
resp = requests.get(f"{BASE_URL}/products/")
print(json.dumps(resp.json(), indent=2))

print("\n2. Get specific product:")
resp = requests.get(f"{BASE_URL}/products/1")
print(json.dumps(resp.json(), indent=2))

print("\n3. Create new product:")
new_product = {"name": "Keyboard", "price": 79.99, "stock": 100}
resp = requests.post(f"{BASE_URL}/products/", json=new_product)
print(json.dumps(resp.json(), indent=2))
