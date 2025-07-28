# 🌐 SwingTradeAI API Documentation

## 🚀 Quick Start

### Start API Server
```bash
./run_api.sh
```

### Access Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📋 Available Endpoints

### 🏥 Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-27T12:00:00",
  "version": "1.0.0",
  "components": {
    "feature_store": "operational",
    "regime_model": "operational"
  }
}
```

### 📊 Market Data
```http
GET /data/{symbol}?days=30
```

**Parameters:**
- `symbol` (path): Stock symbol (e.g., AAPL, MSFT)
- `days` (query, optional): Number of days of data (default: 30)

**Response:**
```json
{
  "symbol": "AAPL",
  "data": [...],
  "count": 30,
  "latest_price": 150.25
}
```

### 🔧 Technical Features
```http
GET /features/{symbol}?limit=100
```

**Parameters:**
- `symbol` (path): Stock symbol
- `limit` (query, optional): Number of records (default: 100)

**Response:**
```json
{
  "symbol": "AAPL",
  "features_count": 23,
  "records_count": 100,
  "retrieval_time": 0.004,
  "data": [...]
}
```

### 🤖 Trade Predictions
```http
POST /predict/{symbol}
```

**Parameters:**
- `symbol` (path): Stock symbol

**Response:**
```json
{
  "symbol": "AAPL",
  "prediction": {
    "regime": "bull_trending",
    "confidence": 0.85,
    "recommendation": "CALL",
    "target_price": 155.00
  },
  "timestamp": "2025-01-27T12:00:00"
}
```

### 📈 Backtesting
```http
GET /backtest/{symbol}?days=252
```

**Parameters:**
- `symbol` (path): Stock symbol
- `days` (query, optional): Backtest period in days (default: 252)

**Response:**
```json
{
  "symbol": "AAPL",
  "backtest_period": "252 days",
  "status": "completed",
  "results": {
    "total_return": "15.2%",
    "sharpe_ratio": 1.45,
    "max_drawdown": "-8.3%",
    "win_rate": "68%"
  }
}
```

## 🔗 Integration Examples

### Python Client
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Get market data
data = requests.get("http://localhost:8000/data/AAPL?days=30")
print(data.json())

# Get prediction
prediction = requests.post("http://localhost:8000/predict/AAPL")
print(prediction.json())
```

### cURL Examples
```bash
# Health check
curl http://localhost:8000/health

# Market data
curl "http://localhost:8000/data/AAPL?days=30"

# Features
curl "http://localhost:8000/features/AAPL?limit=100"

# Prediction
curl -X POST http://localhost:8000/predict/AAPL

# Backtest
curl "http://localhost:8000/backtest/AAPL?days=252"
```

## 🛠️ Testing Endpoints

### Using the Dashboard
1. Start dashboard: `./run_dashboard.sh`
2. Navigate to "System Testing" page
3. Test all endpoints interactively

### Using Swagger UI
1. Start API: `./run_api.sh`
2. Open: http://localhost:8000/docs
3. Click "Try it out" on any endpoint

### Command Line Testing
```bash
# Test all endpoints
python -c "
import requests
import json

base_url = 'http://localhost:8000'

# Test health
health = requests.get(f'{base_url}/health')
print('Health:', health.json())

# Test data
data = requests.get(f'{base_url}/data/AAPL')
print('Data count:', data.json()['count'])

# Test features
features = requests.get(f'{base_url}/features/AAPL')
print('Features:', features.json()['features_count'])
"
```

## 🚨 Error Handling

### Common Errors
- **404**: Symbol not found or no data available
- **500**: Internal server error (check logs)
- **422**: Invalid parameters

### Error Response Format
```json
{
  "detail": "No data found for symbol INVALID"
}
```

## 📝 User Acceptance Criteria

✅ **User can understand and test all endpoints independently**
- Complete API documentation provided
- Interactive Swagger UI available
- Command-line examples included
- Dashboard testing interface available

✅ **Clear error reporting**
- Structured error responses
- Helpful error messages
- Status codes follow HTTP standards

✅ **Performance requirements met**
- Feature queries: <1s response time
- Health checks: <100ms response time
- Comprehensive logging for debugging
