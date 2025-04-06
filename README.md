# Customer Segmentation API

A FastAPI-based REST API for customer segmentation and model monitoring.

## Features

- Customer segmentation using K-means and DBSCAN models
- Model versioning and tracking
- Data drift detection
- Performance monitoring
- Audit logging
- Secure API key authentication

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `config` directory if it doesn't exist
2. Copy `config/api_config.json.example` to `config/api_config.json`
3. Update the configuration as needed

## Running the API

```bash
python src/api.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the API is running, visit `http://localhost:8000/docs` for interactive API documentation.

### Endpoints

#### GET /
- Root endpoint
- Returns API status

#### GET /models
- Returns available models
- Requires API key

#### POST /predict/{model_type}
- Makes predictions using the specified model
- Requires API key
- Request body: List of customer data dictionaries
- Returns predictions with segment assignments

#### GET /model/status/{model_type}
- Returns model status and performance metrics
- Requires API key

#### POST /monitor/drift
- Checks for data drift
- Requires API key
- Request body: Current data for comparison
- Returns drift metrics and alerts

#### GET /segments/{model_type}
- Returns customer segments for the specified model
- Requires API key
- Optional query parameter: limit (default: 100)

#### GET /audit/logs
- Returns audit logs
- Requires API key
- Optional query parameter: limit (default: 100)

## Security

The API uses API key authentication. Include the API key in the request header:
```
X-API-Key: your-api-key
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 500: Internal Server Error

## Rate Limiting

The API implements rate limiting to prevent abuse. The default limit is 60 requests per minute.

## Logging

Logs are written to the console and can be configured in `config/api_config.json`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
