# Alzheimer Detection Backend

Python FastAPI backend for Alzheimer's disease detection using ONNX model and brain scan images.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place ONNX Model
Copy `alzheimer_coatnet.onnx` to the parent directory:
```
../alzheimer_coatnet.onnx
```

Or update the model path in `main.py` if using a different location.

### 3. Run the Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

## API Endpoints

### 1. Health Check
**GET** `/health`
- Check if server and model are running

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Get Model Info
**GET** `/info`
- Get details about the loaded model

**Response:**
```json
{
  "model_type": "ONNX CoAtNet",
  "purpose": "Alzheimer's disease detection",
  "input_shape": "(1, 3, 224, 224)",
  "status": "Ready for inference"
}
```

### 3. Predict
**POST** `/predict`
- Upload brain scan image and get prediction

**Request:**
- Content-Type: `multipart/form-data`
- Parameter: `file` (image file)

**Response:**
```json
{
  "success": true,
  "prediction": {
    "has_alzheimer": false,
    "confidence": 0.85,
    "raw_predictions": [0.15, 0.85]
  },
  "filename": "brain_scan.jpg",
  "message": "Prediction successful"
}
```

## Flutter Integration

### 1. Add http package to pubspec.yaml
```yaml
dependencies:
  http: ^1.1.0
```

### 2. Create API service in Flutter
```dart
import 'package:http/http.dart' as http;
import 'dart:io';

class AlzheimerAPIService {
  final String baseUrl = "http://your-server-ip:8000";

  Future<Map<String, dynamic>> predictAlzheimer(File imageFile) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/predict'),
      );
      
      request.files.add(
        await http.MultipartFile.fromPath('file', imageFile.path),
      );

      var response = await request.send();
      var responseData = await response.stream.bytesToString();
      
      if (response.statusCode == 200) {
        return jsonDecode(responseData);
      } else {
        throw Exception('Prediction failed');
      }
    } catch (e) {
      throw Exception('Error: $e');
    }
  }
}
```

### 3. Use in your screen
```dart
final apiService = AlzheimerAPIService();
final result = await apiService.predictAlzheimer(imageFile);

print('Has Alzheimer: ${result['prediction']['has_alzheimer']}');
print('Confidence: ${result['prediction']['confidence']}');
```

## Deployment Options

### Option 1: Local Development
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Production with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

### Option 3: Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option 4: Cloud Deployment
- **Heroku**: Use Procfile with gunicorn
- **Railway**: Connect GitHub repo
- **Render**: Deploy from GitHub
- **AWS Lambda**: Use Zappa or similar
- **Google Cloud Run**: Containerize with Docker

## Environment Variables

Create `.env` file (optional):
```
MODEL_PATH=./alzheimer_coatnet.onnx
PORT=8000
HOST=0.0.0.0
DEBUG=False
```

## Testing

### Using curl
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@/path/to/image.jpg"
```

### Using Python requests
```python
import requests

with open('brain_scan.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    print(response.json())
```

## Troubleshooting

1. **Model not found error**
   - Ensure `alzheimer_coatnet.onnx` is in the correct path
   - Update `model_path` in `main.py` if needed

2. **Port already in use**
   ```bash
   uvicorn main:app --port 8001
   ```

3. **CORS errors in Flutter**
   - CORS is already enabled for all origins
   - In production, restrict to your app's domain

4. **Image preprocessing issues**
   - Ensure image is in standard format (JPG, PNG, etc.)
   - Check model's expected input size and adjust in `model.py`

## Performance Tips

1. Use image compression before upload (reduce file size)
2. Run backend on same network as Flutter app for lower latency
3. Consider GPU acceleration if available (ONNX Runtime GPU)
4. Add caching for repeated requests

## Security Notes

- In production, remove `allow_origins=["*"]` and specify allowed domains
- Add authentication if needed
- Implement rate limiting
- Use HTTPS instead of HTTP
- Validate file size to prevent DoS attacks
