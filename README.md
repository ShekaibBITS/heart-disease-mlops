# Heart Disease MLOps – Docker & Kubernetes Deployment Guide

This document explains how to **set up, run, and validate** the Heart Disease prediction API using:

* **Docker (local container execution)**
* **Kubernetes (Minikube – production-style deployment)**

---

## PART A: Docker-Based Execution

### 1. Prerequisites

Ensure the following is installed:

* Docker

Verify installation:

```bash
docker version
```

---

### 2. Project Setup

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd heart-disease-mlops
```

Verify required files exist:

```text
deployment/docker/Dockerfile
requirements.txt
api/main.py
models/
```

---

### 3. Build Docker Image

Build the Docker image for the model-serving API:

```bash
docker build -t heart-api -f deployment/docker/Dockerfile .
```

Confirm image creation:

```bash
docker images | grep heart-api
```

---

### 4. Run Docker Container

Run the Docker container locally:

```bash
docker run -p 8000:8000 heart-api
```

The API will start at:

```text
http://localhost:8000
```

---

### 5. Docker API Validation

#### Health Check

```bash
curl http://localhost:8000/health
```

Expected output:

```json
{"status":"ok"}
```

#### Prediction Endpoint Validation

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,
     "fbs":0,"restecg":1,"thalach":168,"exang":0,
     "oldpeak":1.0,"slope":2,"ca":0,"thal":2}'
```

Expected response:

```json
{
  "prediction": 1,
  "confidence": 0.88
}
```

#### Swagger UI (Optional)

Open in browser:

```text
http://localhost:8000/docs
```

---

### 6. Stop Docker Container

Stop execution using:

```text
CTRL + C
```

---

## PART B: Kubernetes (Minikube) Deployment

### 7. Prerequisites

Ensure the following tools are installed:

* Docker
* kubectl
* Minikube

Verify installation:

```bash
docker version
kubectl version --client
minikube version
```

---

### 8. Start Local Kubernetes Cluster

Start Minikube:

```bash
minikube start
```

Verify cluster status:

```bash
kubectl get nodes
```

The node status should be **Ready**.

---

### 9. Enable Ingress Controller (Optional)

Enable NGINX Ingress:

```bash
minikube addons enable ingress
```

Run the Minikube tunnel in a separate terminal (required for Ingress):

```bash
minikube tunnel
```

---

### 10. Deploy Application to Kubernetes

Verify file structure:

```text
deployment/
 └── k8s/
     ├── deployment.yaml
     ├── service.yaml
     ├── ingress.yaml
     └── kustomization.yaml
```

Apply Kubernetes manifests:

```bash
kubectl apply -k deployment/k8s
```

Verify deployment:

```bash
kubectl get pods -n heart-mlops
kubectl get svc -n heart-mlops
```

Wait until all pods are in **Running** state.

---

### 11. Access Services (Port Forwarding)

Run each command in a separate terminal.

**Heart Disease Prediction API**

```bash
kubectl port-forward -n heart-mlops svc/heart-api 8000:8000
```

Access:

```text
http://localhost:8000/docs
```

**MLflow**

```bash
kubectl port-forward -n heart-mlops svc/mlflow 5000:5000
```

Access:

```text
http://localhost:5000
```

**MinIO Console**

```bash
kubectl port-forward -n heart-mlops svc/minio 9001:9001
```

Access:

```text
http://localhost:9001
```

**Prometheus**

```bash
kubectl port-forward -n heart-mlops svc/prometheus 9090:9090
```

**Grafana**

```bash
kubectl port-forward -n heart-mlops svc/grafana 3000:3000
```

Access:

```text
http://localhost:3000
```

Default credentials:

```text
admin / admin
```

---

### 12. Kubernetes API Validation

```bash
curl http://localhost:8000/health
```

Prediction request:

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 0,
  "thal": 2
}'
```

---

### 13. Ingress Access (Optional)

Add host entry:

```text
127.0.0.1 heart.local
```

Verify ingress:

```bash
kubectl get ingress -n heart-mlops
```

Access:

* Heart API: [http://heart.local/docs](http://heart.local/docs)
* MLflow: [http://heart.local/mlflow](http://heart.local/mlflow)
* Grafana: [http://heart.local/grafana](http://heart.local/grafana)
* Prometheus: [http://heart.local/prometheus](http://heart.local/prometheus)

---

### 14. Assignment Evidence Checklist

Include screenshots of:

* Docker build and run output
* `kubectl get pods -n heart-mlops`
* `kubectl get svc -n heart-mlops`
* Swagger UI (`/docs`)
* `/predict` API response
* MLflow UI
* Grafana dashboard

---

### 15. Cleanup

Delete Kubernetes resources:

```bash
kubectl delete -k deployment/k8s
```

Stop Minikube:

```bash
minikube stop
```
