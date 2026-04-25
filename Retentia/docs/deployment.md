## Deployment Guide

### Local (venv)

```
cd Cognitive_Sim
python -m pip install -r requirements/prod.txt
ENV=production python -m src.api
```

### Docker

```
cd Cognitive_Sim
docker-compose up --build
```

### Environment selection

Set `ENV` to pick an environment file:

- `ENV=development` → [`development.yaml`](Cognitive_Sim/configs/development.yaml:1)
- `ENV=testing` → [`testing.yaml`](Cognitive_Sim/configs/testing.yaml:1)
- `ENV=production` → [`production.yaml`](Cognitive_Sim/configs/production.yaml:1)

### API authentication

If `runtime.require_api_key` is `true`, send the key via:

- `x-api-key: <value>`
- `Authorization: Bearer <value>`

Example:

```
curl -H "x-api-key: change-me" http://localhost:8000/status
```
