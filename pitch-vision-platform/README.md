# Pitch Vision platform

This folder contains the Express API and React interface for the Liverpool vs Real Madrid five-minute tactical analysis. The one public project route is:

`/pitch-vision/tactics/liverpool-madrid-five`

## Run locally

```powershell
npm ci
npm run api

cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/pitch-vision/tactics/liverpool-madrid-five`.

The checked-in `data/tactics/liverpool-madrid-five/tactical.json` drives the whole interface. Put the locally licensed excerpt at `data/tactics/liverpool-madrid-five/video.mp4` to enable synchronized footage; it is deliberately excluded from Git.

## Validate

```powershell
npm test

cd frontend
npm run build
```

The API can read from the local data folder or object storage. See `infra/template.yaml` for the existing AWS deployment template.
