# Face Swap Studio — Render

## Deploy
1. Create a GitHub repository and upload all files/folders from this ZIP.
2. In Render, create a new **Web Service** from the GitHub repository.
3. Render can use the included `render.yaml`, or use:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn server:app`
4. Deploy.

## Important
The frontend calls `/api/swap`. The included backend accepts both images and returns the target image as a placeholder result.

It does **not** perform a real face swap yet. To make it a real face-swap application, connect `/api/swap` to a face-swap model or an external image-generation/face-swap API.

For production, add file-size limits, authentication/rate limiting, temporary-file cleanup, and appropriate consent/privacy handling for uploaded faces.
