import asyncio
import websockets
import json
import base64

async def send_video():
    # Path to your video file
    video_path = "hello_1.mp4"  # <- replace with your file

    # Read and encode the video as base64
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")

    # WebSocket URL
    ws_url = "ws://127.0.0.1:8000/video"

    async with websockets.connect(ws_url) as ws:
        # Send JSON with video
        msg = {
            "video": video_b64,
            "format": "mp4"
        }
        await ws.send(json.dumps(msg))

        # Wait for response
        response = await ws.recv()
        print("Response:", response)

# Run the async function
asyncio.run(send_video())
