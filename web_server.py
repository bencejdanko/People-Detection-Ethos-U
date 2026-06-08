import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from datetime import datetime
from collections import deque

from pydantic import BaseModel
from fastapi import HTTPException, status

app = FastAPI()

# Cache the last 20 log entries (newest first)
logs = deque(maxlen=20)

class CountPayload(BaseModel):
    count: int
    token: str

@app.post("/count")
async def update_count(payload: CountPayload):
    # Check authorization token
    if payload.token != "your-secure-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid security token"
        )
        
    count = payload.count
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Store at the left side so newest are first
    logs.appendleft({"time": now, "count": count})
    print(f"[{now}] Received count: {count}")
    return {"status": "ok", "received": count}

@app.get("/", response_class=HTMLResponse)
async def get_logs():
    html_content = """
    <html>
    <head>
        <title>People Counting Logs</title>
        <meta http-equiv="refresh" content="2">
    </head>
    <body>
        <h1>People Counting Logs</h1>
        <p>Last 20 updates (auto-refreshes every 2 seconds):</p>
        <table border="1" cellpadding="5" cellspacing="0">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody>
    """
    for entry in logs:
        html_content += f"""
                <tr>
                    <td>{entry['time']}</td>
                    <td>{entry['count']}</td>
                </tr>
        """
    if not logs:
        html_content += """
                <tr>
                    <td colspan="2" style="text-align:center;">No records received yet</td>
                </tr>
        """
        
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    print("Starting minimal web server on http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
