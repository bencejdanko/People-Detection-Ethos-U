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

@app.get("/api/logs")
async def get_api_logs():
    return list(logs)

@app.get("/", response_class=HTMLResponse)
async def get_logs():
    html_content = """
    <html>
    <head>
        <title>M55M1D Logs</title>
    </head>
    <body>
        <h1>M55M1D Edge-AI People Counting Logs</h1>
        <table border="1" cellpadding="5" cellspacing="0">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody id="log-rows">
                <tr>
                    <td colspan="2" style="text-align:center;">Loading logs...</td>
                </tr>
            </tbody>
        </table>

        <script>
            async function updateLogs() {
                try {
                    const response = await fetch('/api/logs');
                    const data = await response.json();
                    const tbody = document.getElementById('log-rows');
                    
                    if (data.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="2" style="text-align:center;">No records received yet</td></tr>';
                        return;
                    }
                    
                    let html = '';
                    for (const entry of data) {
                        html += `<tr><td>${entry.time}</td><td>${entry.count}</td></tr>`;
                    }
                    tbody.innerHTML = html;
                } catch (err) {
                    console.error('Failed to fetch logs:', err);
                }
            }

            // Poll every 1 second reactively without reloading the page
            setInterval(updateLogs, 1000);
            updateLogs();
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    print("Starting minimal web server on http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
