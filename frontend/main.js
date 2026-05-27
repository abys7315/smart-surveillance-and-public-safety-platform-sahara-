// S.A.H.A.R.A Dashboard Frontend Logic

document.addEventListener('DOMContentLoaded', () => {
  const alertsContainer = document.getElementById('alerts-container');
  const clearBtn = document.getElementById('clear-alerts');
  const wsUrl = "ws://127.0.0.1:8000/alerts";
  
  let ws;

  function connectWebSocket() {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log("Connected to S.A.H.A.R.A Backend.");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "alert") {
        addAlert(data.threat, data.timestamp);
      }
    };

    ws.onclose = () => {
      console.log("Disconnected from backend. Reconnecting in 3s...");
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
      console.error("WebSocket Error:", error);
      ws.close();
    };
  }

  function formatTime(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  function addAlert(threatType, timestamp) {
    // Remove empty state if present
    const emptyState = document.querySelector('.empty-state');
    if (emptyState) {
      emptyState.remove();
    }

    const alertEl = document.createElement('div');
    alertEl.className = 'alert-item';
    
    alertEl.innerHTML = `
      <div class="alert-header">
        <span class="alert-title">⚠ THREAT DETECTED</span>
        <span class="alert-time">${formatTime(timestamp)}</span>
      </div>
      <div class="alert-desc">${threatType}</div>
    `;

    // Prepend to top of container
    alertsContainer.prepend(alertEl);

    // Keep max 50 alerts in DOM to prevent performance issues
    if (alertsContainer.children.length > 50) {
      alertsContainer.removeChild(alertsContainer.lastChild);
    }
  }

  clearBtn.addEventListener('click', () => {
    alertsContainer.innerHTML = '<div class="empty-state">No incidents detected.</div>';
  });

  // Start connection
  connectWebSocket();
});
