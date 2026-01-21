const video = document.getElementById('videoElement');
const canvas = document.getElementById('canvasElement');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const countBadge = document.getElementById('count-badge');

if (canvas) {
    const ctx = canvas.getContext('2d');
    let stream = null;
    let ws = null;
    let isRunning = false;
    let lastFrameTime = 0;

    // Set canvas size
    canvas.width = 640;
    canvas.height = 480;

    if (startBtn) startBtn.addEventListener('click', startCamera);
    if (stopBtn) stopBtn.addEventListener('click', stopCamera);

    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
            video.srcObject = stream;
            
            // Connect WebSocket
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/stream`);
            
            ws.onopen = () => {
                console.log("WebSocket Connected");
                isRunning = true;
                startBtn.disabled = true;
                stopBtn.disabled = false;
                requestAnimationFrame(sendFrame);
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                drawResults(data.faces);
                if (countBadge) countBadge.textContent = data.attendance_count;
            };
            
            ws.onclose = () => {
                console.log("WebSocket Disconnected");
                stopCamera();
            };

        } catch (err) {
            console.error("Error accessing camera:", err);
            alert("无法访问摄像头，请确保使用HTTPS或localhost。");
        }
    }

    function stopCamera() {
        isRunning = false;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        if (ws) {
            ws.close();
            ws = null;
        }
        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    function sendFrame(timestamp) {
        if (!isRunning || !ws || ws.readyState !== WebSocket.OPEN) return;
        
        // Throttle FPS to ~5-10 to reduce load
        if (timestamp - lastFrameTime < 200) { // 200ms = 5fps
            requestAnimationFrame(sendFrame);
            return;
        }
        lastFrameTime = timestamp;

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 640;
        tempCanvas.height = 480;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0, 640, 480);
        
        const dataURL = tempCanvas.toDataURL('image/jpeg', 0.6); // Quality 0.6
        ws.send(dataURL);
        
        requestAnimationFrame(sendFrame);
    }

    function drawResults(faces) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Since we enforced fixed size 640x480 in HTML/CSS, we don't need dynamic scaling.
        // Backend returns coordinates based on 640x480.
        // Video is displayed at 640x480.
        // Canvas is 640x480.
        
        faces.forEach(face => {
            const [x1, y1, x2, y2] = face.box;
            
            // Direct mapping
            const sx1 = x1;
            const sy1 = y1;
            const sx2 = x2;
            const sy2 = y2;
            
            const name = face.name;
            const score = face.score;
            
            ctx.strokeStyle = name === 'Unknown' ? 'red' : '#0d6efd';
            ctx.lineWidth = 3;
            ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
            
            ctx.fillStyle = name === 'Unknown' ? 'red' : '#0d6efd';
            ctx.font = 'bold 18px Arial';
            ctx.fillText(`${name} (${score.toFixed(2)})`, sx1, sy1 - 10);
        });
    }
}

