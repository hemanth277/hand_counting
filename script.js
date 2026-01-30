const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const fingerCountElement = document.getElementById('finger-count');
const statusElement = document.getElementById('status');
const fpsElement = document.getElementById('fps');
const volumeBar = document.getElementById('volume-bar');
const volumeValue = document.getElementById('volume-value');
const brightnessBar = document.getElementById('brightness-bar');
const brightnessValue = document.getElementById('brightness-value');

let currentVolume = 50;
let currentBrightness = 50;

let lastTime = 0;

function onResults(results) {
    // Update Performance
    const now = performance.now();
    const fps = Math.round(1000 / (now - lastTime));
    lastTime = now;
    fpsElement.innerText = fps;
    statusElement.innerText = "Tracking Active";

    // Draw
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    let totalFingers = 0;
    const handGestures = [];

    if (results.multiHandLandmarks && results.multiHandedness) {
        for (let index = 0; index < results.multiHandLandmarks.length; index++) {
            const landmarks = results.multiHandLandmarks[index];
            const classification = results.multiHandedness[index];
            const isRightHand = classification.label === 'Right';

            // Draw Landmarks
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#ffffff', lineWidth: 2 });
            drawLandmarks(canvasCtx, landmarks, { color: '#6366f1', lineWidth: 1, radius: 3 });

            // Finger counting logic
            const fingers = [];

            // Thumb
            // Tip (4) vs IP (3) vs MCP (2)
            // Note: MediaPipe JS labels might be mirrored depending on camera
            if (isRightHand) {
                if (landmarks[4].x < landmarks[2].x) fingers.push(1);
                else fingers.push(0);
            } else {
                if (landmarks[4].x > landmarks[2].x) fingers.push(1);
                else fingers.push(0);
            }

            // Other 4 fingers: Tip (8,12,16,20) vs PIP (6,10,14,18)
            const tipIds = [8, 12, 16, 20];
            for (const tipId of tipIds) {
                if (landmarks[tipId].y < landmarks[tipId - 2].y) {
                    fingers.push(1);
                } else {
                    fingers.push(0);
                }
            }

            totalFingers += fingers.filter(f => f === 1).length;

            // Gesture Logic
            const thumbUp = landmarks[4].y < landmarks[2].y - 0.05;
            const thumbDown = landmarks[4].y > landmarks[2].y + 0.05;
            const othersClosed = [8, 12, 16, 20].every(id => landmarks[id].y > landmarks[id - 2].y);

            if (othersClosed) {
                handGestures.push({ up: thumbUp, down: thumbDown });
            }
        }

        // Apply Gestures
        if (handGestures.length === 1) {
            const { up, down } = handGestures[0];
            if (up) currentVolume = Math.min(100, currentVolume + 2);
            else if (down) currentVolume = Math.max(0, currentVolume - 2);
        } else if (handGestures.length === 2) {
            const g1 = handGestures[0];
            const g2 = handGestures[1];
            if (g1.up && g2.up) currentBrightness = Math.min(100, currentBrightness + 2);
            else if (g1.down && g2.down) currentBrightness = Math.max(0, currentBrightness - 2);
        }

        // Update UI
        volumeBar.style.width = `${currentVolume}%`;
        volumeValue.innerText = `${Math.round(currentVolume)}%`;
        brightnessBar.style.width = `${currentBrightness}%`;
        brightnessValue.innerText = `${Math.round(currentBrightness)}%`;
    }

    fingerCountElement.innerText = totalFingers;
    canvasCtx.restore();
}

const hands = new Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});

hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

hands.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({ image: videoElement });
    },
    width: 640,
    height: 480
});

camera.start().then(() => {
    statusElement.innerText = "Camera Started";
}).catch(err => {
    statusElement.innerText = "Error: " + err;
    console.error(err);
});

// Resize handler
function resize() {
    canvasElement.width = canvasElement.clientWidth;
    canvasElement.height = canvasElement.clientHeight;
}
window.addEventListener('resize', resize);
resize();
