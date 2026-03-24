const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let model;

// 🎥 Start webcam
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true
    });

    video.srcObject = stream;

    return new Promise(resolve => {
        video.onloadedmetadata = () => {
            video.play();
            resolve(video);
        };
    });
}

// 🤖 Load PoseNet
async function loadModel() {
    model = await posenet.load();
    console.log("PoseNet Loaded");
}

// 🧍 Detect pose
async function detectPose() {

    const pose = await model.estimateSinglePose(video);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    pose.keypoints.forEach(point => {
        if (point.score > 0.5) {
            ctx.beginPath();
            ctx.arc(point.position.x, point.position.y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = "lime";
            ctx.fill();
        }
    });

    requestAnimationFrame(detectPose);
}

// 🚀 Main
async function main() {
    await setupCamera();
    await loadModel();
    detectPose();
}

main();