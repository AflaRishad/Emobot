let video = document.getElementById("video");

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
});

function detectEmotion(){
    let canvas = document.createElement("canvas");
    canvas.width = 300;
    canvas.height = 300;
    let ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, 300, 300);

    let dataURL = canvas.toDataURL("image/png");

    fetch("/predict", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({image: dataURL})
    })
    .then(res => res.json())
    .then(data => {
        if(data.error){
            document.getElementById("result").innerText = "Error: " + data.error;
            return;
        }

        document.getElementById("result").innerText =
            "Emotion Detected: " + data.emotion;

        setTimeout(()=>{
            window.location.href="/chatbot";
        }, 1200);
    });
}