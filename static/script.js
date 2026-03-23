const video = document.getElementById("video")

let interval = null
let lastGesture = ""
let lastTime = 0

navigator.mediaDevices.getUserMedia({video:true})
.then(stream=>{
    video.srcObject = stream
})

function capture(){

    const canvas = document.createElement("canvas")
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    const ctx = canvas.getContext("2d")
    ctx.drawImage(video,0,0)

    const image = canvas.toDataURL("image/png")

    fetch("/predict",{
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({image:image})
    })
    .then(res=>res.json())
    .then(data=>{
        if(data.prediction){
            handleGesture(data.prediction)
        }
    })
}

function handleGesture(pred){

    const now = Date.now()

    if(pred === lastGesture && (now - lastTime < 3000)){
        return
    }

    lastGesture = pred
    lastTime = now

    let text = ""

    if(pred == '1'){
        text = "Halo! " + new Date().toLocaleTimeString()
    }
    else if(pred == '2'){
        text = "Membuka ChatGPT..."
        window.open("https://chat.openai.com","_blank")
    }
    else if(pred == '3'){
        text = "Membuka YouTube..."
        window.open("https://youtube.com","_blank")
    }
    else if(pred == '4'){
        text = "Membuka Instagram..."
        window.open("https://instagram.com","_blank")
    }
    else{
        text = "Jumlah jari: " + pred
    }

    document.getElementById("result").innerText = text
}

function startAuto(){
    if(!interval){
        interval = setInterval(capture, 2000)
    }
}

function stopAuto(){
    clearInterval(interval)
    interval = null
}
