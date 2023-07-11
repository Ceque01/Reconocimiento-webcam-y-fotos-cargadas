const elVideo = document.getElementById('video');

navigator.getMedia = (navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia);

const cargarCamera = () => {
    navigator.getMedia (
        // Restricciones (contraints) *Requerido
        {
            video: true,
            audio: false
        },
        stream => elVideo.srcObject = stream,
        console.error
    );
}

Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
    faceapi.nets.ageGenderNet.loadFromUri('/models'),
    faceapi.nets.faceExpressionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.faceLandmark68TinyNet.loadFromUri('/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
    faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
]).then(cargarCamera);


elVideo.addEventListener('play', async () => {
    // Crea el canvas con los elementos del Face API
    const canvas = faceapi.createCanvasFromMedia(elVideo);
    
    document.body.append(canvas);

    // Tamaño del canvas
    const displaySize = { width: elVideo.width, height: elVideo.height };
    faceapi.matchDimensions(canvas, displaySize);

    // Carga las imágenes de las personas conocidas
    const labeledFaceDescriptors = await loadLabeledImages();

    setInterval(async () => {
        // Realiza las detecciones
        const detections = await faceapi
            .detectAllFaces(elVideo)
            .withFaceLandmarks()
            .withFaceExpressions()
            .withAgeAndGender()
            .withFaceDescriptors();

        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        // Limpia el canvas
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

        // Dibuja las líneas
        faceapi.draw.drawDetections(canvas, resizedDetections);
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
        faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

        resizedDetections.forEach(detection => {
            const { descriptor, age, gender, detection: { box } } = detection;

            const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);
            const bestMatch = faceMatcher.findBestMatch(descriptor);

            const drawBox = new faceapi.draw.DrawBox(box, {
                label: `${bestMatch.label} (${Math.round(age)} años, ${gender})`
            });
            // Dibuja el cuadro con el nombre
            drawBox.draw(canvas);
        });
    });
});

async function loadLabeledImages() {
    const labels = ['Persona1', 'Persona2']; // Nombres de las personas conocidas
    return Promise.all(
        labels.map(async label => {
            const descriptions = [];
            for (let i = 1; i <= 1; i++) { // Se recomienda cargar 3 o más imágenes por persona en caso de cargar mas de 1, cambiar el for
                const img = await faceapi.fetchImage(`fotos/${label}/${i}.jpg`); // Ruta de la imagen
                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                descriptions.push(detections.descriptor);
            }
            return new faceapi.LabeledFaceDescriptors(label, descriptions);
        })
    );
}