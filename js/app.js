//Declaramos las variables
let webCamera = null;
let model = null;
let letterBuffer = [];
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
const btnLoad = document.getElementById('btnLoad');
const btnSave = document.getElementById('btnSave');
const btnTrain = document.getElementById('btnTrain');
let modelTraining = false;

//Claes de las letras del alfabeto
const classes = { 1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 13: "M", 14: "N", 15: "Ñ", 16: "O", 17: "P", 18: "Q", 19: "R", 20: "S", 21: "T", 22: "U", 23: "V", 24: "W", 25: "X", 26: "Y", 27: "Z" };

// Variable global para almacenar el modelo de Handpose
let handposeModel = null;

// Función asincrónica para cargar el modelo Handpose de TensorFlow.js.
// Una vez cargado, muestra un mensaje en la consola.
async function loadHandposeModel() {
  handposeModel = await handpose.load();
  console.log("Handpose model loaded");
}

// Función para dibujar la mano detectada por el modelo Handpose en un canvas.
// Toma las predicciones de Handpose y el contexto del canvas como argumentos.
function drawHand(handPredictions, ctx) {
  // Limpia el canvas antes de dibujar
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // Recorre cada predicción de mano detectada
  for (let i = 0; i < handPredictions.length; i++) {
    const keypoints = handPredictions[i].landmarks;

    // Dibuja puntos en cada punto clave de la mano
    for (let j = 0; j < keypoints.length; j++) {
      const [x, y, z] = keypoints[j];
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 3 * Math.PI);
      ctx.fillStyle = "red";
      ctx.fill();
    }

    // Dibuja las conexiones (huesos) entre los puntos clave
    const fingers = handPredictions[i].annotations;
    for (let finger in fingers) {
      const points = fingers[finger];
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);

      for (let j = 1; j < points.length; j++) {
        ctx.lineTo(points[j][0], points[j][1]);
      }

      ctx.strokeStyle = "blue";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}


// Función para limpiar las letras formadas
const clearWord = () => {
  letterBuffer = []; // Limpiar el arreglo de letras
  processWord(); // Actualizar la interfaz
};

// Agregar ejemplo de imagen al modelo
const addImgExample = async (classId) => {
  if (modelTraining) {
    const img = await webCamera.capture();
    const activation = model.infer(img, "conv_preds");
    classifier.addExample(activation, classId);
    img.dispose();
  } else {
    swal("¡Oops, el modelo no está en modo de entrenamiento!", " Haz clic en 'Entrenar' para activar el modo de entrenamiento.", "error");
  }
};

// Función para procesar y mostrar la palabra formada
const processWord = () => {
  const word = letterBuffer.join('');
  // Muestra la palabra en la interfaz de usuario en cada paso
  document.getElementById("result2").innerHTML = `
      <p><b>Palabra:</b> ${word}</p>
  `;
};

// Inicialización de la aplicación
const app = async () => {
  try {
    webCamera = await tf.data.webcam(webcamElement);
    model = await mobilenet.load();
    await loadHandposeModel();  // Cargar handpose

    // Crea un intervalo que se ejecuta cada 2 segundos para hacer predicciones
    const interval = setInterval(async () => {
      // Verifica si el modelo y el clasificador están cargados y listos
      if (model !== null && classifier.getNumClasses() > 0 && handposeModel !== null) {
        // Captura una imagen de la cámara web
        const img = await webCamera.capture();

        // Obtiene predicciones de mano usando el modelo Handpose
        const handPredictions = await handposeModel.estimateHands(img);

        // Procede solo si se detecta al menos una mano
        if (handPredictions.length > 0) {
          const canvas = document.getElementById('canvas');
          const ctx = canvas.getContext('2d');

          // Dibuja la mano detectada en el canvas
          drawHand(handPredictions, ctx);

          // Procede con la clasificación de la imagen capturada
          const activation = model.infer(img, "conv_preds");
          const result = await classifier.predictClass(activation);

          // Almacena y muestra la predicción de la letra
          letterBuffer.push(classes[result.label]);
          document.getElementById("result").innerHTML = `
          <p><b>Letra:</b> ${classes[result.label]}</p>
          <p><b>Probabilidad:</b> ${result.confidences[result.label]}</p>
          `;

          // Actualiza la palabra formada en la interfaz de usuario
          processWord();
        }

        // Libera los recursos de la imagen capturada
        img.dispose();
      } else {
        // Muestra un mensaje si no hay un modelo cargado
        document.getElementById("result").innerHTML = `
      <h4>¡Oops, no me hemos encontrado un modelo entrenado!</h4>
      <p>Por favor, carga un modelo entrenado o entrena uno nuevo. :)</p>
    `;
      }
    }, 2000); // Intervalo de 2 segundos

    // Nuevo botón para limpiar letras formadas
    const btnClear = document.getElementById("btnClear");
    btnClear.addEventListener("click", () => {
      clearWord();
    });

    // Limpiar el intervalo cuando se cierre la ventana
    window.addEventListener("beforeunload", () => {
      clearInterval(interval);
    });
  } catch (error) {
    console.error("Error al cargar el modelo:", error);
  }
};


function secretMessage() {
  swal({
    title: "¡Hola!",
    text: "Si eres nuestro profe, pasanos con 100 uwu",
    icon: "success",
    buttons: {
      si1: {
        text: "¡Por supuesto!",
      },
      si2: {
        text: "Acepto",
      },
      si3: {
        text: "Ya diga que sí :3",
      },
    },
  });
}

function showInstructions() {
  // Crear un elemento modal
  const modal = document.createElement('div');
  modal.classList.add('modal');

  // Contenido de las instrucciones
  const instructionsContent = `
        <h2>Instrucciones de Uso</h2>
        <p>1. Haga clic en "Entrenar" para comenzar a entrenar un nuevo modelo.</p>
        <p>2. Una vez que ha empezado el entrenamiento, vaya haciendo seña por seña
        de las letras del alfabeto mientras da clic en los botones correspondientes a cada letra para entrenar el modelo.</p>
        <p>3. Puede visualizar los resultados de predicción en el lado derecho de la pantalla.</p>
        <p>4. Es necesario dar acceso a tu cámara para el funcionamiento correcto de la web app.</p>
        <p>5. Una vez que este satisfecho con las predicciónes, guarde el modelo.</p>
        <p>6. En caso de ya tener un modelo, puedo cargarlo en "Cargar modelo".</p>
        <p>7. En caso de error, reinicia la página o comprueba tu conectividad a internet :)</p>
        <p>8. ¡Disfrute de Signa Lens! :)</p>
    `;

  modal.innerHTML = instructionsContent;

  // Agregar el modal a la página
  document.body.appendChild(modal);

  // Agregar un botón para cerrar el modal
  const closeButton = document.createElement('button');
  closeButton.innerText = 'Cerrar';
  closeButton.onclick = function () {
    document.body.removeChild(modal);
  };

  modal.appendChild(closeButton);
}

const buttons = document.getElementsByClassName('alpha-btn');
for (let i = 0; i < buttons.length; i++) {
  buttons[i].addEventListener("click", () => {
    const classId = buttons[i].getAttribute("data-position");
    addImgExample(classId);
  });
}


// Eventos de los botones (nav-bar)
btnLoad.addEventListener("click", async () => {
  const input = document.createElement("input"); // Crear un input
  input.type = "file"; // Establecer el tipo de input a file
  input.accept = ".json"; // Aceptar solo archivos JSON

  input.onchange = async () => {
    const file = input.files[0];
    if (file) {
      try {
        const jsonContent = await file.text(); // Leer el archivo
        const loadedDataset = JSON.parse(jsonContent); // Convertir a JSON
        const tensorObj = Object.entries(loadedDataset).reduce( // Convertir a tensores
          (obj, [classId, data]) => {
            obj[classId] = tf.tensor(data.data, data.shape, data.dtype);
            return obj;
          },
          {}
        );

        classifier.setClassifierDataset(tensorObj); // Cargar el dataset
      } catch (error) {
        console.error(
          "Error al cargar el modelo desde el archivo JSON:",
          error
        );
      }
      swal("¡Bien hecho!", "¡Modelo cargado exitosamente!", "success", { buttons: false, timer: 2000, });
    }
  };

  input.click();
});

// Entrenar el modelo
btnTrain.addEventListener("click", () => {
  if (!model) {
    swal("Oops", "El modelo no se ha cargado, por favor espera unos segundos", "error");
    return;
  }

  // Activar el modo de entrenamiento
  modelTraining = !modelTraining;
  btnTrain.innerText = "Entrenando...";
  btnTrain.disabled = true;
  btnSave.disabled = false;
  swal("¡Bien hecho!", "¡Comenzando entrenamiento!", "success", {
    buttons: false,
    timer: 3000,
  });

  // Añadir la clase para mostrar los botones de letras
  document.getElementById("alphabet-container").classList.toggle("alphabet-btn-visible", modelTraining);

});

// Guardar el modelo
btnSave.addEventListener("click", async () => {
  if (modelTraining) {
    // Guardar el modelo solo si está en modo de entrenamiento.
    const dataset = classifier.getClassifierDataset();

    // Ajustar la estructura del dataset
    const adjustedDataset = Object.entries(dataset).reduce(
      (obj, [classId, data]) => {
        obj[classId] = {
          data: Array.from(data.dataSync()), // Convertir a un array
          shape: data.shape,
          dtype: data.dtype,
        };
        return obj;
      },
      {}
    );

    const jsonDataset = JSON.stringify(adjustedDataset); // Convertir a JSON

    // Crear un Blob con el JSON
    const blob = new Blob([jsonDataset], { type: "application/json" });

    // Crear un enlace de descarga
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "modelo_entrenado.json";

    // Simular un clic en el enlace para iniciar la descarga
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    swal("¡Bien hecho!", "¡Modelo guardado exitosamente!", "success");
  } else {
    swal("¡Ey para!", "No hay ningun modelo entrenado. Primero entrena uno y después guardalo.", "error");
  }


  // Desactivar el modo de entrenamiento.
  modelTraining = false;
  btnTrain.innerText = "Entrenar";
  btnTrain.disabled = false;
  btnSave.disabled = true;
});


app();